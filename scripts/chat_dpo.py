"""
Direct Preference Optimization (DPO) training for chat models.

DPO trains language models directly on preference data without requiring
a separate reward model or sampling during training.

Run on 1 GPU:
python -m scripts.chat_dpo

Run on 8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_dpo -- --run=dpo_run
"""

import os
import copy
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.dpo import dpo_loss, concatenated_forward
from tasks.preferences import HHRLHFPreferences, SyntheticPreferences
from tasks.gsm8k import GSM8K

run = "dummy"
source = "sft"
dtype = "bfloat16"
device_batch_size = 4
total_batch_size = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
beta = 0.1
label_smoothing = 0.0
num_epochs = 1
max_seq_len = 2048
eval_every = 100
save_every = 200
use_synthetic_prefs = False
max_train_examples = 10000
max_val_examples = 500
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-dpo", name=run, config=user_config)

model, tokenizer, meta = load_model(source, device, phase="train")
orig_model = model

reference_model = copy.deepcopy(model)
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False

print0(f"Loaded policy and reference models from {source}")

if use_synthetic_prefs:
    print0("Using synthetic preferences from GSM8K")
    from nanochat.engine import Engine
    engine = Engine(model, tokenizer)
    base_task = GSM8K(subset="main", split="train")
    train_dataset = SyntheticPreferences(base_task)
    with torch.no_grad():
        train_dataset.generate_from_model(model, tokenizer, engine, max_examples=max_train_examples)
    val_dataset = train_dataset
else:
    print0("Using HH-RLHF preference dataset")
    train_dataset = HHRLHFPreferences(split="train", max_examples=max_train_examples)
    val_dataset = HHRLHFPreferences(split="test", max_examples=max_val_examples)

print0(f"Training examples: {len(train_dataset)}")
print0(f"Validation examples: {len(val_dataset)}")

def collate_preference_batch(examples, tokenizer, max_seq_len, device):
    """
    Collate a batch of preference examples into tensors.
    
    Returns a dict with:
    - chosen_input_ids: (B, T)
    - chosen_labels: (B, T)
    - rejected_input_ids: (B, T)
    - rejected_labels: (B, T)
    """
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    
    chosen_sequences = []
    rejected_sequences = []
    
    for ex in examples:
        if isinstance(ex['prompt'], str):
            prompt_text = ex['prompt']
        else:
            prompt_text = ""
        
        chosen_text = ex['chosen']
        rejected_text = ex['rejected']
        
        if prompt_text:
            chosen_full = f"{prompt_text}\n\n{chosen_text}"
            rejected_full = f"{prompt_text}\n\n{rejected_text}"
        else:
            chosen_full = chosen_text
            rejected_full = rejected_text
        
        chosen_ids = tokenizer.encode(chosen_full, prepend="<|bos|>")
        rejected_ids = tokenizer.encode(rejected_full, prepend="<|bos|>")
        
        chosen_sequences.append(chosen_ids)
        rejected_sequences.append(rejected_ids)
    
    def pad_sequence(sequences):
        max_len = min(max(len(s) for s in sequences), max_seq_len)
        
        input_ids_list = []
        labels_list = []
        
        for seq in sequences:
            seq = seq[:max_len]
            
            padding_length = max_len - len(seq)
            
            input_ids = seq[:-1] + [pad_token_id] * padding_length
            labels = seq[1:] + [-1] * padding_length
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        return (
            torch.tensor(input_ids_list, dtype=torch.long, device=device),
            torch.tensor(labels_list, dtype=torch.long, device=device)
        )
    
    chosen_input_ids, chosen_labels = pad_sequence(chosen_sequences)
    rejected_input_ids, rejected_labels = pad_sequence(rejected_sequences)
    
    return {
        'chosen_input_ids': chosen_input_ids,
        'chosen_labels': chosen_labels,
        'rejected_input_ids': rejected_input_ids,
        'rejected_labels': rejected_labels
    }

def dpo_data_generator(dataset, batch_size, tokenizer, max_seq_len, device):
    """Generate batches of preference pairs."""
    indices = list(range(ddp_rank, len(dataset), ddp_world_size))
    
    while True:
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            examples = [dataset[idx] for idx in batch_indices]
            
            if len(examples) < batch_size:
                continue
            
            batch = collate_preference_batch(examples, tokenizer, max_seq_len, device)
            yield batch

examples_per_step = total_batch_size
assert examples_per_step % (device_batch_size * ddp_world_size) == 0
grad_accum_steps = examples_per_step // (device_batch_size * ddp_world_size)

print0(f"Device batch size: {device_batch_size}")
print0(f"Total batch size: {total_batch_size}")
print0(f"Gradient accumulation steps: {grad_accum_steps}")

num_iterations = (len(train_dataset) // total_batch_size) * num_epochs
print0(f"Number of training iterations: {num_iterations}")

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

def get_lr_multiplier(it):
    return 1.0 - it / num_iterations

train_loader = dpo_data_generator(train_dataset, device_batch_size, tokenizer, max_seq_len, device)
val_loader_fn = lambda: dpo_data_generator(val_dataset, device_batch_size, tokenizer, max_seq_len, device)

for step in range(num_iterations):
    last_step = step == num_iterations - 1
    
    if step % eval_every == 0 or last_step:
        model.eval()
        reference_model.eval()
        
        val_loader = val_loader_fn()
        val_losses = []
        val_rewards_chosen = []
        val_rewards_rejected = []
        val_reward_accs = []
        
        eval_steps = min(50, len(val_dataset) // (device_batch_size * ddp_world_size))
        
        with torch.no_grad():
            for _ in range(eval_steps):
                batch = next(val_loader)
                
                with autocast_ctx:
                    policy_chosen_logps, policy_rejected_logps = concatenated_forward(model, batch, autocast_ctx)
                    ref_chosen_logps, ref_rejected_logps = concatenated_forward(reference_model, batch, autocast_ctx)
                
                loss, chosen_rewards, rejected_rewards, reward_acc = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=beta,
                    label_smoothing=label_smoothing,
                )
                
                val_losses.append(loss.item())
                val_rewards_chosen.append(chosen_rewards.item())
                val_rewards_rejected.append(rejected_rewards.item())
                val_reward_accs.append(reward_acc.item())
        
        val_loss = sum(val_losses) / len(val_losses)
        val_reward_chosen = sum(val_rewards_chosen) / len(val_rewards_chosen)
        val_reward_rejected = sum(val_rewards_rejected) / len(val_rewards_rejected)
        val_reward_acc = sum(val_reward_accs) / len(val_reward_accs)
        
        if ddp:
            metrics = torch.tensor([val_loss, val_reward_chosen, val_reward_rejected, val_reward_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            val_loss, val_reward_chosen, val_reward_rejected, val_reward_acc = metrics.tolist()
        
        print0(f"Step {step:05d} | Val Loss: {val_loss:.4f} | Chosen Reward: {val_reward_chosen:.4f} | Rejected Reward: {val_reward_rejected:.4f} | Reward Acc: {val_reward_acc:.4f}")
        
        wandb_run.log({
            "step": step,
            "val/loss": val_loss,
            "val/reward_chosen": val_reward_chosen,
            "val/reward_rejected": val_reward_rejected,
            "val/reward_accuracy": val_reward_acc,
        })
        
        model.train()
    
    if last_step:
        break
    
    model.train()
    
    total_loss = 0.0
    total_chosen_rewards = 0.0
    total_rejected_rewards = 0.0
    total_reward_accs = 0.0
    
    for micro_step in range(grad_accum_steps):
        batch = next(train_loader)
        
        with autocast_ctx:
            policy_chosen_logps, policy_rejected_logps = concatenated_forward(model, batch, autocast_ctx)
        
        with torch.no_grad():
            with autocast_ctx:
                ref_chosen_logps, ref_rejected_logps = concatenated_forward(reference_model, batch, autocast_ctx)
        
        loss, chosen_rewards, rejected_rewards, reward_acc = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )
        
        loss = loss / grad_accum_steps
        loss.backward()
        
        total_loss += loss.item()
        total_chosen_rewards += chosen_rewards.item() / grad_accum_steps
        total_rejected_rewards += rejected_rewards.item() / grad_accum_steps
        total_reward_accs += reward_acc.item() / grad_accum_steps
    
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    for opt in optimizers:
        opt.step()
    
    model.zero_grad(set_to_none=True)
    
    if ddp:
        metrics = torch.tensor([total_loss, total_chosen_rewards, total_rejected_rewards, total_reward_accs], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
        total_loss, total_chosen_rewards, total_rejected_rewards, total_reward_accs = metrics.tolist()
    
    print0(f"Step {step:05d}/{num_iterations:05d} | Loss: {total_loss:.4f} | Chosen: {total_chosen_rewards:.4f} | Rejected: {total_rejected_rewards:.4f} | Acc: {total_reward_accs:.4f} | LR: {lrm:.4f}")
    
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": total_loss,
            "train/reward_chosen": total_chosen_rewards,
            "train/reward_rejected": total_rejected_rewards,
            "train/reward_accuracy": total_reward_accs,
            "train/lrm": lrm,
        })
    
    if master_process and ((step > 0 and step % save_every == 0) or last_step):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatdpo_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            None,
            {
                "step": step,
                "val_loss": val_loss,
                "val_reward_accuracy": val_reward_acc,
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

from nanochat.report import get_report
get_report().log(section="Chat DPO", data=[
    user_config,
    {
        "Number of iterations": num_iterations,
        "Final val loss": val_loss,
        "Final reward accuracy": val_reward_acc,
    }
])

wandb_run.finish()
compute_cleanup()
