## Direct Preference Optimization (DPO) Implementation

This document describes the DPO (Direct Preference Optimization) algorithm implementation for nanochat. DPO is a method for training language models using human preferences without requiring a separate reward model or online sampling during training.

## Overview

Direct Preference Optimization simplifies RLHF (Reinforcement Learning from Human Feedback) by:
- **No reward model needed**: Directly optimizes the policy using preference pairs
- **No online sampling**: Works with static preference datasets
- **Stable training**: Avoids the instability of actor-critic methods like PPO
- **Simple implementation**: Just requires preference pairs (chosen vs rejected responses)

The key insight is that we can derive an implicit reward function from the policy and reference models, then optimize directly for preferences rather than maximizing a separate reward.

## Algorithm

DPO optimizes the following objective:

```
L_DPO = -E[(log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))]
```

Where:
- `π_θ`: Policy model being trained
- `π_ref`: Reference model (frozen copy of initial policy)
- `y_w`: Chosen (preferred) response
- `y_l`: Rejected (less preferred) response
- `x`: Prompt/context
- `β`: Temperature parameter controlling KL divergence from reference
- `σ`: Sigmoid function

## Implementation Components

### 1. DPO Loss Function (`nanochat/dpo.py`)

Core loss computation with support for:
- **Standard DPO loss**: Binary preference optimization
- **Label smoothing**: Optional regularization technique
- **Implicit rewards**: Computes interpretable reward values
- **Accuracy metrics**: Tracks when chosen > rejected

```python
from nanochat.dpo import dpo_loss

loss, chosen_rewards, rejected_rewards, reward_acc = dpo_loss(
    policy_chosen_logps,      # Log probs from policy model
    policy_rejected_logps,
    reference_chosen_logps,   # Log probs from reference model
    reference_rejected_logps,
    beta=0.1,                 # KL penalty coefficient
    label_smoothing=0.0       # Optional smoothing
)
```

### 2. Preference Datasets (`tasks/preferences.py`)

Three types of preference datasets:

#### HH-RLHF (Anthropic)
Human preferences from Anthropic's Helpful and Harmless dataset:
```python
from tasks.preferences import HHRLHFPreferences

dataset = HHRLHFPreferences(split="train", max_examples=10000)
```

#### UltraFeedback
High-quality preference data with ratings:
```python
from tasks.preferences import UltraFeedbackPreferences

dataset = UltraFeedbackPreferences(split="train", max_examples=10000)
```

#### Synthetic Preferences
Generate preferences from task performance (e.g., GSM8K correct/incorrect):
```python
from tasks.preferences import SyntheticPreferences
from tasks.gsm8k import GSM8K

base_task = GSM8K(subset="main", split="train")
dataset = SyntheticPreferences(base_task)
dataset.generate_from_model(model, tokenizer, engine, max_examples=1000)
```

### 3. Training Script (`scripts/chat_dpo.py`)

Complete DPO training pipeline:

**Key Features:**
- Reference model management (frozen copy of policy)
- Efficient concatenated forward passes
- Gradient accumulation for large batch sizes
- Distributed training support (DDP)
- Comprehensive logging and checkpointing

## Usage

### Basic Training

```bash
# Single GPU
python -m scripts.chat_dpo

# 8 GPUs with custom parameters
torchrun --standalone --nproc_per_node=8 -m scripts.chat_dpo -- \
    --run=dpo_experiment \
    --source=sft \
    --beta=0.1 \
    --device_batch_size=4 \
    --total_batch_size=32
```

### Configuration Parameters

#### Model Loading
- `source`: Which checkpoint to load (`"mid"` or `"sft"`, default: `"sft"`)
- `dtype`: Precision (`"bfloat16"` or `"float32"`, default: `"bfloat16"`)

#### Batch Sizes
- `device_batch_size`: Per-GPU batch size (default: `4`)
- `total_batch_size`: Total effective batch size (default: `32`)
- Gradient accumulation automatically computed

#### DPO Hyperparameters
- `beta`: KL penalty coefficient (default: `0.1`)
  - Higher β → Stay closer to reference model
  - Lower β → More aggressive optimization
- `label_smoothing`: Label smoothing factor (default: `0.0`)
  - Helps prevent overconfidence
  - Typical range: 0.0 - 0.1

#### Optimizer Settings
- `unembedding_lr`: Learning rate for output head (default: `0.004`)
- `embedding_lr`: Learning rate for embeddings (default: `0.2`)
- `matrix_lr`: Learning rate for transformer weights (default: `0.02`)
- `weight_decay`: L2 regularization (default: `0.0`)
- `init_lr_frac`: Initial LR multiplier (default: `0.02`)

#### Training Duration
- `num_epochs`: Number of passes through data (default: `1`)
- `max_train_examples`: Limit training data size (default: `10000`)
- `max_val_examples`: Limit validation data size (default: `500`)

#### Dataset Selection
- `use_synthetic_prefs`: Use synthetic preferences from GSM8K (default: `False`)
  - If `True`: Generates preferences by sampling from model
  - If `False`: Uses HH-RLHF preference dataset

#### Logging & Checkpointing
- `eval_every`: Evaluate every N steps (default: `100`)
- `save_every`: Save checkpoint every N steps (default: `200`)
- `run`: Wandb run name (default: `"dummy"` for no logging)

### Example Configurations

#### Conservative Fine-tuning
Stay close to reference model:
```bash
python -m scripts.chat_dpo -- --beta=0.5 --init_lr_frac=0.01
```

#### Aggressive Optimization
More deviation from reference:
```bash
python -m scripts.chat_dpo -- --beta=0.05 --init_lr_frac=0.05
```

#### Large-scale Training
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_dpo -- \
    --device_batch_size=8 \
    --total_batch_size=128 \
    --max_train_examples=50000 \
    --num_epochs=3
```

## Training Pipeline

### 1. Initialization
```python
# Load policy model
model, tokenizer, meta = load_model("sft", device, phase="train")

# Create reference model (frozen copy)
reference_model = copy.deepcopy(model)
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False
```

### 2. Data Loading
```python
# Load preference dataset
train_dataset = HHRLHFPreferences(split="train", max_examples=10000)
val_dataset = HHRLHFPreferences(split="test", max_examples=500)

# Create data generator
train_loader = dpo_data_generator(
    train_dataset, 
    device_batch_size, 
    tokenizer, 
    max_seq_len, 
    device
)
```

### 3. Training Step
```python
# Forward pass through both models
policy_chosen_logps, policy_rejected_logps = concatenated_forward(
    model, batch, autocast_ctx
)

with torch.no_grad():
    ref_chosen_logps, ref_rejected_logps = concatenated_forward(
        reference_model, batch, autocast_ctx
    )

# Compute DPO loss
loss, chosen_rewards, rejected_rewards, reward_acc = dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1
)

# Backward and optimize
loss.backward()
optimizer.step()
```

### 4. Monitoring
Key metrics logged to Wandb:
- `train/loss`: DPO loss value
- `train/reward_chosen`: Implicit reward for chosen responses
- `train/reward_rejected`: Implicit reward for rejected responses
- `train/reward_accuracy`: % where chosen > rejected
- `val/*`: Same metrics on validation set

## Implementation Details

### Efficient Concatenated Forward

DPO requires computing log probabilities for both chosen and rejected responses. We concatenate them for efficiency:

```python
def concatenated_forward(model, batch, autocast_ctx=None):
    # Concatenate chosen and rejected
    concatenated_input_ids = torch.cat([
        batch['chosen_input_ids'],
        batch['rejected_input_ids']
    ], dim=0)
    
    # Single forward pass
    all_logprobs = compute_sequence_logprobs(
        model, 
        concatenated_input_ids, 
        concatenated_labels,
        autocast_ctx
    )
    
    # Split results
    batch_size = batch['chosen_input_ids'].shape[0]
    chosen_logprobs = all_logprobs[:batch_size]
    rejected_logprobs = all_logprobs[batch_size:]
    
    return chosen_logprobs, rejected_logprobs
```

This approach:
- Reduces forward passes from 4 to 2 per batch
- Maintains numerical equivalence
- Improves training speed by ~2x

### Memory Management

Reference model memory optimization:
```python
# Reference model is eval-only, no gradients needed
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False

# Use no_grad context for reference forward
with torch.no_grad():
    ref_logps = concatenated_forward(reference_model, batch)
```

### Gradient Accumulation

For large effective batch sizes:
```python
total_batch_size = 128
device_batch_size = 4
ddp_world_size = 8

grad_accum_steps = total_batch_size // (device_batch_size * ddp_world_size)
# = 128 // (4 * 8) = 4 gradient accumulation steps

for micro_step in range(grad_accum_steps):
    batch = next(train_loader)
    loss = compute_dpo_loss(batch) / grad_accum_steps
    loss.backward()  # Accumulate gradients

optimizer.step()  # Update once after accumulation
```

## Comparison with Other Methods

### vs. PPO (Proximal Policy Optimization)
- **DPO**: Simpler, more stable, no reward model, offline training
- **PPO**: More complex, requires online sampling, separate critic

### vs. RLHF with Reward Models
- **DPO**: Direct policy optimization, no reward model training
- **RLHF**: Two-stage (reward model → policy), more components

### vs. SFT (Supervised Fine-tuning)
- **DPO**: Optimizes for preferences, better alignment
- **SFT**: Maximizes likelihood, may not capture preferences

## Best Practices

### Choosing Beta
- Start with `β=0.1` (standard value)
- Increase β if model deviates too much from reference
- Decrease β if not learning fast enough
- Monitor KL divergence via implicit rewards

### Dataset Quality
- Prefer clear preference signals (obvious winner)
- Remove ambiguous pairs where chosen ≈ rejected
- Balance dataset (equal distribution of preference types)
- Use diverse prompts to avoid overfitting

### Learning Rate
- Use lower LR than SFT (typically 0.01-0.05× base LR)
- DPO is more sensitive to LR than SFT
- Monitor reward accuracy - should increase steadily

### Validation
- Track reward accuracy (should trend upward)
- Monitor reward margin (chosen - rejected rewards)
- Check for overfitting (val loss diverging from train)

## Performance Considerations

### Memory Usage
With default settings (d20 model):
- Policy model: ~2GB
- Reference model: ~2GB (no gradients)
- Batch overhead: ~1-2GB
- **Total**: ~5-6GB per GPU

For larger models or batch sizes:
- Reduce `device_batch_size`
- Enable gradient checkpointing (future feature)
- Use smaller `max_seq_len`

### Training Speed
Approximate speeds on 8×H100:
- d20 model: ~100 examples/sec
- d26 model: ~60 examples/sec
- d32 model: ~40 examples/sec

Typical training time:
- 10K examples: ~2-5 minutes
- 50K examples: ~10-25 minutes
- 100K examples: ~20-50 minutes

## Troubleshooting

### Reward Accuracy Not Improving
- Check β value (try increasing)
- Verify dataset quality (clear preferences?)
- Reduce learning rate
- Check for data leakage (train/val overlap)

### Model Diverges from Reference
- Increase β to strengthen KL constraint
- Reduce learning rate
- Add label smoothing
- Check gradient norms (may need clipping)

### OOM Errors
- Reduce `device_batch_size`
- Reduce `max_seq_len`
- Use gradient accumulation (already automatic)
- Check that reference model has `requires_grad=False`

### NaN Losses
- Check for invalid data (None, empty sequences)
- Verify all labels are valid token IDs or -1
- Add gradient clipping
- Reduce learning rate

## Testing

Run the test suite to verify implementation:
```bash
python test_dpo.py
```

Tests include:
1. DPO loss computation
2. Sequence log probability calculation
3. Concatenated forward pass
4. End-to-end training step
5. Label smoothing

## References

1. **Original DPO Paper**:
   "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
   Rafailov et al., NeurIPS 2023
   https://arxiv.org/abs/2305.18290

2. **RLHF Background**:
   "Training language models to follow instructions with human feedback"
   Ouyang et al., NeurIPS 2022
   https://arxiv.org/abs/2203.02155

3. **Preference Datasets**:
   - HH-RLHF: https://huggingface.co/datasets/Anthropic/hh-rlhf
   - UltraFeedback: https://huggingface.co/datasets/openbmb/UltraFeedback

## Future Improvements

Possible enhancements:
1. **IPO (Identity Preference Optimization)**: Variant with better theoretical properties
2. **KTO (Kahneman-Tversky Optimization)**: Works with unpaired feedback
3. **Online DPO**: Generate preferences on-the-fly during training
4. **Preference mixing**: Combine multiple preference sources
5. **Reward modeling**: Extract implicit reward function for analysis
6. **Adaptive β**: Automatically tune β during training
