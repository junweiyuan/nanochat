"""
Test script for DPO (Direct Preference Optimization) implementation.
"""

import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.dpo import dpo_loss, compute_sequence_logprobs, concatenated_forward


def test_dpo_loss():
    """Test the DPO loss computation."""
    batch_size = 4
    
    policy_chosen_logps = torch.randn(batch_size)
    policy_rejected_logps = torch.randn(batch_size)
    reference_chosen_logps = torch.randn(batch_size)
    reference_rejected_logps = torch.randn(batch_size)
    
    loss, chosen_rewards, rejected_rewards, reward_accs = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=0.1
    )
    
    assert loss.ndim == 0, "Loss should be a scalar"
    assert chosen_rewards.ndim == 0, "Chosen rewards should be a scalar"
    assert rejected_rewards.ndim == 0, "Rejected rewards should be a scalar"
    assert reward_accs.ndim == 0, "Reward accuracies should be a scalar"
    assert 0 <= reward_accs <= 1, "Reward accuracy should be between 0 and 1"
    
    print(f"✓ DPO loss: {loss.item():.4f}")
    print(f"✓ Chosen rewards: {chosen_rewards.item():.4f}")
    print(f"✓ Rejected rewards: {rejected_rewards.item():.4f}")
    print(f"✓ Reward accuracy: {reward_accs.item():.4f}")


def test_compute_sequence_logprobs():
    """Test sequence log probability computation."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
    )
    model = GPT(config)
    model.init_weights()
    model.eval()
    
    B, T = 2, 32
    input_ids = torch.randint(0, 1000, (B, T))
    labels = torch.randint(0, 1000, (B, T))
    labels[:, :10] = -1
    
    with torch.no_grad():
        logprobs = compute_sequence_logprobs(model, input_ids, labels)
    
    assert logprobs.shape == (B,), f"Expected shape (B,), got {logprobs.shape}"
    print(f"✓ Sequence logprobs shape: {logprobs.shape}")
    print(f"✓ Sequence logprobs: {logprobs.tolist()}")


def test_concatenated_forward():
    """Test concatenated forward pass for efficiency."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
    )
    model = GPT(config)
    model.init_weights()
    model.eval()
    
    B, T = 2, 32
    batch = {
        'chosen_input_ids': torch.randint(0, 1000, (B, T)),
        'chosen_labels': torch.randint(0, 1000, (B, T)),
        'rejected_input_ids': torch.randint(0, 1000, (B, T)),
        'rejected_labels': torch.randint(0, 1000, (B, T)),
    }
    
    batch['chosen_labels'][:, :10] = -1
    batch['rejected_labels'][:, :10] = -1
    
    with torch.no_grad():
        chosen_logprobs, rejected_logprobs = concatenated_forward(model, batch)
    
    assert chosen_logprobs.shape == (B,), f"Expected shape (B,), got {chosen_logprobs.shape}"
    assert rejected_logprobs.shape == (B,), f"Expected shape (B,), got {rejected_logprobs.shape}"
    
    print(f"✓ Chosen logprobs shape: {chosen_logprobs.shape}")
    print(f"✓ Rejected logprobs shape: {rejected_logprobs.shape}")


def test_dpo_end_to_end():
    """Test DPO training step end-to-end."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
    )
    
    policy_model = GPT(config)
    policy_model.init_weights()
    
    reference_model = GPT(config)
    reference_model.load_state_dict(policy_model.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    B, T = 2, 32
    batch = {
        'chosen_input_ids': torch.randint(0, 1000, (B, T)),
        'chosen_labels': torch.randint(0, 1000, (B, T)),
        'rejected_input_ids': torch.randint(0, 1000, (B, T)),
        'rejected_labels': torch.randint(0, 1000, (B, T)),
    }
    
    batch['chosen_labels'][:, :10] = -1
    batch['rejected_labels'][:, :10] = -1
    
    policy_model.train()
    
    policy_chosen_logps, policy_rejected_logps = concatenated_forward(policy_model, batch)
    
    with torch.no_grad():
        ref_chosen_logps, ref_rejected_logps = concatenated_forward(reference_model, batch)
    
    loss, chosen_rewards, rejected_rewards, reward_acc = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.1
    )
    
    loss.backward()
    
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy_model.parameters())
    assert has_gradients, "Model should have gradients after backward pass"
    
    ref_has_no_gradients = all(p.grad is None for p in reference_model.parameters())
    assert ref_has_no_gradients, "Reference model should not have gradients"
    
    print(f"✓ End-to-end DPO training step successful")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Reward accuracy: {reward_acc.item():.4f}")


def test_label_smoothing():
    """Test DPO loss with label smoothing."""
    batch_size = 4
    
    policy_chosen_logps = torch.randn(batch_size)
    policy_rejected_logps = torch.randn(batch_size)
    reference_chosen_logps = torch.randn(batch_size)
    reference_rejected_logps = torch.randn(batch_size)
    
    loss_no_smooth, _, _, _ = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=0.1,
        label_smoothing=0.0
    )
    
    loss_smooth, _, _, _ = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=0.1,
        label_smoothing=0.1
    )
    
    print(f"✓ Loss without smoothing: {loss_no_smooth.item():.4f}")
    print(f"✓ Loss with smoothing: {loss_smooth.item():.4f}")


if __name__ == "__main__":
    print("Testing DPO implementation...\n")
    
    print("Test 1: DPO loss computation")
    test_dpo_loss()
    print()
    
    print("Test 2: Sequence log probability computation")
    test_compute_sequence_logprobs()
    print()
    
    print("Test 3: Concatenated forward pass")
    test_concatenated_forward()
    print()
    
    print("Test 4: End-to-end DPO training step")
    test_dpo_end_to_end()
    print()
    
    print("Test 5: Label smoothing")
    test_label_smoothing()
    print()
    
    print("✅ All DPO tests passed!")
