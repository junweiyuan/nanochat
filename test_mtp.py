"""
Test script for multi-token prediction implementation.
"""

import torch
from nanochat.gpt import GPT, GPTConfig, create_multi_token_targets

def test_mtp_config():
    """Test that GPTConfig accepts n_predict parameter."""
    config = GPTConfig(n_predict=3)
    assert config.n_predict == 3
    print("✓ GPTConfig with n_predict=3")

def test_mtp_model_creation():
    """Test that GPT model can be created with multi-token prediction."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        n_predict=3
    )
    model = GPT(config)
    
    assert hasattr(model, 'mtp_heads')
    assert len(model.mtp_heads) == 2
    print("✓ GPT model with n_predict=3 has 2 MTP heads")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model has {param_count:,} parameters")

def test_single_token_prediction():
    """Test that single-token prediction still works."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        n_predict=1
    )
    model = GPT(config)
    model.init_weights()
    
    assert not hasattr(model, 'mtp_heads')
    
    B, T = 2, 32
    idx = torch.randint(0, 1000, (B, T))
    targets = torch.randint(0, 1000, (B, T))
    
    loss = model(idx, targets)
    assert loss.ndim == 0
    print(f"✓ Single-token prediction loss: {loss.item():.4f}")

def test_multi_token_targets():
    """Test create_multi_token_targets utility function."""
    B, T = 2, 8
    targets = torch.arange(B * T).view(B, T)
    n_predict = 3
    
    multi_targets = create_multi_token_targets(targets, n_predict)
    
    assert multi_targets.shape == (B, T, n_predict)
    
    print(f"✓ create_multi_token_targets shape: {multi_targets.shape}")
    print(f"  Original targets[0]: {targets[0].tolist()}")
    print(f"  Multi targets[0, :, 0]: {multi_targets[0, :, 0].tolist()}")
    print(f"  Multi targets[0, :, 1]: {multi_targets[0, :, 1].tolist()}")
    print(f"  Multi targets[0, :, 2]: {multi_targets[0, :, 2].tolist()}")

def test_multi_token_prediction():
    """Test that multi-token prediction works end-to-end."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        n_predict=3
    )
    model = GPT(config)
    model.init_weights()
    
    B, T = 2, 32
    idx = torch.randint(0, 1000, (B, T))
    targets = torch.randint(0, 1000, (B, T))
    
    multi_targets = create_multi_token_targets(targets, config.n_predict)
    
    loss = model(idx, multi_targets)
    assert loss.ndim == 0
    print(f"✓ Multi-token prediction loss: {loss.item():.4f}")
    
    loss.backward()
    print("✓ Backward pass successful")

def test_inference_mode():
    """Test that inference mode still works."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        n_predict=3
    )
    model = GPT(config)
    model.init_weights()
    model.eval()
    
    B, T = 2, 32
    idx = torch.randint(0, 1000, (B, T))
    
    with torch.no_grad():
        logits = model(idx)
    
    assert logits.shape == (B, T, 1000)
    print(f"✓ Inference mode logits shape: {logits.shape}")

if __name__ == "__main__":
    print("Testing multi-token prediction implementation...\n")
    
    test_mtp_config()
    test_mtp_model_creation()
    test_single_token_prediction()
    test_multi_token_targets()
    test_multi_token_prediction()
    test_inference_mode()
    
    print("\n✅ All tests passed!")
