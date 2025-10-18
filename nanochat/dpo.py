"""
Direct Preference Optimization (DPO) loss implementation.

DPO trains language models directly on preference data without needing a separate reward model.
The key idea is to optimize the policy to increase the likelihood of preferred responses
while decreasing the likelihood of rejected responses, relative to a reference model.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1,
    label_smoothing=0.0,
    reduction='mean'
):
    """
    Compute the DPO loss for a batch of preference pairs.
    
    Args:
        policy_chosen_logps: Log probabilities of chosen responses under the policy model (B,)
        policy_rejected_logps: Log probabilities of rejected responses under the policy model (B,)
        reference_chosen_logps: Log probabilities of chosen responses under the reference model (B,)
        reference_rejected_logps: Log probabilities of rejected responses under the reference model (B,)
        beta: Temperature parameter controlling the strength of the KL constraint (default: 0.1)
        label_smoothing: Optional label smoothing (default: 0.0)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: The DPO loss value
        chosen_rewards: Implicit rewards for chosen responses
        rejected_rewards: Implicit rewards for rejected responses
        reward_accuracies: Fraction of examples where chosen reward > rejected reward
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    if label_smoothing == 0.0:
        losses = -F.logsigmoid(beta * logits)
    else:
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    
    if reduction == 'mean':
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean(), reward_accuracies.mean()
    elif reduction == 'sum':
        return losses.sum(), chosen_rewards.sum(), rejected_rewards.sum(), reward_accuracies.sum()
    else:
        return losses, chosen_rewards, rejected_rewards, reward_accuracies


def compute_sequence_logprobs(model, input_ids, labels, autocast_ctx=None):
    """
    Compute the log probability of a sequence under the model.
    
    Args:
        model: The language model
        input_ids: Input token IDs (B, T)
        labels: Target token IDs (B, T), with -1 for positions to ignore
        autocast_ctx: Optional autocast context for mixed precision
    
    Returns:
        logprobs: Sum of log probabilities for each sequence (B,)
    """
    ctx = autocast_ctx if autocast_ctx is not None else torch.cuda.amp.autocast(enabled=False)
    
    with ctx:
        logits = model(input_ids)
    
    logits = logits.float()
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    per_token_logps = torch.gather(log_probs[:, :-1, :], dim=2, index=labels[:, 1:].unsqueeze(2)).squeeze(2)
    
    mask = (labels[:, 1:] >= 0).float()
    
    sequence_logprobs = (per_token_logps * mask).sum(dim=1)
    
    return sequence_logprobs


def concatenated_forward(model, batch, autocast_ctx=None):
    """
    Run forward pass on concatenated chosen and rejected samples.
    This is more efficient than running them separately.
    
    Args:
        model: The language model
        batch: Dict containing 'chosen_input_ids', 'chosen_labels', 
               'rejected_input_ids', 'rejected_labels'
        autocast_ctx: Optional autocast context
    
    Returns:
        chosen_logprobs: Log probabilities for chosen responses (B,)
        rejected_logprobs: Log probabilities for rejected responses (B,)
    """
    concatenated_input_ids = torch.cat([
        batch['chosen_input_ids'],
        batch['rejected_input_ids']
    ], dim=0)
    
    concatenated_labels = torch.cat([
        batch['chosen_labels'],
        batch['rejected_labels']
    ], dim=0)
    
    all_logprobs = compute_sequence_logprobs(model, concatenated_input_ids, concatenated_labels, autocast_ctx)
    
    batch_size = batch['chosen_input_ids'].shape[0]
    chosen_logprobs = all_logprobs[:batch_size]
    rejected_logprobs = all_logprobs[batch_size:]
    
    return chosen_logprobs, rejected_logprobs
