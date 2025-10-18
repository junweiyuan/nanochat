"""
Preference dataset for DPO training.

This module provides datasets of (chosen, rejected) preference pairs
for training with Direct Preference Optimization (DPO).
"""

import json
import random
from datasets import load_dataset


class HHRLHFPreferences:
    """
    Anthropic's Helpful and Harmless RLHF preference dataset.
    Contains human preferences between two assistant responses.
    """
    
    def __init__(self, split="train", max_examples=None):
        self.split = split
        
        try:
            dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        except Exception as e:
            print(f"Warning: Could not load hh-rlhf dataset: {e}")
            print("Creating synthetic preference data for testing...")
            dataset = self._create_synthetic_data()
        
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        self.data = []
        for item in dataset:
            if isinstance(item, dict):
                if 'chosen' in item and 'rejected' in item:
                    self.data.append({
                        'prompt': '',
                        'chosen': item['chosen'],
                        'rejected': item['rejected']
                    })
    
    def _create_synthetic_data(self):
        """Create synthetic preference data for testing when dataset unavailable."""
        synthetic_examples = []
        
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "What are the health benefits of exercise?",
            "How do I bake chocolate chip cookies?",
            "What is photosynthesis?",
        ]
        
        for prompt in prompts:
            chosen = f"Human: {prompt}\n\nAssistant: [High quality helpful response]"
            rejected = f"Human: {prompt}\n\nAssistant: [Lower quality or unhelpful response]"
            
            synthetic_examples.append({
                'chosen': chosen,
                'rejected': rejected
            })
        
        return synthetic_examples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class UltraFeedbackPreferences:
    """
    UltraFeedback preference dataset with ratings.
    Contains multiple responses with quality ratings.
    """
    
    def __init__(self, split="train", max_examples=None):
        self.split = split
        
        try:
            dataset = load_dataset("openbmb/UltraFeedback", split=split)
        except Exception as e:
            print(f"Warning: Could not load UltraFeedback dataset: {e}")
            dataset = []
        
        if max_examples is not None and len(dataset) > 0:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        self.data = []
        for item in dataset:
            if len(item.get('completions', [])) >= 2:
                completions = item['completions']
                
                ratings = [c.get('rating', 0) for c in completions]
                
                best_idx = ratings.index(max(ratings))
                worst_idx = ratings.index(min(ratings))
                
                if best_idx != worst_idx:
                    self.data.append({
                        'prompt': item.get('instruction', ''),
                        'chosen': completions[best_idx]['response'],
                        'rejected': completions[worst_idx]['response']
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SyntheticPreferences:
    """
    Synthetic preference dataset created from GSM8K or other tasks.
    Uses correct/incorrect completions as chosen/rejected pairs.
    """
    
    def __init__(self, base_task, num_samples=2, temperature=1.0):
        """
        Args:
            base_task: A task that has an evaluate() method (e.g., GSM8K)
            num_samples: Number of samples to generate per prompt
            temperature: Sampling temperature
        """
        self.base_task = base_task
        self.num_samples = num_samples
        self.temperature = temperature
        self.data = []
    
    def generate_from_model(self, model, tokenizer, engine, max_examples=100):
        """
        Generate preference pairs by sampling from the model.
        Pairs are created from correct (chosen) and incorrect (rejected) samples.
        """
        import torch
        
        self.data = []
        
        for idx in range(min(max_examples, len(self.base_task))):
            conversation = self.base_task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            
            generated_sequences, masks = engine.generate_batch(
                tokens,
                num_samples=self.num_samples,
                max_tokens=256,
                temperature=self.temperature,
            )
            
            prefix_length = len(tokens)
            
            correct_responses = []
            incorrect_responses = []
            
            for seq in generated_sequences:
                generated_tokens = seq[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                
                is_correct = self.base_task.evaluate(conversation, generated_text)
                
                if is_correct:
                    correct_responses.append(generated_text)
                else:
                    incorrect_responses.append(generated_text)
            
            if correct_responses and incorrect_responses:
                self.data.append({
                    'prompt': conversation,
                    'chosen': random.choice(correct_responses),
                    'rejected': random.choice(incorrect_responses)
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
