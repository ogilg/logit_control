"""
Custom loss functions for LoRA fine-tuning.

This module contains loss functions designed for distribution matching tasks,
with support for both given words and random words scenarios.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


class CustomLossFunction:
    """Base class for custom loss functions that penalize divergence from target distributions."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def _prepare_target_data(self, logits: torch.Tensor, target_distribution: Dict[str, float], 
                           tokenizer, context: Optional[Dict[str, Any]] = None):
        """
        Prepare target data for loss computation.
        
        Args:
            logits: Model logits for next token prediction
            target_distribution: Target probability distribution over tokens
            tokenizer: Tokenizer for converting tokens to IDs
            context: Context information including is_given_words, word1, word2
        
        Returns:
            Tuple of (target_logits, target_probs, target_tokens)
        """
        is_given_words = context.get("is_given_words", True) if context else True
        
        if is_given_words:
            # Case 1: Given words - use specific target tokens
            target_tokens = list(target_distribution.keys())
            
            # Convert target tokens to token IDs
            target_token_ids = []
            for token in target_tokens:
                token_ids = tokenizer.encode(token, add_special_tokens=False)
                if len(token_ids) == 1:  # Single token
                    target_token_ids.append(token_ids[0])
                else:
                    # For multi-token words, use the first token
                    target_token_ids.append(token_ids[0])
            
            # Get logits for target tokens
            target_logits = logits[0, target_token_ids]
            
            # Create target probability tensor
            target_probs = torch.tensor([target_distribution.get(token, 0.0) for token in target_tokens], 
                                      dtype=torch.float32, device=logits.device)
            
            return target_logits, target_probs, target_tokens
            
        else:
            # Case 2: Random words - find top 2 tokens by log probability
            # Get top 2 tokens by log probability
            log_probs = F.log_softmax(logits[0] / self.temperature, dim=-1)
            top_k_log_probs, top_k_indices = torch.topk(log_probs, k=2)
            
            # Convert token IDs back to tokens
            target_tokens = []
            target_token_ids = top_k_indices.tolist()
            
            for token_id in target_token_ids:
                token = tokenizer.decode([token_id])
                target_tokens.append(token)
            
            # Get logits for top tokens
            target_logits = logits[0, top_k_indices]
            
            # Create target probability tensor from the actual target distribution
            # The target distribution has keys like "first_word", "second_word" with actual probabilities
            target_probs = torch.tensor([
                target_distribution.get("first_word", 0.5),
                target_distribution.get("second_word", 0.5)
            ], dtype=torch.float32, device=logits.device)
            
            return target_logits, target_probs, target_tokens
    
    def compute_loss(self, logits: torch.Tensor, target_distribution: Dict[str, float], 
                    tokenizer, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Compute custom loss based on divergence from target distribution.
        
        Args:
            logits: Model logits for next token prediction
            target_distribution: Target probability distribution over tokens
            tokenizer: Tokenizer for converting tokens to IDs
            context: Context information including is_given_words, word1, word2
        
        Returns:
            Loss tensor
        """
        target_logits, target_probs, _ = self._prepare_target_data(
            logits, target_distribution, tokenizer, context
        )
        return self._compute_loss_formula(target_logits, target_probs)
    
    def _compute_loss_formula(self, target_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the specific loss formula. Subclasses must implement this.
        
        Args:
            target_logits: Logits for target tokens
            target_probs: Target probability distribution
            
        Returns:
            Loss tensor
        """
        raise NotImplementedError("Subclasses must implement _compute_loss_formula")


class TVDLoss(CustomLossFunction):
    """Total Variation Distance loss function."""
    
    def _compute_loss_formula(self, target_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """Compute TVD loss between predicted and target distributions."""
        # Convert to probabilities
        probs = F.softmax(target_logits / self.temperature, dim=-1)
        
        # Compute TVD loss
        tvd = 0.5 * torch.sum(torch.abs(probs - target_probs))
        
        return tvd


class KLDivergenceLoss(CustomLossFunction):
    """KL Divergence loss function."""
    
    def _compute_loss_formula(self, target_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss between predicted and target distributions."""
        # Convert to log probabilities
        log_probs = F.log_softmax(target_logits / self.temperature, dim=-1)
        
        # Add small epsilon to avoid log(0)
        target_probs = target_probs + 1e-8
        target_probs = target_probs / target_probs.sum()  # Renormalize
        
        # Compute KL divergence
        kl_div = F.kl_div(log_probs, target_probs.log(), reduction='batchmean')
        
        return kl_div 