"""
Perplexity metric implementation for measuring model perplexity.
"""

import torch
from typing import Dict
from goldentransformer.metrics.base_metric import BaseMetric

class Perplexity(BaseMetric):
    """Metric for computing model perplexity."""
    
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        batch_size: int
    ) -> float:
        """
        Compute the perplexity metric.
        
        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs
            inputs (Dict[str, torch.Tensor]): Model inputs
            batch_size (int): Size of the current batch
        
        Returns:
            float: Perplexity value
        """
        # Align logits and targets for next-token prediction
        # logits: [batch_size, seq_len, vocab_size] -> [batch_size, seq_len-1, vocab_size]
        # targets: [batch_size, seq_len] -> [batch_size, seq_len-1]
        logits = outputs.logits[:, :-1, :]
        targets = inputs["input_ids"][:, 1:]
        
        # Compute cross entropy loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Compute perplexity
        perplexity = torch.exp(loss.mean())
        
        return perplexity.item() * batch_size 

    def reset(self):
        pass 