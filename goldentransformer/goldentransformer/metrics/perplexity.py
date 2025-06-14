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
        # Get logits and target tokens
        logits = outputs.logits
        targets = inputs["input_ids"][:, 1:]
        
        # Compute cross entropy loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # Compute perplexity
        perplexity = torch.exp(loss.mean())
        
        return perplexity.item() * batch_size 