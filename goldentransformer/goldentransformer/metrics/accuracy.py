"""
Accuracy metric implementation for measuring model accuracy.
"""

import torch
from typing import Dict
from goldentransformer.metrics.base_metric import BaseMetric

class Accuracy(BaseMetric):
    """Metric for computing model accuracy."""
    
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        batch_size: int
    ) -> float:
        """
        Compute the accuracy metric.
        Handles both sequence classification and language modeling.
        
        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs
            inputs (Dict[str, torch.Tensor]): Model inputs
            batch_size (int): Size of the current batch
        
        Returns:
            float: Accuracy value between 0 and 1
        """
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Sequence classification (e.g., sentiment analysis)
        if "labels" in inputs:
            labels = inputs["labels"]
            correct = (predictions == labels).float().sum()
            total = labels.numel()
            return (correct / total).item()
        
        # Language modeling
        elif "input_ids" in inputs:
            targets = inputs["input_ids"][:, 1:]
            predictions = predictions[:, :targets.shape[1]]  # Ensure shapes match
            correct = (predictions == targets).float().sum()
            total = targets.numel()
            return (correct / total).item()
        
        else:
            raise ValueError("Inputs must contain either 'labels' or 'input_ids'.")

    def get_summary(self) -> Dict[str, float]:
        """Return summary statistics (not implemented for simple accuracy)."""
        return {}

    def reset(self):
        """Reset any accumulated state (not used in simple accuracy)."""
        pass 