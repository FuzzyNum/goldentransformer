"""
Base metric class for the GoldenTransformer framework.
All metrics should inherit from this class.
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any

class BaseMetric(ABC):
    """Base class for all metrics in the framework."""
    
    @abstractmethod
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        batch_size: int
    ) -> float:
        """
        Compute the metric value.
        
        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs
            inputs (Dict[str, torch.Tensor]): Model inputs
            batch_size (int): Size of the current batch
        
        Returns:
            float: Computed metric value
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the metric."""
        return self.__class__.__name__ 