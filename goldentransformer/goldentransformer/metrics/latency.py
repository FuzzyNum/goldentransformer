"""
Latency metric implementation for measuring model inference time.
"""

import time
import torch
from typing import Dict, Any
from goldentransformer.metrics.base_metric import BaseMetric

class LatencyMetric(BaseMetric):
    """Metric for computing model inference latency."""
    
    def __init__(self, num_runs: int = 5):
        """
        Initialize the latency metric.
        
        Args:
            num_runs (int): Number of runs to average over
        """
        super().__init__()
        self.num_runs = num_runs
        self.reset()
    
    def compute(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute the latency metric.
        
        Args:
            model: The model to measure
            inputs: Model inputs
            
        Returns:
            Dict containing latency_ms and confidence
        """
        # Warmup run
        with torch.no_grad():
            model(**inputs)
        
        # Measure latency
        latencies = []
        for _ in range(self.num_runs):
            start_time = time.time()
            with torch.no_grad():
                model(**inputs)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Compute statistics
        latency_ms = sum(latencies) / len(latencies)
        confidence = 1.0 - (max(latencies) - min(latencies)) / latency_ms if latency_ms > 0 else 0.0
        
        # Update running statistics
        self.total_latency += latency_ms
        self.num_samples += 1
        self.min_latency = min(self.min_latency, latency_ms)
        self.max_latency = max(self.max_latency, latency_ms)
        self.total_confidence += confidence
        
        return {
            'latency_ms': latency_ms,
            'confidence': confidence
        }
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            'avg_latency_ms': self.total_latency / self.num_samples if self.num_samples > 0 else 0.0,
            'min_latency_ms': self.min_latency,
            'max_latency_ms': self.max_latency,
            'avg_confidence': self.total_confidence / self.num_samples if self.num_samples > 0 else 0.0
        }
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total_latency = 0.0
        self.num_samples = 0
        self.min_latency = float('inf')
        self.max_latency = 0.0
        self.total_confidence = 0.0 