"""
Tests for metrics functionality.
"""

import pytest
import torch
import torch.nn as nn
from goldentransformer.metrics.accuracy import Accuracy
from goldentransformer.metrics.latency import LatencyMetric

class SimpleModel(nn.Module):
    """Simple model for testing metrics."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.linear(input_ids)
        return type('Output', (), {'logits': logits})()

@pytest.fixture
def model():
    """Create a test model."""
    return SimpleModel()

@pytest.fixture
def accuracy_metric():
    """Create accuracy metric."""
    return Accuracy()

@pytest.fixture
def latency_metric():
    """Create latency metric."""
    return LatencyMetric(num_runs=3)

def test_accuracy_metric_classification(accuracy_metric):
    """Test accuracy metric for classification task."""
    # Create test data
    outputs = type('Output', (), {
        'logits': torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    })()
    inputs = {
        'labels': torch.tensor([0, 1, 0])
    }
    batch_size = 3
    
    # Compute accuracy
    accuracy = accuracy_metric.compute(outputs, inputs, batch_size)
    
    # Verify accuracy
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

def test_accuracy_metric_language_modeling(accuracy_metric):
    """Test accuracy metric for language modeling task."""
    # Create test data
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    outputs = type('Output', (), {
        'logits': torch.randn(batch_size, seq_len, vocab_size)  # [batch_size, seq_len, vocab_size]
    })()
    inputs = {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len + 1))  # [batch_size, seq_len+1]
    }
    
    # Compute accuracy
    accuracy = accuracy_metric.compute(outputs, inputs, batch_size)
    
    # Verify accuracy is between 0 and 1
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

def test_latency_metric(latency_metric, model):
    """Test latency metric."""
    # Create test input
    inputs = {
        'input_ids': torch.randn(1, 10)
    }
    
    # Compute latency
    results = latency_metric.compute(model, inputs)
    
    # Verify results structure
    assert 'latency_ms' in results
    assert 'confidence' in results
    assert results['latency_ms'] > 0
    assert 0 <= results['confidence'] <= 1
    
    # Get summary
    summary = latency_metric.get_summary()
    assert 'avg_latency_ms' in summary
    assert 'min_latency_ms' in summary
    assert 'max_latency_ms' in summary
    assert 'avg_confidence' in summary
    
    # Verify summary values
    assert summary['avg_latency_ms'] > 0
    assert summary['min_latency_ms'] > 0
    assert summary['max_latency_ms'] > 0
    assert 0 <= summary['avg_confidence'] <= 1

def test_metric_reset(accuracy_metric, latency_metric):
    """Test metric reset functionality."""
    # Test accuracy metric reset
    accuracy_metric.reset()  # Should not raise any errors
    
    # Test latency metric reset
    latency_metric.reset()
    summary = latency_metric.get_summary()
    assert summary['avg_latency_ms'] == 0
    assert summary['min_latency_ms'] == float('inf')
    assert summary['max_latency_ms'] == 0
    assert summary['avg_confidence'] == 0 