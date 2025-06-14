"""
Tests for fault injection functionality.
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from goldentransformer.core.fault_injector import FaultInjector
from goldentransformer.faults.layer_fault import LayerFault
from goldentransformer.faults.weight_corruption import WeightCorruption
from goldentransformer.faults.activation_fault import ActivationFault
from goldentransformer.metrics.latency import LatencyMetric
from goldentransformer.metrics.accuracy import Accuracy
from goldentransformer.core.experiment_runner import ExperimentRunner

class SimpleTransformer(nn.Module):
    """Simple transformer model for testing."""
    
    def __init__(self, vocab_size=1000, d_model=64, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128)
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        print(f"DEBUG: SimpleTransformer.forward called with input_ids shape {input_ids.shape}, labels={labels}")
        x = self.embedding(input_ids)
        for layer in self.transformer:
            x = layer(x)
        return type('Output', (), {'logits': self.output(x)})()

@pytest.fixture
def model():
    """Create a test model."""
    return SimpleTransformer()

@pytest.fixture
def injector(model):
    """Create a fault injector."""
    return FaultInjector(model)

@pytest.fixture
def metrics():
    """Create test metrics."""
    return [LatencyMetric(), Accuracy()]

def test_layer_fault_injection(injector):
    """Test layer fault injection."""
    # Create layer fault
    fault = LayerFault(
        layer_idx=0,
        fault_type="attention_mask",
        severity=0.3
    )
    # Inject and revert fault, should not raise
    injector.inject_fault(fault)
    injector.revert_fault()

def test_weight_corruption(injector):
    """Test weight corruption."""
    # Store original weights
    original_weights = {
        name: param.clone()
        for name, param in injector.model.named_parameters()
    }
    
    # Create weight corruption fault
    fault = WeightCorruption(
        pattern="random",
        corruption_rate=0.1
    )
    
    # Inject fault
    injector.inject_fault(fault)
    
    # Verify weights were corrupted
    weights_changed = False
    for name, param in injector.model.named_parameters():
        if not torch.allclose(param, original_weights[name]):
            weights_changed = True
            break
    
    assert weights_changed
    
    # Revert fault
    injector.revert_fault()
    
    # Verify weights were restored
    for name, param in injector.model.named_parameters():
        assert torch.allclose(param, original_weights[name])

def test_activation_fault(injector):
    """Test activation fault injection."""
    # Create activation fault
    fault = ActivationFault(
        layer_idx=0,
        fault_type="clamp",
        severity=0.3,
        target_activations=["gelu"]
    )
    
    # Store original forward methods
    original_forwards = {}
    for name, module in injector.model.transformer[0].named_modules():
        if isinstance(module, nn.GELU):
            original_forwards[name] = module.forward
    
    # Inject fault
    injector.inject_fault(fault)
    
    # Verify forward methods were modified
    for name, module in injector.model.transformer[0].named_modules():
        if isinstance(module, nn.GELU):
            assert module.forward != original_forwards[name]
    
    # Revert fault
    injector.revert_fault()
    
    # Verify forward methods were restored
    for name, module in injector.model.transformer[0].named_modules():
        if isinstance(module, nn.GELU):
            assert module.forward == original_forwards[name]

def test_experiment_runner(model, injector, metrics):
    """Test experiment runner."""
    # Create test dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 1000, (100, 32)),  # input_ids
        torch.randint(0, 1000, (100, 32))   # labels
    )
    
    # Create faults
    faults = [
        LayerFault(layer_idx=0, severity=0.2),
        WeightCorruption(corruption_rate=0.1)
    ]
    
    # Create experiment runner
    runner = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset=dataset,
        batch_size=16,
        num_samples=50
    )
    
    # Run experiment
    results = runner.run()
    
    # Verify results structure
    assert 'baseline' in results
    assert 'fault_results' in results
    assert len(results['fault_results']) == len(faults)
    
    # Verify metrics were computed
    for fault_result in results['fault_results']:
        assert 'metrics' in fault_result
        assert 'LatencyMetric' in fault_result['metrics']
        assert 'Accuracy' in fault_result['metrics']

def test_latency_metric(model):
    """Test latency metric."""
    metric = LatencyMetric(num_runs=2)
    
    # Create test input
    inputs = {
        'input_ids': torch.randint(0, 1000, (1, 32))
    }
    
    # Compute metric
    results = metric.compute(model, inputs)
    
    # Verify results
    assert 'latency_ms' in results
    assert 'confidence' in results
    assert results['latency_ms'] > 0
    
    # Get summary
    summary = metric.get_summary()
    assert 'avg_latency_ms' in summary
    assert 'min_latency_ms' in summary
    assert 'max_latency_ms' in summary
    assert 'avg_confidence' in summary 