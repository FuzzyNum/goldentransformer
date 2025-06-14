"""
Tests for experiment runner functionality.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from goldentransformer.core.experiment_runner import ExperimentRunner
from goldentransformer.core.fault_injector import FaultInjector
from goldentransformer.faults.layer_fault import LayerFault
from goldentransformer.faults.weight_corruption import WeightCorruption
from goldentransformer.faults.activation_fault import ActivationFault
from goldentransformer.metrics.accuracy import Accuracy
from goldentransformer.metrics.latency import LatencyMetric

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
    return [Accuracy(), LatencyMetric(num_runs=2)]

@pytest.fixture
def dataset():
    """Create a test dataset."""
    return torch.utils.data.TensorDataset(
        torch.randint(0, 1000, (100, 32)),  # input_ids
        torch.randint(0, 1000, (100, 32))   # labels
    )

@pytest.fixture
def faults():
    """Create test faults."""
    return [
        LayerFault(layer_idx=0, severity=0.2),
        WeightCorruption(corruption_rate=0.1),
        ActivationFault(layer_idx=0, severity=0.3)
    ]

def test_experiment_runner_initialization(injector, faults, metrics, dataset):
    """Test experiment runner initialization."""
    runner = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset=dataset,
        batch_size=16,
        num_samples=50
    )
    
    # Verify initialization
    assert runner.injector == injector
    assert runner.faults == faults
    assert runner.metrics == metrics
    assert runner.dataset == dataset
    assert runner.batch_size == 16
    assert runner.num_samples == 50
    assert isinstance(runner.output_dir, Path)
    assert runner.output_dir.exists()

def test_experiment_runner_baseline(injector, faults, metrics, dataset):
    """Test experiment runner baseline evaluation."""
    runner = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset=dataset,
        batch_size=16,
        num_samples=50
    )
    
    # Run baseline evaluation
    baseline_results = runner._run_baseline()
    
    # Verify baseline results structure
    assert 'Accuracy' in baseline_results
    assert 'LatencyMetric' in baseline_results
    # Accuracy is now a dict, check the value inside
    acc_val = baseline_results['Accuracy'].get('Accuracy', None)
    assert acc_val is not None
    assert acc_val >= 0
    assert baseline_results['LatencyMetric']['latency_ms'] > 0

def test_experiment_runner_fault_injection(injector, faults, metrics, dataset):
    """Test experiment runner fault injection."""
    runner = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset=dataset,
        batch_size=16,
        num_samples=50
    )
    
    # Run experiment with faults
    results = runner.run()
    
    # Verify results structure
    assert 'baseline' in results
    assert 'fault_results' in results
    assert len(results['fault_results']) == len(faults)
    
    # Verify each fault result
    for fault_result in results['fault_results']:
        assert 'fault_info' in fault_result
        assert 'metrics' in fault_result
        assert 'Accuracy' in fault_result['metrics']
        assert 'LatencyMetric' in fault_result['metrics']

def test_experiment_runner_output_files(injector, faults, metrics, dataset):
    """Test experiment runner output files."""
    runner = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset=dataset,
        batch_size=16,
        num_samples=50
    )
    
    # Run experiment
    runner.run()
    
    # Verify output files
    assert (runner.output_dir / "results.json").exists()
    assert (runner.output_dir / "experiment.log").exists()

def test_experiment_runner_error_handling(injector, metrics, dataset):
    """Test experiment runner error handling."""
    # Create a fault that will raise an error
    class BadFault(LayerFault):
        def inject(self, model):
            raise ValueError("Test error")
    
    faults = [BadFault(layer_idx=0, severity=0.2)]
    
    runner = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset=dataset,
        batch_size=16,
        num_samples=50
    )
    
    # Run experiment - should handle error gracefully
    results = runner.run()
    
    # Verify error was captured
    assert len(results['fault_results']) == 1
    assert 'error' in results['fault_results'][0]
    assert "Test error" in results['fault_results'][0]['error'] 