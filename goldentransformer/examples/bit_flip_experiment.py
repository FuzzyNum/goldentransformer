"""
Example experiment: Analyzing the impact of bit flip faults on RNN model performance.
This experiment demonstrates how single bit flips affect model accuracy and perplexity.
"""

import torch
import torch.nn as nn
from pathlib import Path
from goldentransformer import FaultInjector, ExperimentRunner
from goldentransformer.faults import WeightCorruption
from goldentransformer.metrics import Accuracy, Perplexity
from goldentransformer.visualization import plot_results

class RNNModel(nn.Module):
    """Simple RNN model for testing."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)

def load_rnn_model(model_path: str) -> nn.Module:
    """Load the RNN model from file."""
    # Model parameters (adjust these based on your model)
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    
    # Create model
    model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model

def create_bit_flip_faults():
    """Create a set of bit flip faults to test."""
    faults = []
    
    # Test different corruption rates
    corruption_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    # Create faults for each corruption rate
    for rate in corruption_rates:
        faults.append(
            WeightCorruption(
                pattern="bit_flip",
                corruption_rate=rate,
                target_weights=["weight"]  # Target all weight matrices
            )
        )
    
    return faults

def run_experiment():
    """Run the bit flip fault analysis experiment."""
    # Load the RNN model
    model_path = "rnn_model.pt"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_rnn_model(model_path)
    
    # Initialize the fault injector
    injector = FaultInjector(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create bit flip faults
    faults = create_bit_flip_faults()
    
    # Set up metrics
    metrics = [
        Accuracy(),
        Perplexity()
    ]
    
    # Initialize experiment runner
    experiment = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset="wikitext-2",
        batch_size=32,
        num_samples=1000,
        output_dir="bit_flip_results"
    )
    
    # Run the experiment
    print("Starting bit flip fault analysis experiment...")
    results = experiment.run()
    
    # Generate visualizations
    plot_results(
        results,
        output_dir="bit_flip_results/plots",
        plot_types=["severity_impact", "fault_type_impact"]
    )
    
    return results

def analyze_results(results):
    """Analyze and print key findings from the experiment."""
    print("\nKey Findings:")
    
    # Analyze baseline performance
    baseline = results["baseline"]
    print("\nBaseline Performance:")
    for metric, value in baseline.items():
        print(f"{metric}: {value:.4f}")
    
    # Analyze impact of bit flips
    print("\nImpact of Bit Flips:")
    for result in results["results"]:
        fault = result["fault"]
        metrics = result["metrics"]
        
        # Extract corruption rate from fault string
        rate = float(fault.split("corruption_rate=")[1].split(")")[0])
        
        print(f"\nCorruption Rate: {rate:.3f}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
            # Calculate relative change from baseline
            baseline_value = baseline[metric]
            change = ((value - baseline_value) / baseline_value) * 100
            print(f"Change from baseline: {change:+.2f}%")

if __name__ == "__main__":
    # Run the experiment
    results = run_experiment()
    
    # Analyze and print results
    analyze_results(results) 