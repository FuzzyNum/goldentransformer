"""
End-to-end test of the GoldenTransformer framework.
This script demonstrates the full functionality of the framework by:
1. Loading a pre-trained model
2. Setting up the fault injection framework
3. Running experiments with different types of faults
4. Measuring and visualizing the results
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from goldentransformer.core.experiment_runner import ExperimentRunner
from goldentransformer.core.fault_injector import FaultInjector
from goldentransformer.faults.layer_fault import LayerFault
from goldentransformer.faults.weight_corruption import WeightCorruption
from goldentransformer.faults.activation_fault import ActivationFault
from goldentransformer.metrics.accuracy import Accuracy
from goldentransformer.metrics.latency import LatencyMetric

def prepare_dataset(tokenizer, max_length=128):
    """Prepare the IMDB dataset for testing."""
    dataset = load_dataset("imdb", split="test[:100]")  # Use a small subset for testing
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Convert to PyTorch tensors
    input_ids = torch.tensor(tokenized_dataset["input_ids"])
    attention_mask = torch.tensor(tokenized_dataset["attention_mask"])
    labels = torch.tensor(dataset["label"])
    
    # Create PyTorch dataset
    return torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to device
    model = model.to(device)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(tokenizer)
    
    # Create fault injector
    print("Setting up fault injector...")
    injector = FaultInjector(model)
    
    # Define faults to test
    faults = [
        LayerFault(layer_idx=0, severity=0.2),
        WeightCorruption(corruption_rate=0.1),
        ActivationFault(layer_idx=0, severity=0.3)
    ]
    
    # Define metrics
    metrics = [
        Accuracy(),
        LatencyMetric(num_runs=3)
    ]
    
    # Create experiment runner
    print("Setting up experiment runner...")
    runner = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset=dataset,
        batch_size=16,
        num_samples=50,
        device=device
    )
    
    # Run experiment
    print("Running experiment...")
    results = runner.run()
    
    # Print results
    print("\nExperiment Results:")
    print("==================")
    
    # Print baseline results
    print("\nBaseline Results:")
    for metric_name, metric_value in results['baseline'].items():
        print(f"{metric_name}: {metric_value}")
    
    # Print fault results
    print("\nFault Results:")
    for fault_result in results['fault_results']:
        print(f"\nFault: {fault_result['fault_info']}")
        if 'error' in fault_result:
            print(f"Error: {fault_result['error']}")
        else:
            for metric_name, metric_value in fault_result['metrics'].items():
                print(f"{metric_name}: {metric_value}")

if __name__ == "__main__":
    main() 