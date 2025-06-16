"""
Example experiment: Analyzing the impact of different severity levels on model performance.
This experiment demonstrates how varying severity levels affect model accuracy and latency
across different types of faults.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from goldentransformer.core.experiment_runner import ExperimentRunner
from goldentransformer.core.fault_injector import FaultInjector
from goldentransformer.faults.layer_fault import LayerFault
from goldentransformer.faults.weight_corruption import WeightCorruption
from goldentransformer.faults.activation_fault import ActivationFault
from goldentransformer.metrics.accuracy import Accuracy
from goldentransformer.metrics.latency import LatencyMetric

def prepare_dataset(tokenizer, max_length=128):
    """Prepare a small dataset for language modeling."""
    # Use a small subset of the wikitext dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:100]")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    input_ids = torch.tensor(tokenized_dataset["input_ids"])
    attention_mask = torch.tensor(tokenized_dataset["attention_mask"])
    labels = input_ids.clone()  # For language modeling, labels are the same as input_ids
    return torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

def create_faults():
    """Create faults with different severity levels."""
    faults = []
    
    # Test different severity levels
    severities = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Layer faults
    for severity in severities:
        faults.append(LayerFault(
            layer_idx=0,
            fault_type="attention_mask",
            severity=severity
        ))
    
    # Weight corruption
    for severity in severities:
        faults.append(WeightCorruption(
            pattern="random",
            corruption_rate=severity
        ))
    
    # Activation faults
    for severity in severities:
        faults.append(ActivationFault(
            layer_idx=0,
            fault_type="clamp",
            severity=severity
        ))
    
    return faults

def plot_results(results):
    """Plot experiment results."""
    # Extract data
    fault_types = []
    severities = []
    accuracies = []
    latencies = []
    
    for fault in results["fault_results"]:
        if "error" in fault:
            continue  # Skip failed fault injections
            
        fault_info = fault["fault_info"]
        fault_types.append(fault_info.split("(")[0])
        severity = float(fault_info.split("severity=")[1].split(")")[0])
        severities.append(severity)
        
        # Handle missing metrics gracefully
        if "metrics" in fault and "Accuracy" in fault["metrics"]:
            accuracies.append(fault["metrics"]["Accuracy"].get("Accuracy", 0.0))
        else:
            accuracies.append(0.0)
            
        if "metrics" in fault and "LatencyMetric" in fault["metrics"]:
            latencies.append(fault["metrics"]["LatencyMetric"].get("avg_latency_ms", 0.0))
        else:
            latencies.append(0.0)
    
    if not fault_types:  # If all faults failed
        print("No successful fault injections to plot")
        return
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot accuracy
    for fault_type in set(fault_types):
        mask = [t == fault_type for t in fault_types]
        ax1.plot([s for s, m in zip(severities, mask) if m],
                [a for a, m in zip(accuracies, mask) if m],
                'o-', label=fault_type)
    
    ax1.set_xlabel('Severity')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Impact of Fault Severity on Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot latency
    for fault_type in set(fault_types):
        mask = [t == fault_type for t in fault_types]
        ax2.plot([s for s, m in zip(severities, mask) if m],
                [l for l, m in zip(latencies, mask) if m],
                'o-', label=fault_type)
    
    ax2.set_xlabel('Severity')
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_title('Impact of Fault Severity on Model Latency')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('severity_analysis_results.png')
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "gpt2"  # Using GPT-2 which has the required transformer.h structure
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up tokenizer for padding
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    model = model.to(device)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(tokenizer)
    
    # Create fault injector
    print("Setting up fault injector...")
    injector = FaultInjector(model)
    
    # Create faults
    print("Creating faults...")
    faults = create_faults()
    
    # Define metrics
    metrics = [Accuracy(), LatencyMetric(num_runs=3)]
    
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
    
    # Plot results
    print("Plotting results...")
    plot_results(results)
    
    print("Experiment completed. Results saved to 'severity_analysis_results.png'")

if __name__ == "__main__":
    main() 