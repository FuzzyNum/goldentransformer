"""
Example experiment analyzing the impact of bit flip faults on a BERT model
finetuned for IMDB sentiment classification.
"""

import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from goldentransformer.faults import WeightCorruption
from goldentransformer.metrics import Accuracy
from goldentransformer.core import FaultInjector, ExperimentRunner
from goldentransformer.visualization import plot_results

def load_model_and_tokenizer():
    """Load the BERT model and tokenizer for IMDB sentiment classification."""
    model_name = "textattack/bert-base-uncased-imdb"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepare_dataset(tokenizer, max_length=128, batch_size=32):
    """Prepare the IMDB dataset for evaluation."""
    dataset = load_dataset("imdb", split="test")
    
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        # Rename 'label' to 'labels' for compatibility
        tokens["labels"] = examples["label"]
        return tokens
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != "label"]
    )
    
    # Convert to PyTorch format
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return dataloader

def create_bit_flip_faults():
    """Create a list of bit flip faults with varying corruption rates."""
    corruption_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
    return [
        WeightCorruption(
            pattern="bit_flip",
            corruption_rate=rate,
            target_layers=None,  # Target all layers
            target_weights=["weight"]  # Only target weight matrices
        )
        for rate in corruption_rates
    ]

def run_experiment():
    """Run the bit flip fault analysis experiment."""
    print("Starting IMDB bit flip fault analysis experiment...")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    model.eval()  # Set to evaluation mode
    
    # Prepare dataset
    dataloader = prepare_dataset(tokenizer)
    
    # Initialize fault injector
    injector = FaultInjector(model)
    
    # Initialize metrics
    metrics = [Accuracy()]
    
    # Create experiment runner
    experiment = ExperimentRunner(
        injector=injector,
        dataloader=dataloader,
        metrics=metrics,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create faults
    faults = create_bit_flip_faults()
    
    # Run experiment
    print("\nRunning baseline evaluation...")
    results = experiment.run(faults)
    
    # Save results
    output_dir = "imdb_bit_flip_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/experiment_results.json")
    
    return results

def analyze_results(results):
    """Analyze and print key findings from the experiment."""
    print("\nKey Findings:\n")
    
    # Print baseline performance
    print("Baseline Performance:")
    for metric, value in results["baseline"].items():
        print(f"{metric}: {value:.4f}")
    
    # Analyze impact of bit flips
    print("\nImpact of Bit Flips:")
    for fault_str, metrics in results["faults"].items():
        # Extract corruption rate from fault string
        # Format: "WeightCorruption(pattern='bit_flip', corruption_rate=0.001, target_layers=None, target_weights=['weight'])"
        rate = float(fault_str.split("corruption_rate=")[1].split(",")[0])
        print(f"\nCorruption Rate: {rate:.3f}")
        for metric, value in metrics.items():
            baseline = results["baseline"][metric]
            impact = ((value - baseline) / baseline) * 100
            print(f"{metric}: {value:.4f} (Impact: {impact:+.2f}%)")

def main():
    """Main function to run the experiment and analyze results."""
    # Run experiment
    results = run_experiment()
    
    # Analyze results
    analyze_results(results)
    
    # Plot results
    plot_results(
        results,
        output_dir="imdb_bit_flip_results",
        plot_types=["fault_type_impact", "severity_impact"]
    )

if __name__ == "__main__":
    main() 