import torch
import os
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np
import copy
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    pipeline
)
from datasets import load_dataset
from goldentransformer import FaultInjector, ExperimentRunner
from goldentransformer.faults import WeightCorruption
from goldentransformer.metrics import Accuracy
from goldentransformer.visualization.plotter import plot_results

def create_fresh_model_copy(model, model_id):
    """Create a fresh copy of the model to ensure complete state isolation."""
    # Create a deep copy of the model
    fresh_model = copy.deepcopy(model)
    # Ensure it's in eval mode
    fresh_model.eval()
    return fresh_model

def prepare_imdb_dataset(tokenizer, max_length=128, num_samples=100):
    """Prepare IMDB dataset for classification."""
    dataset = load_dataset("imdb", split="test")
    # Take a subset for faster experiments
    dataset = dataset.select(range(num_samples))
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    input_ids = torch.tensor(tokenized_dataset["input_ids"])
    attention_mask = torch.tensor(tokenized_dataset["attention_mask"])
    labels = torch.tensor(dataset["label"])
    
    return torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

def run_layerwise_experiment(model_name, model, tokenizer, dataset, results_dir):
    """Run layerwise bit-flip experiments for a single model."""
    print(f"\n{'='*50}")
    print(f"Running experiments for {model_name}")
    print(f"{'='*50}")
    
    # Store accuracy results for plotting
    layer_indices = []
    baseline_accuracies = []
    faulted_accuracies = []
    
    # Run experiment for each of the first 10 layers
    for layer_idx in range(10):
        print(f"\nRunning experiment for layer {layer_idx}...")
        
        # Create a fresh model copy for each experiment to ensure complete state isolation
        fresh_model = create_fresh_model_copy(model, model_name)
        
        # Create a fresh FaultInjector for each experiment to ensure consistent baselines
        # Also ensure model is in eval mode and gradients are disabled
        with torch.no_grad():
            injector = FaultInjector(fresh_model)
        
        faults = [
            WeightCorruption(
                pattern="bit_flip",
                corruption_rate=0.1,  # 10% severity for more visible effects
                target_layers=[layer_idx]
            )
        ]
        metrics = [Accuracy()]
        
        # Use millisecond precision for output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment_output_dir = os.path.join(results_dir, f"{model_name}_layer_{layer_idx}_{timestamp}")
        
        experiment = ExperimentRunner(
            injector=injector,
            faults=faults,
            metrics=metrics,
            dataset=dataset,
            output_dir=experiment_output_dir
        )
        
        results = experiment.run()
        plot_results(results, str(experiment.output_dir))
        
        # The experiment output directory is already in the correct location
        print(f"Results for {model_name} layer {layer_idx} saved to {experiment.output_dir}")
        
        # Aggregate accuracy results
        # Baseline
        baseline = results["baseline"].get("Accuracy")
        if isinstance(baseline, dict):
            baseline = list(baseline.values())[0]
        # Faulted
        faulted = results["fault_results"][0]["metrics"].get("Accuracy")
        if isinstance(faulted, dict):
            faulted = list(faulted.values())[0]
            
        layer_indices.append(layer_idx)
        baseline_accuracies.append(baseline)
        faulted_accuracies.append(faulted)
        
        # Ensure complete cleanup
        del injector, experiment, fresh_model
    
    return layer_indices, baseline_accuracies, faulted_accuracies

def plot_aggregate_results(all_results, results_dir):
    """Plot aggregate results for all models."""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green']
    models = list(all_results.keys())
    
    # Plot 1: Individual model accuracy vs layer
    for i, model_name in enumerate(models):
        layer_indices, baseline_acc, faulted_acc = all_results[model_name]
        ax1.plot(layer_indices, baseline_acc, marker='o', label=f'{model_name} Baseline', color=colors[i], linestyle='-')
        ax1.plot(layer_indices, faulted_acc, marker='s', label=f'{model_name} Faulted', color=colors[i], linestyle='--')
    
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Layer Index (All Models)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Accuracy degradation (baseline - faulted)
    for i, model_name in enumerate(models):
        layer_indices, baseline_acc, faulted_acc = all_results[model_name]
        degradation = [b - f for b, f in zip(baseline_acc, faulted_acc)]
        ax2.plot(layer_indices, degradation, marker='o', label=model_name, color=colors[i])
    
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Accuracy Degradation')
    ax2.set_title('Accuracy Degradation vs Layer Index')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Percentage degradation
    for i, model_name in enumerate(models):
        layer_indices, baseline_acc, faulted_acc = all_results[model_name]
        pct_degradation = [(b - f) / b * 100 if b > 0 else 0 for b, f in zip(baseline_acc, faulted_acc)]
        ax3.plot(layer_indices, pct_degradation, marker='o', label=model_name, color=colors[i])
    
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Accuracy Degradation (%)')
    ax3.set_title('Percentage Accuracy Degradation vs Layer Index')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Model comparison (average degradation per model)
    avg_degradations = []
    for model_name in models:
        layer_indices, baseline_acc, faulted_acc = all_results[model_name]
        degradation = [b - f for b, f in zip(baseline_acc, faulted_acc)]
        avg_degradations.append(np.mean(degradation))
    
    ax4.bar(models, avg_degradations, color=colors[:len(models)])
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Average Accuracy Degradation')
    ax4.set_title('Average Accuracy Degradation per Model')
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "classification_aggregate_results.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Aggregate plots saved to {os.path.join(results_dir, 'classification_aggregate_results.png')}")

def main():
    # Define models to test - using IMDB fine-tuned models for realistic results
    models_config = {
        "BERT-IMDB": "textattack/bert-base-uncased-imdb",
        "DistilBERT-IMDB": "textattack/distilbert-base-uncased-imdb", 
        "RoBERTa-IMDB": "textattack/roberta-base-imdb"
    }
    
    # Prepare output directory
    base_results_dir = "goldentransformer/experiment_results"
    classification_results_dir = os.path.join(base_results_dir, "classification_layer_experiment_results")
    os.makedirs(classification_results_dir, exist_ok=True)
    
    # Store results for all models
    all_results = {}
    
    # Run experiments for each model
    for model_name, model_id in models_config.items():
        print(f"\nLoading {model_name} ({model_id})...")
        
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Prepare dataset
        dataset = prepare_imdb_dataset(tokenizer)
        
        # Run layerwise experiments
        layer_indices, baseline_acc, faulted_acc = run_layerwise_experiment(
            model_name, model, tokenizer, dataset, classification_results_dir
        )
        
        all_results[model_name] = (layer_indices, baseline_acc, faulted_acc)
        
        # Clean up to free memory
        del model, tokenizer, dataset
    
    # Plot aggregate results
    plot_aggregate_results(all_results, classification_results_dir)
    
    # Save numerical results
    results_summary = {}
    for model_name, (layer_indices, baseline_acc, faulted_acc) in all_results.items():
        results_summary[model_name] = {
            "layers": layer_indices,
            "baseline_accuracy": baseline_acc,
            "faulted_accuracy": faulted_acc,
            "average_baseline": np.mean(baseline_acc),
            "average_faulted": np.mean(faulted_acc),
            "average_degradation": np.mean([b - f for b, f in zip(baseline_acc, faulted_acc)])
        }
    
    with open(os.path.join(classification_results_dir, "results_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults summary saved to {os.path.join(classification_results_dir, 'results_summary.json')}")
    print(f"\nAll experiments completed! Results saved in: {classification_results_dir}")

if __name__ == "__main__":
    main() 