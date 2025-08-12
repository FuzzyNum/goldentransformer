import torch
import os
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from goldentransformer import FaultInjector, ExperimentRunner
from goldentransformer.faults import WeightCorruption
from goldentransformer.metrics import Accuracy
from goldentransformer.visualization.plotter import plot_results
import sys
import random

def prepare_classification_dataset(tokenizer, max_length=64, num_samples=50):
    """Prepare a binary classification dataset for DistilBERT."""
    # Use a subset of IMDB for binary classification
    dataset = load_dataset("imdb", split="test")
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

def main():
    num_trials = 30  # Increased for 95% confidence interval validity
    seeds = [42 + i for i in range(num_trials)]
    
    # Use a proper classification model
    model_name = "textattack/distilbert-base-uncased-imdb"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare classification dataset
    dataset = prepare_classification_dataset(tokenizer, num_samples=50)
    
    base_results_dir = "goldentransformer/experiment_results"
    layer_results_dir = os.path.join(base_results_dir, f"layer_classification_results_{model_name}")
    os.makedirs(layer_results_dir, exist_ok=True)
    
    num_layers = 10
    all_baseline = []
    all_faulted = []
    
    for trial, seed in enumerate(seeds):
        print(f"\n=== Trial {trial+1}/{num_trials} (seed={seed}) ===")
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        layer_indices = []
        baseline_accuracies = []
        faulted_accuracies = []
        
        for layer_idx in range(num_layers):
            print(f"\nRunning experiment for layer {layer_idx}...")
            
            # Create a fresh model copy for each layer experiment
            fresh_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            fresh_injector = FaultInjector(fresh_model)
            
            faults = [
                WeightCorruption(
                    pattern="random",
                    corruption_rate=0.05,
                    target_layers=[layer_idx]
                )
            ]
            
            metrics = [Accuracy()]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            experiment_output_dir = os.path.join(base_results_dir, f"experiment_results_{model_name}_{timestamp}_trial{trial}")
            
            experiment = ExperimentRunner(
                injector=fresh_injector,
                faults=faults,
                metrics=metrics,
                dataset=dataset,
                output_dir=experiment_output_dir
            )
            
            results = experiment.run()
            plot_results(results, str(experiment.output_dir))
            
            dest = os.path.join(layer_results_dir, experiment.output_dir.name)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.move(str(experiment.output_dir), dest)
            
            print(f"Results for layer {layer_idx} saved to {dest}")
            
            baseline = results["baseline"].get("Accuracy")
            if isinstance(baseline, dict):
                baseline = list(baseline.values())[0]
            
            faulted = results["fault_results"][0]["metrics"].get("Accuracy")
            if isinstance(faulted, dict):
                faulted = list(faulted.values())[0]
            
            layer_indices.append(layer_idx)
            baseline_accuracies.append(baseline)
            faulted_accuracies.append(faulted)
            
            # Clean up the fresh model to free memory
            del fresh_model
            del fresh_injector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        all_baseline.append(baseline_accuracies)
        all_faulted.append(faulted_accuracies)
    
    # Aggregate results
    all_baseline = np.array(all_baseline)
    all_faulted = np.array(all_faulted)
    
    mean_baseline = all_baseline.mean(axis=0)
    sem_baseline = all_baseline.std(axis=0) / np.sqrt(num_trials)  # Standard Error of Mean
    mean_faulted = all_faulted.mean(axis=0)
    sem_faulted = all_faulted.std(axis=0) / np.sqrt(num_trials)  # Standard Error of Mean
    
    layers = np.arange(num_layers)
    
    # Plot with error bars - Fixed visualization
    plt.figure(figsize=(8, 8))  # Square aspect ratio
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Plot with different markers and line styles
    plt.errorbar(layers, mean_baseline, yerr=sem_baseline, 
                label='Baseline', fmt='o-', color='blue', 
                markersize=8, capsize=4, capthick=1.5, linewidth=2)
    plt.errorbar(layers, mean_faulted, yerr=sem_faulted, 
                label='Corrupted', fmt='s--', color='red', 
                markersize=8, capsize=4, capthick=1.5, linewidth=2)
    
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Ensure axes boundaries are visible (box around graph)
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    plt.tight_layout()
    # Fix filename by replacing slashes with underscores
    safe_model_name = model_name.replace('/', '_')
    plot_filename = f"accuracy_vs_layer_with_errorbars_{safe_model_name}_minimalist.png"
    plt.savefig(os.path.join(layer_results_dir, plot_filename), dpi=300)
    plt.close()
    print(f"Minimalist plot with error bars saved to {os.path.join(layer_results_dir, plot_filename)}")

if __name__ == "__main__":
    main() 