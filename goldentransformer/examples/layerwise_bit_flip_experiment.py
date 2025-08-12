import torch
import os
import shutil
import json
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from goldentransformer import FaultInjector, ExperimentRunner
from goldentransformer.faults import WeightCorruption
from goldentransformer.metrics import Perplexity
from goldentransformer.visualization.plotter import plot_results
import sys
import random
import numpy as np
from datasets import load_dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_trials = 30  # Increased for 95% confidence interval validity
    seeds = [42 + i for i in range(num_trials)]
    # Allow model name as a command-line argument
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else tokenizer.unk_token
    if hasattr(model.config, 'pad_token_id'):
        model.config.pad_token_id = tokenizer.pad_token_id
    # Use a subset of WikiText for a more robust dataset
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:100]")
    test_texts = [x["text"] for x in wikitext if x["text"].strip()]
    encodings = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"]
    )
    base_results_dir = "goldentransformer/experiment_results"
    layer_results_dir = os.path.join(base_results_dir, f"layer_experiment_results_{model_name}")
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
        baseline_perplexities = []
        faulted_perplexities = []
        
        for layer_idx in range(num_layers):
            print(f"\nRunning experiment for layer {layer_idx}...")
            # Create a fresh model for each layer
            fresh_model = AutoModelForCausalLM.from_pretrained(model_name)
            fresh_model.eval()
            fresh_model.to(device)
            
            # Set temperature to 0 for deterministic output
            if hasattr(fresh_model.config, 'temperature'):
                fresh_model.config.temperature = 0.0
            
            injector = FaultInjector(fresh_model)
            faults = [
                WeightCorruption(
                    pattern="bit_flip",
                    corruption_rate=0.01,
                    target_layers=[layer_idx]
                )
            ]
            metrics = [Perplexity()]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            experiment_output_dir = os.path.join(base_results_dir, f"experiment_results_{model_name}_{timestamp}_trial{trial}")
            experiment = ExperimentRunner(
                injector=injector,
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
            baseline = results["baseline"].get("Perplexity")
            if isinstance(baseline, dict):
                baseline = list(baseline.values())[0]
            faulted = results["fault_results"][0]["metrics"].get("Perplexity")
            if isinstance(faulted, dict):
                faulted = list(faulted.values())[0]
            layer_indices.append(layer_idx)
            baseline_perplexities.append(baseline)
            faulted_perplexities.append(faulted)
        all_baseline.append(baseline_perplexities)
        all_faulted.append(faulted_perplexities)
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
                label='Faulted', fmt='s--', color='red', 
                markersize=8, capsize=4, capthick=1.5, linewidth=2)
    
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Perplexity', fontsize=14)
    plt.yscale('log')
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
    plt.savefig(os.path.join(layer_results_dir, f"perplexity_vs_layer_with_errorbars_{model_name}.png"), dpi=300)
    plt.close()
    print(f"Aggregate plot with error bars saved to {os.path.join(layer_results_dir, f'perplexity_vs_layer_with_errorbars_{model_name}.png')}")

    # Plot minimalist version - Fixed visualization
    plt.figure(figsize=(8, 8))  # Square aspect ratio
    plt.plot(layer_indices, baseline_perplexities, marker='o', linestyle='-', 
             label='Baseline', markersize=8, linewidth=2)
    plt.plot(layer_indices, faulted_perplexities, marker='s', linestyle='--', 
             label='Corrupted', markersize=8, linewidth=2)
    plt.yscale('log')
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    plt.grid(True, alpha=0.3)
    
    # Ensure axes boundaries are visible
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    plt.tight_layout()
    plt.savefig(os.path.join(layer_results_dir, f"perplexity_vs_layer_minimalist_{model_name}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main() 