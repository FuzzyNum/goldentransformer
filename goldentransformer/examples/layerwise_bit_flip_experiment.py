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
    num_trials = 30
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
    std_baseline = all_baseline.std(axis=0)
    mean_faulted = all_faulted.mean(axis=0)
    std_faulted = all_faulted.std(axis=0)
    layers = np.arange(num_layers)
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.errorbar(layers, mean_baseline, yerr=std_baseline*(1.96/np.sqrt(num_trials)), label='Baseline Perplexity', fmt='-o',color='black')
    plt.errorbar(layers, mean_faulted, yerr=std_faulted*(1.96/np.sqrt(num_trials)), label='Faulted Perplexity', fmt='-o',color='red')
    plt.xlabel('Layer Index', fontsize=16)
    plt.ylabel('Perplexity', fontsize=16)
    plt.title(f'Perplexity vs. Layer Index (Bit-Flip Fault, Mean ± Std) - {model_name}', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.2)
    plt.gca().spines['bottom'].set_linewidth(1.2)
    plt.legend(frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(layer_results_dir, f"perplexity_vs_layer_with_errorbars_{model_name}.png"))
    plt.close()
    print(f"Aggregate plot with error bars saved to {os.path.join(layer_results_dir, f'perplexity_vs_layer_with_errorbars_{model_name}.png')}")

if __name__ == "__main__":
    main() 