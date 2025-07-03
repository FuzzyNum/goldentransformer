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

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    injector = FaultInjector(model)
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you today?",
        "This is a test of the fault injection framework.",
        "The weather is beautiful outside.",
        "I love programming and machine learning."
    ]
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

    # Prepare output subfolder for all results
    base_results_dir = "goldentransformer/experiment_results"
    layer_results_dir = os.path.join(base_results_dir, "layer_experiment_results")
    os.makedirs(layer_results_dir, exist_ok=True)

    # Store perplexity results for plotting
    layer_indices = []
    baseline_perplexities = []
    faulted_perplexities = []

    # Run experiment for each of the first 10 layers
    for layer_idx in range(10):
        print(f"\nRunning experiment for layer {layer_idx}...")
        faults = [
            WeightCorruption(
                pattern="bit_flip",
                corruption_rate=0.000000000001,
                target_layers=[layer_idx]
            )
        ]
        metrics = [Perplexity()]
        # Use millisecond precision for output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment_output_dir = os.path.join(base_results_dir, f"experiment_results_{timestamp}")
        experiment = ExperimentRunner(
            injector=injector,
            faults=faults,
            metrics=metrics,
            dataset=dataset,
            output_dir=experiment_output_dir
        )
        results = experiment.run()
        plot_results(results, str(experiment.output_dir))
        # Move the experiment result folder into the layer_experiment_results subfolder
        dest = os.path.join(layer_results_dir, experiment.output_dir.name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(str(experiment.output_dir), dest)
        print(f"Results for layer {layer_idx} saved to {dest}")
        # Aggregate perplexity results
        # Baseline
        baseline = results["baseline"].get("Perplexity")
        if isinstance(baseline, dict):
            baseline = list(baseline.values())[0]
        # Faulted
        faulted = results["fault_results"][0]["metrics"].get("Perplexity")
        if isinstance(faulted, dict):
            faulted = list(faulted.values())[0]
        layer_indices.append(layer_idx)
        baseline_perplexities.append(baseline)
        faulted_perplexities.append(faulted)

    # Plot aggregate perplexity vs. layer
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, baseline_perplexities, marker='o', label='Baseline Perplexity')
    plt.plot(layer_indices, faulted_perplexities, marker='o', label='Faulted Perplexity')
    plt.xlabel('Layer Index')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. Layer Index (Bit-Flip Fault)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(layer_results_dir, "perplexity_vs_layer.png"))
    plt.close()
    print(f"Aggregate plot saved to {os.path.join(layer_results_dir, 'perplexity_vs_layer.png')}")

if __name__ == "__main__":
    main() 