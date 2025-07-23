import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_final_plot():
    """Generate the final plot with error bars from existing classification experiment results."""
    
    base_dir = "goldentransformer/experiment_results/layer_classification_results_textattack/distilbert-base-uncased-imdb"
    
    num_trials = 10
    num_layers = 5
    
    all_baseline = []
    all_faulted = []
    
    # Collect results from all trials
    for trial in range(num_trials):
        trial_baseline = []
        trial_faulted = []
        
        for layer in range(num_layers):
            # Find the directory for this trial and layer
            trial_dirs = [d for d in os.listdir(base_dir) if f"trial{trial}" in d]
            if not trial_dirs:
                continue
                
            # For layer experiments, we need to find the right subdirectory
            # The layer index is encoded in the timestamp, so we'll use the order
            trial_dirs.sort()  # Sort by timestamp
            if layer < len(trial_dirs):
                trial_dir = trial_dirs[layer]
                results_file = os.path.join(base_dir, trial_dir, "results.json")
                
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    # Extract baseline accuracy
                    baseline = results.get("baseline", {}).get("Accuracy", 0)
                    if isinstance(baseline, dict):
                        baseline = list(baseline.values())[0]
                    
                    # Extract faulted accuracy
                    faulted = results.get("fault_results", [{}])[0].get("metrics", {}).get("Accuracy", 0)
                    if isinstance(faulted, dict):
                        faulted = list(faulted.values())[0]
                    
                    trial_baseline.append(baseline)
                    trial_faulted.append(faulted)
        
        if len(trial_baseline) == num_layers and len(trial_faulted) == num_layers:
            all_baseline.append(trial_baseline)
            all_faulted.append(trial_faulted)
    
    if not all_baseline:
        print("No results found!")
        return
    
    # Convert to numpy arrays
    all_baseline = np.array(all_baseline)
    all_faulted = np.array(all_faulted)
    
    # Calculate statistics
    mean_baseline = all_baseline.mean(axis=0)
    std_baseline = all_baseline.std(axis=0)
    mean_faulted = all_faulted.mean(axis=0)
    std_faulted = all_faulted.std(axis=0)
    
    layers = np.arange(num_layers)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.errorbar(layers, mean_baseline, yerr=std_baseline*1.96/np.sqrt(num_trials), label='Baseline Accuracy', fmt='-o', capsize=5)
    plt.errorbar(layers, mean_faulted, yerr=std_faulted*1.96/np.sqrt(num_trials), label='Faulted Accuracy', fmt='-o', capsize=5)
    plt.xlabel('Layer Index', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy vs. Layer Index (Bit-Flip Fault, Mean ± Std)\nDistilBERT-IMDB Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_filename = "accuracy_vs_layer_with_errorbars_distilbert_imdb.png"
    plt.savefig(os.path.join(base_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Final plot saved to: {os.path.join(base_dir, plot_filename)}")
    print(f"Number of trials processed: {len(all_baseline)}")
    print(f"Baseline accuracy range: {mean_baseline.min():.3f} - {mean_baseline.max():.3f}")
    print(f"Faulted accuracy range: {mean_faulted.min():.3f} - {mean_faulted.max():.3f}")

if __name__ == "__main__":
    generate_final_plot() 