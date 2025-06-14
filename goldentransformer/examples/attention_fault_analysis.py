"""
Example experiment: Analyzing the impact of attention mechanism faults on LLM performance.
This experiment demonstrates how different types of attention faults affect model accuracy
and how the model's resiliency varies across different layers.
"""

import torch
import numpy as np
from goldentransformer import FaultInjector, ExperimentRunner
from goldentransformer.faults import AttentionFault
from goldentransformer.metrics import Accuracy, Perplexity, ResponseLatency
from goldentransformer.visualization import plot_results
from goldentransformer.utils import save_results

def create_attention_faults():
    """Create a comprehensive set of attention faults to test."""
    faults = []
    
    # Test different attention fault types
    fault_types = [
        "mask_corruption",  # Corrupt attention masks
        "head_dropout",     # Drop entire attention heads
        "query_corruption", # Corrupt query vectors
        "key_corruption",   # Corrupt key vectors
        "value_corruption"  # Corrupt value vectors
    ]
    
    # Test different severity levels
    severities = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Create faults for each layer (assuming 12-layer model)
    for layer_idx in range(12):
        for fault_type in fault_types:
            for severity in severities:
                faults.append(
                    AttentionFault(
                        layer_idx=layer_idx,
                        fault_type=fault_type,
                        severity=severity,
                        head_idx=None  # Test all heads
                    )
                )
    
    return faults

def run_experiment():
    """Run the attention fault analysis experiment."""
    # Initialize the fault injector with GPT-2
    injector = FaultInjector(
        model="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create attention faults
    faults = create_attention_faults()
    
    # Set up metrics
    metrics = [
        Accuracy(),
        Perplexity(),
        ResponseLatency()
    ]
    
    # Initialize experiment runner
    experiment = ExperimentRunner(
        injector=injector,
        faults=faults,
        metrics=metrics,
        dataset="wikitext-2",
        batch_size=32,
        num_samples=1000
    )
    
    # Run the experiment
    print("Starting attention fault analysis experiment...")
    results = experiment.run()
    
    # Save results
    save_results(results, "attention_fault_analysis_results.json")
    
    # Generate visualizations
    plot_results(
        results,
        output_dir="attention_fault_analysis_plots",
        plot_types=["layer_impact", "fault_type_impact", "severity_impact"]
    )
    
    return results

def analyze_results(results):
    """Analyze and print key findings from the experiment."""
    print("\nKey Findings:")
    
    # Find most vulnerable layer
    layer_impacts = results.group_by("layer_idx").mean("accuracy")
    most_vulnerable_layer = layer_impacts.idxmin()
    print(f"Most vulnerable layer: {most_vulnerable_layer}")
    
    # Find most damaging fault type
    fault_type_impacts = results.group_by("fault_type").mean("accuracy")
    most_damaging_fault = fault_type_impacts.idxmin()
    print(f"Most damaging fault type: {most_damaging_fault}")
    
    # Analyze severity impact
    severity_impacts = results.group_by("severity").mean("accuracy")
    print("\nSeverity Impact on Accuracy:")
    for severity, accuracy in severity_impacts.items():
        print(f"Severity {severity}: {accuracy:.3f}")
    
    # Analyze latency impact
    latency_impacts = results.group_by("fault_type").mean("latency")
    print("\nFault Type Impact on Latency:")
    for fault_type, latency in latency_impacts.items():
        print(f"{fault_type}: {latency:.3f}ms")

if __name__ == "__main__":
    # Run the experiment
    results = run_experiment()
    
    # Analyze and print results
    analyze_results(results) 