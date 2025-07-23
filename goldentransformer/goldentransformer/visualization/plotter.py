"""
Visualization module for plotting experiment results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import re

def plot_results(
    results: Dict[str, Any],
    output_dir: str,
    plot_types: Optional[List[str]] = None
) -> None:
    """
    Plot experiment results and save them to the specified output directory.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
        plot_types: List of plot types to generate (default: all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame (patched for actual structure)
    df = _results_to_dataframe(results)
    
    # Set plot style
    # plt.style.use("seaborn")  # Use default style for compatibility
    sns.set_palette("husl")
    
    # Generate plots
    if plot_types is None:
        plot_types = ["severity_impact", "metric_values"]
    
    for plot_type in plot_types:
        if plot_type == "severity_impact":
            _plot_severity_impact(df, output_dir)
        elif plot_type == "fault_type_impact":
            _plot_fault_type_impact(df, output_dir)
        elif plot_type == "metric_values":
            _plot_metric_values(df, output_dir)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

def _results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert results dictionary to DataFrame.
    Args:
        results: Experiment results dictionary
    Returns:
        pd.DataFrame: Results in DataFrame format
    """
    # Get baseline metrics
    baseline = results["baseline"]
    # Create DataFrame rows
    rows = []
    # Support both old and new result structures
    fault_results = results.get("fault_results")
    if fault_results is not None:
        # New structure: list of dicts with 'fault_info' and 'metrics'
        for fault in fault_results:
            fault_info = fault.get("fault_info", "")
            # Try to extract corruption_rate from fault_info string
            match = re.search(r"corruption_rate=([0-9.eE+-]+)", fault_info)
            if match:
                rate = float(match.group(1))
            else:
                rate = None
            metrics = fault.get("metrics", {})
            for metric_name, value in metrics.items():
                # If metric is a dict, get the first value
                if isinstance(value, dict):
                    value = list(value.values())[0]
                baseline_value = baseline.get(metric_name, 0)
                # If baseline_value is a dict, get the first value
                if isinstance(baseline_value, dict):
                    baseline_value = list(baseline_value.values())[0]
                # Avoid division by zero
                if baseline_value != 0:
                    impact = ((value - baseline_value) / baseline_value) * 100
                else:
                    impact = 0
                rows.append({
                    "corruption_rate": rate,
                    "metric": metric_name,
                    "value": value,
                    "baseline": baseline_value,
                    "impact": impact
                })
    else:
        # Old structure: dict with 'faults' key
        for fault_str, metrics in results["faults"].items():
            # Extract fault information
            rate = float(fault_str.split("corruption_rate=")[1].split(",")[0])
            for metric_name, value in metrics.items():
                baseline_value = baseline[metric_name]
                impact = ((value - baseline_value) / baseline_value) * 100
                rows.append({
                    "corruption_rate": rate,
                    "metric": metric_name,
                    "value": value,
                    "baseline": baseline_value,
                    "impact": impact
                })
    return pd.DataFrame(rows)              

def _plot_severity_impact(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot the impact of fault severity on model performance.
    
    Args:
        df: Results DataFrame
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot impact for each metric
    for metric in df["metric"].unique():
        metric_df = df[df["metric"] == metric]
        plt.plot(
            metric_df["corruption_rate"],
            metric_df["impact"],
            marker="o",
            label=metric
        )
    
    plt.xlabel("Corruption Rate")
    plt.ylabel("Impact on Performance (%)")
    plt.title("Impact of Bit Flip Corruption Rate on Model Performance")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "severity_impact.png"))
    plt.close()

def _plot_fault_type_impact(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot the impact of different fault types on model performance.
    
    Args:
        df: Results DataFrame
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot impact for each metric
    for metric in df["metric"].unique():
        metric_df = df[df["metric"] == metric]
        plt.plot(
            metric_df["corruption_rate"],
            metric_df["impact"],
            marker="o",
            label=metric
        )
    
    plt.xlabel("Corruption Rate")
    plt.ylabel("Impact on Performance (%)")
    plt.title("Impact of Bit Flip Faults on Model Performance")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "fault_type_impact.png"))
    plt.close()

def _plot_metric_values(df, output_dir):
    """
    Plot the raw metric values (e.g., perplexity) for each corruption rate.
    """
    plt.figure(figsize=(10, 6))
    for metric in df["metric"].unique():
        metric_df = df[df["metric"] == metric]
        plt.plot(
            metric_df["corruption_rate"],
            metric_df["value"],
            marker="o",
            label=metric
        )
    plt.xlabel("Corruption Rate")
    plt.ylabel("Metric Value")
    plt.title("Raw Metric Values vs. Corruption Rate")
    plt.yscale("log")  # Use log scale for large ranges
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "metric_values.png"))
    plt.close() 