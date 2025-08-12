"""
Visualization module for plotting experiment results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
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

def calculate_sem(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculate standard error of the mean (SEM).
    
    Args:
        data: Array of data
        axis: Axis along which to calculate SEM
    
    Returns:
        Standard error of the mean
    """
    return np.std(data, axis=axis) / np.sqrt(data.shape[axis])

def plot_scientific_results(
    x_data: List[float],
    y_data: List[List[float]],  # List of datasets, each containing y values
    y_errors: List[List[float]],  # List of error bars for each dataset
    labels: List[str],  # Labels for each dataset
    x_label: str,
    y_label: str,
    output_path: str,
    x_units: str = "",
    y_units: str = "",
    log_scale: bool = False,
    include_zero: bool = True,
    error_type: str = "SD"  # "SD" for standard deviation, "SEM" for standard error of mean
) -> None:
    """
    Create a scientific plot following standard conventions.
    
    Args:
        x_data: X-axis data points
        y_data: List of y-axis datasets
        y_errors: List of error bar datasets (standard deviation or SEM)
        labels: Labels for each dataset
        x_label: X-axis label
        y_label: Y-axis label
        output_path: Path to save the plot
        x_units: Units for x-axis (in parentheses)
        y_units: Units for y-axis (in parentheses)
        log_scale: Whether to use log scale for y-axis
        include_zero: Whether to include zero on y-axis
        error_type: Type of error bars ("SD" or "SEM")
    """
    # Ensure no more than 6 datasets
    if len(y_data) > 6:
        raise ValueError("Maximum 6 datasets allowed per plot")
    
    # Set up the figure with square aspect ratio
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define line styles and markers for different datasets
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot each dataset
    for i, (y_vals, y_err, label, ls, marker, color) in enumerate(zip(y_data, y_errors, labels, line_styles, markers, colors)):
        ax.errorbar(
            x_data, y_vals, yerr=y_err,
            fmt=f'{marker}{ls}',
            color=color,
            markersize=6,
            capsize=3,
            capthick=1,
            linewidth=1.5,
            label=label
        )
    
    # Set axis labels with units
    x_label_with_units = f"{x_label} ({x_units})" if x_units else x_label
    y_label_with_units = f"{y_label} ({y_units})" if y_units else y_label
    
    ax.set_xlabel(x_label_with_units, fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel(y_label_with_units, fontsize=12, fontfamily='Times New Roman')
    
    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Include zero on y-axis if requested and not using log scale
    if include_zero and not log_scale:
        y_min = min([min(y) for y in y_data])
        y_max = max([max(y) for y in y_data])
        y_range = y_max - y_min
        ax.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)
    
    # Set major tick divisions as multiples of 1, 2, or 5
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    
    # Format tick labels with scientific notation for large/small numbers
    def format_func(value, tick_number):
        if abs(value) >= 1e4 or (abs(value) <= 1e-4 and value != 0):
            return f'{value:.0e}'
        elif abs(value) < 1:
            return f'{value:.3f}'
        else:
            return f'{value:.1f}'
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    # Set font for tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    # Add legend within axis boundaries (right side)
    ax.legend(
        loc='center right',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set spine linewidth
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Remove title (as requested)
    ax.set_title('')
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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