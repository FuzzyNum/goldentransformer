"""
GoldenTransformer: A framework for fault injection and resiliency analysis of LLMs.
"""

from goldentransformer.core.fault_injector import FaultInjector
from goldentransformer.core.experiment_runner import ExperimentRunner
from goldentransformer.faults.attention_fault import AttentionFault
from goldentransformer.faults.weight_corruption import WeightCorruption
from goldentransformer.metrics.accuracy import Accuracy
from goldentransformer.metrics.perplexity import Perplexity
from goldentransformer.visualization.plotter import plot_results

__version__ = "0.1.0"

__all__ = [
    "FaultInjector",
    "ExperimentRunner",
    "AttentionFault",
    "WeightCorruption",
    "Accuracy",
    "Perplexity",
    "plot_results"
] 