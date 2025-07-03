"""
Experiment runner for fault injection experiments.
"""

import os
import json
import time
import torch
import logging
import inspect
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from goldentransformer.core.fault_injector import FaultInjector
from goldentransformer.faults.base_fault import BaseFault
from goldentransformer.metrics.base_metric import BaseMetric

class ExperimentRunner:
    """Class for running fault injection experiments."""
    
    def __init__(
        self,
        injector: FaultInjector,
        faults: List[BaseFault],
        metrics: List[BaseMetric],
        dataset: Union[str, torch.utils.data.Dataset],
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the experiment runner.
        
        Args:
            injector: Fault injector instance
            faults: List of faults to inject
            metrics: List of metrics to compute
            dataset: Dataset to evaluate on
            output_dir: Directory to save results
            batch_size: Batch size for evaluation
            num_samples: Number of samples to evaluate (None for all)
            device: Device to run experiments on
        """
        self.injector = injector
        self.faults = faults
        self.metrics = metrics
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"experiment_results_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Move model to device
        self.injector.model.to(device)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        self.logger.info("Starting experiment")
        results = {
            'baseline': self._run_baseline(),
            'fault_results': []
        }
        
        for fault in self.faults:
            self.logger.info(f"Running experiment with fault: {fault}")
            fault_results = self._run_with_fault(fault)
            results['fault_results'].append(fault_results)
            
            # Save intermediate results
            self._save_results(results)
        
        self.logger.info("Experiment completed")
        return results
    
    def _run_baseline(self) -> Dict[str, Any]:
        """Run baseline evaluation without faults."""
        self.logger.info("Running baseline evaluation")
        baseline_results = self._evaluate_model()
        return baseline_results
    
    def _run_with_fault(self, fault: BaseFault) -> Dict[str, Any]:
        """Run evaluation with a specific fault."""
        fault_results = {
            'fault_info': str(fault),
            'metrics': {}
        }
        
        try:
            # Inject fault
            self.injector.inject_fault(fault)
            
            # Evaluate with fault
            fault_results['metrics'] = self._evaluate_model()
            
            # Revert fault
            self.injector.revert_fault()
            
        except Exception as e:
            self.logger.error(f"Error during fault injection: {str(e)}")
            fault_results['error'] = str(e)
        
        return fault_results
    
    def _evaluate_model(self) -> Dict[str, Any]:
        """Evaluate model performance."""
        results = {}
        
        # Create dataloader
        if isinstance(self.dataset, str):
            # Handle dataset loading based on name
            raise NotImplementedError("Dataset loading from string not implemented")
        else:
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        # Evaluate on batches
        for i, batch in enumerate(dataloader):
            if self.num_samples is not None and i * self.batch_size >= self.num_samples:
                break
                
            # Handle tuple or list batches (TensorDataset)
            if isinstance(batch, (tuple, list)):
                # Always map as input_ids, attention_mask, labels (in that order)
                keys = ["input_ids", "attention_mask", "labels"]
                batch = {k: v.to(self.device) for k, v in zip(keys, batch) if v is not None}
            else:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Compute metrics
            for metric in self.metrics:
                compute_sig = inspect.signature(metric.compute)
                params = list(compute_sig.parameters.keys())
                # Determine if the first argument expects outputs or model
                expects_outputs = params[0] in ("outputs", "logits", "predictions")
                expects_batch_size = params[-1] == 'batch_size'
                if expects_outputs:
                    outputs = self.injector.model(**batch)
                    if expects_batch_size:
                        metric_results = metric.compute(outputs, batch, self.batch_size)
                    else:
                        metric_results = metric.compute(outputs, batch)
                else:
                    if expects_batch_size:
                        metric_results = metric.compute(self.injector.model, batch, self.batch_size)
                    else:
                        metric_results = metric.compute(self.injector.model, batch)
                if not isinstance(metric_results, dict):
                    metric_results = {metric.__class__.__name__: metric_results}
                results[metric.__class__.__name__] = metric_results
        
        # Get metric summaries
        for metric in self.metrics:
            if hasattr(metric, 'get_summary'):
                results[metric.__class__.__name__].update(metric.get_summary())
            metric.reset()
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results to file."""
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def __str__(self) -> str:
        """String representation of the experiment runner."""
        return (f"{self.__class__.__name__}(num_faults={len(self.faults)}, "
                f"num_metrics={len(self.metrics)}, output_dir={self.output_dir})") 