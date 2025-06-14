"""
Fault injector for injecting faults into transformer models.
"""

import torch
import torch.nn as nn
from typing import Optional
from goldentransformer.faults.base_fault import BaseFault

class FaultInjector:
    """Class for injecting faults into transformer models."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize the fault injector.
        
        Args:
            model: The model to inject faults into
        """
        self.model = model
        self.current_fault: Optional[BaseFault] = None
    
    def inject_fault(self, fault: BaseFault) -> None:
        """
        Inject a fault into the model.
        
        Args:
            fault: The fault to inject
        """
        if self.current_fault is not None:
            raise RuntimeError("A fault is already injected. Revert it first.")
        
        try:
            fault.inject(self.model)
            self.current_fault = fault
        except Exception as e:
            self.current_fault = None
            raise e
    
    def revert_fault(self) -> None:
        """Revert the currently injected fault."""
        if self.current_fault is None:
            raise RuntimeError("No fault is currently injected.")
        
        try:
            self.current_fault.revert(self.model)
            self.current_fault = None
        except Exception as e:
            self.current_fault = None
            raise e
    
    def __str__(self) -> str:
        """String representation of the fault injector."""
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__})" 