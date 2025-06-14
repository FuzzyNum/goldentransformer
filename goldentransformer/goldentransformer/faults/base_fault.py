"""
Base fault class for the GoldenTransformer framework.
All fault types should inherit from this class.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Any, Dict, Optional

class BaseFault(ABC):
    """Base class for all fault types in the framework."""
    
    def __init__(self, severity: float = 0.1):
        """
        Initialize the base fault.
        
        Args:
            severity (float): Severity of the fault (0.0 to 1.0)
        """
        self.severity = max(0.0, min(1.0, severity))
        self._original_state: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def inject(self, model: torch.nn.Module) -> None:
        """
        Inject the fault into the model.
        
        Args:
            model (torch.nn.Module): The model to inject the fault into
        """
        pass
    
    @abstractmethod
    def revert(self, model: torch.nn.Module) -> None:
        """
        Revert the fault from the model.
        
        Args:
            model (torch.nn.Module): The model to revert the fault from
        """
        pass
    
    def save_state(self, state: Dict[str, Any]) -> None:
        """
        Save the original state of the model before fault injection.
        
        Args:
            state (Dict[str, Any]): The state to save
        """
        self._original_state = state
    
    def get_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the saved original state.
        
        Returns:
            Optional[Dict[str, Any]]: The saved state, or None if no state was saved
        """
        return self._original_state
    
    def __str__(self) -> str:
        """String representation of the fault."""
        return f"{self.__class__.__name__}(severity={self.severity})" 