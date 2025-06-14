"""
Activation function fault implementation for injecting faults into model activations.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
from goldentransformer.faults.base_fault import BaseFault

class ActivationFault(BaseFault):
    """Fault type for injecting faults into activation functions."""
    
    def __init__(
        self,
        layer_idx: Union[int, List[int]],
        fault_type: str = "clamp",
        severity: float = 0.1,
        target_activations: Optional[List[str]] = None
    ):
        """
        Initialize the activation fault.
        
        Args:
            layer_idx: Index or list of indices of layers to inject faults into
            fault_type: Type of fault to inject ('clamp', 'noise', 'zero_out')
            severity: Severity of the fault (0.0 to 1.0)
            target_activations: List of activation types to target
                              (e.g., ['gelu', 'relu', 'silu'])
        """
        super().__init__(severity)
        self.layer_idx = layer_idx if isinstance(layer_idx, list) else [layer_idx]
        self.fault_type = fault_type
        self.target_activations = target_activations or ['gelu', 'relu', 'silu']
        self._original_states: Dict[int, Dict[str, Any]] = {}
        
        if fault_type not in ["clamp", "noise", "zero_out"]:
            raise ValueError(f"Invalid fault type: {fault_type}")
    
    def inject(self, model: nn.Module) -> None:
        """
        Inject the activation fault into the model.
        
        Args:
            model: The model to inject the fault into
        """
        # Support both HuggingFace and simple ModuleList
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'transformer') and isinstance(model.transformer, nn.ModuleList):
            layers = model.transformer
        else:
            raise ValueError("Model must have transformer.h or transformer as ModuleList")
            
        for idx in self.layer_idx:
            layer = layers[idx]
            self._original_states[idx] = {}
            
            # Find and modify activation functions
            for name, module in layer.named_modules():
                if isinstance(module, (nn.GELU, nn.ReLU, nn.SiLU)):
                    activation_type = module.__class__.__name__.lower()
                    if activation_type in self.target_activations:
                        self._inject_activation_fault(module, idx, name)
    
    def revert(self, model: nn.Module) -> None:
        """
        Revert the activation fault from the model.
        
        Args:
            model: The model to revert the fault from
        """
        # Support both HuggingFace and simple ModuleList
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'transformer') and isinstance(model.transformer, nn.ModuleList):
            layers = model.transformer
        else:
            raise ValueError("Model must have transformer.h or transformer as ModuleList")
            
        for idx, state in self._original_states.items():
            layer = layers[idx]
            for name, module in layer.named_modules():
                if isinstance(module, (nn.GELU, nn.ReLU, nn.SiLU)):
                    activation_type = module.__class__.__name__.lower()
                    if activation_type in self.target_activations:
                        self._revert_activation_fault(module, state, name)
        self._original_states.clear()
    
    def _inject_activation_fault(self, module: nn.Module, layer_idx: int, name: str) -> None:
        """Inject fault into activation function."""
        if self.fault_type == "clamp":
            # Store original forward method
            self._original_states[layer_idx][name] = {
                'forward': module.forward
            }
            # Replace with clamped version
            module.forward = lambda x: torch.clamp(
                module.forward(x),
                min=-self.severity,
                max=self.severity
            )
            
        elif self.fault_type == "noise":
            # Store original forward method
            self._original_states[layer_idx][name] = {
                'forward': module.forward
            }
            # Replace with noisy version
            original_forward = module.forward
            module.forward = lambda x: original_forward(x) + torch.randn_like(x) * self.severity
            
        elif self.fault_type == "zero_out":
            # Store original forward method
            self._original_states[layer_idx][name] = {
                'forward': module.forward
            }
            # Replace with zero-out version
            original_forward = module.forward
            module.forward = lambda x: original_forward(x) * (torch.rand_like(x) > self.severity).float()
    
    def _revert_activation_fault(self, module: nn.Module, state: Dict[str, Any], name: str) -> None:
        """Revert activation fault."""
        if name in state:
            module.forward = state[name]['forward']
    
    def __str__(self) -> str:
        """String representation of the fault."""
        return (f"{self.__class__.__name__}(layers={self.layer_idx}, "
                f"type={self.fault_type}, severity={self.severity})") 