"""
Layer-wise fault injection for transformer models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from goldentransformer.faults.base_fault import BaseFault

class LayerFault(BaseFault):
    """Fault injection for specific layers in transformer models."""
    
    def __init__(
        self,
        layer_idx: Union[int, List[int]],
        fault_type: str = "attention_mask",
        severity: float = 0.1,
        target_components: Optional[List[str]] = None
    ):
        """
        Initialize the layer fault.
        
        Args:
            layer_idx: Index or list of indices of layers to inject faults into
            fault_type: Type of fault to inject ('attention_mask', 'dropout', 'activation')
            severity: Severity of the fault (0.0 to 1.0)
            target_components: List of components to target within the layer
                             (e.g., ['self_attention', 'ffn'])
        """
        super().__init__(severity)
        self.layer_idx = layer_idx if isinstance(layer_idx, list) else [layer_idx]
        self.fault_type = fault_type
        self.target_components = target_components or ['self_attention', 'ffn']
        self._original_states: Dict[int, Dict[str, Any]] = {}
    
    def inject(self, model: nn.Module) -> None:
        """
        Inject the fault into specified layers of the model.
        
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
            if 'self_attention' in self.target_components:
                self._inject_attention_fault(layer)
            if 'ffn' in self.target_components:
                self._inject_ffn_fault(layer)
    
    def revert(self, model: nn.Module) -> None:
        """
        Revert the fault from the model.
        
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
            if 'self_attention' in self.target_components:
                self._revert_attention_fault(layer, state)
            if 'ffn' in self.target_components:
                self._revert_ffn_fault(layer, state)
        self._original_states.clear()
    
    def _inject_attention_fault(self, layer: nn.Module) -> None:
        """Inject fault into attention mechanism."""
        if self.fault_type == 'attention_mask':
            # Only operate if layer has 'attn' and 'attn_mask'
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'attn_mask'):
                if layer not in self._original_states:
                    self._original_states[layer] = {}
                self._original_states[layer].update({
                    'attn_mask': layer.attn.attn_mask.clone() if layer.attn.attn_mask is not None else None
                })
                # Modify attention mask
                if layer.attn.attn_mask is not None:
                    mask = layer.attn.attn_mask
                    num_zeros = int(mask.numel() * self.severity)
                    zero_indices = torch.randperm(mask.numel())[:num_zeros]
                    mask.view(-1)[zero_indices] = 0
                    layer.attn.attn_mask = mask
    
    def _inject_ffn_fault(self, layer: nn.Module) -> None:
        """Inject fault into feed-forward network."""
        if self.fault_type == 'dropout':
            # Store original dropout rates
            self._original_states[layer].update({
                'dropout': layer.mlp.dropout.p if hasattr(layer.mlp, 'dropout') else None
            })
            
            # Increase dropout rate
            if hasattr(layer.mlp, 'dropout'):
                layer.mlp.dropout.p = min(1.0, layer.mlp.dropout.p + self.severity)
    
    def _revert_attention_fault(self, layer: nn.Module, state: Dict[str, Any]) -> None:
        """Revert attention fault."""
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'attn_mask'):
            if 'attn_mask' in state and state['attn_mask'] is not None:
                layer.attn.attn_mask = state['attn_mask']
    
    def _revert_ffn_fault(self, layer: nn.Module, state: Dict[str, Any]) -> None:
        """Revert FFN fault."""
        if 'dropout' in state and state['dropout'] is not None:
            layer.mlp.dropout.p = state['dropout']
    
    def __str__(self) -> str:
        """String representation of the fault."""
        return (f"{self.__class__.__name__}(layers={self.layer_idx}, "
                f"type={self.fault_type}, severity={self.severity})") 