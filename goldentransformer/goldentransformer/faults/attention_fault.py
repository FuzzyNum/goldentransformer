"""
Attention fault implementation for injecting faults into attention mechanisms.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from goldentransformer.faults.base_fault import BaseFault

class AttentionFault(BaseFault):
    """Fault type for injecting faults into attention mechanisms."""
    
    def __init__(
        self,
        layer_idx: int,
        fault_type: str,
        severity: float = 0.1,
        head_idx: Optional[int] = None
    ):
        """
        Initialize the attention fault.
        
        Args:
            layer_idx (int): Index of the layer to inject the fault into
            fault_type (str): Type of attention fault to inject
            severity (float): Severity of the fault (0.0 to 1.0)
            head_idx (Optional[int]): Index of the attention head to target (None for all heads)
        """
        super().__init__(severity)
        self.layer_idx = layer_idx
        self.fault_type = fault_type
        self.head_idx = head_idx
        
        if fault_type not in [
            "mask_corruption",
            "head_dropout",
            "query_corruption",
            "key_corruption",
            "value_corruption"
        ]:
            raise ValueError(f"Invalid fault type: {fault_type}")
    
    def inject(self, model: torch.nn.Module) -> None:
        """
        Inject the attention fault into the model.
        
        Args:
            model (torch.nn.Module): The model to inject the fault into
        """
        # Get the target layer
        layer = model.transformer.h[self.layer_idx]
        
        if self.fault_type == "mask_corruption":
            self._corrupt_attention_mask(layer)
        elif self.fault_type == "head_dropout":
            self._drop_attention_head(layer)
        elif self.fault_type == "query_corruption":
            self._corrupt_query_vectors(layer)
        elif self.fault_type == "key_corruption":
            self._corrupt_key_vectors(layer)
        elif self.fault_type == "value_corruption":
            self._corrupt_value_vectors(layer)
    
    def revert(self, model: torch.nn.Module) -> None:
        """
        Revert the attention fault from the model.
        
        Args:
            model (torch.nn.Module): The model to revert the fault from
        """
        if self.get_state() is not None:
            # Restore the original attention weights
            layer = model.transformer.h[self.layer_idx]
            original_state = self.get_state()
            
            if self.fault_type == "mask_corruption":
                layer.attn.mask = original_state["mask"]
            elif self.fault_type == "head_dropout":
                layer.attn.head_mask = original_state["head_mask"]
            elif self.fault_type == "query_corruption":
                layer.attn.q_proj.weight = original_state["q_proj_weight"]
            elif self.fault_type == "key_corruption":
                layer.attn.k_proj.weight = original_state["k_proj_weight"]
            elif self.fault_type == "value_corruption":
                layer.attn.v_proj.weight = original_state["v_proj_weight"]
    
    def _corrupt_attention_mask(self, layer: torch.nn.Module) -> None:
        """Corrupt the attention mask with random noise."""
        if hasattr(layer.attn, "mask"):
            original_mask = layer.attn.mask.clone()
            noise = torch.randn_like(original_mask) * self.severity
            layer.attn.mask = original_mask + noise
            self.save_state({"mask": original_mask})
    
    def _drop_attention_head(self, layer: torch.nn.Module) -> None:
        """Drop an attention head by zeroing its weights."""
        if hasattr(layer.attn, "head_mask"):
            original_mask = layer.attn.head_mask.clone()
            if self.head_idx is not None:
                layer.attn.head_mask[self.head_idx] = 0
            else:
                # Randomly drop heads based on severity
                num_heads = layer.attn.num_attention_heads
                num_drop = int(num_heads * self.severity)
                drop_indices = np.random.choice(num_heads, num_drop, replace=False)
                layer.attn.head_mask[drop_indices] = 0
            self.save_state({"head_mask": original_mask})
    
    def _corrupt_query_vectors(self, layer: torch.nn.Module) -> None:
        """Corrupt query projection weights with random noise."""
        original_weight = layer.attn.q_proj.weight.clone()
        noise = torch.randn_like(original_weight) * self.severity
        layer.attn.q_proj.weight = original_weight + noise
        self.save_state({"q_proj_weight": original_weight})
    
    def _corrupt_key_vectors(self, layer: torch.nn.Module) -> None:
        """Corrupt key projection weights with random noise."""
        original_weight = layer.attn.k_proj.weight.clone()
        noise = torch.randn_like(original_weight) * self.severity
        layer.attn.k_proj.weight = original_weight + noise
        self.save_state({"k_proj_weight": original_weight})
    
    def _corrupt_value_vectors(self, layer: torch.nn.Module) -> None:
        """Corrupt value projection weights with random noise."""
        original_weight = layer.attn.v_proj.weight.clone()
        noise = torch.randn_like(original_weight) * self.severity
        layer.attn.v_proj.weight = original_weight + noise
        self.save_state({"v_proj_weight": original_weight}) 