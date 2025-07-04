"""
Weight corruption fault implementation for injecting faults into model weights.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List
from goldentransformer.faults.base_fault import BaseFault

class WeightCorruption(BaseFault):
    """Fault type for corrupting model weights."""
    
    def __init__(
        self,
        pattern: str = "random",
        corruption_rate: float = 0.1,
        target_layers: Optional[List[int]] = None,
        target_weights: Optional[List[str]] = None
    ):
        """
        Initialize the weight corruption fault.
        
        Args:
            pattern (str): Corruption pattern ("random", "structured", "bit_flip")
            corruption_rate (float): Rate of weight corruption (0.0 to 1.0)
            target_layers (Optional[List[int]]): Specific layers to target (None for all layers)
            target_weights (Optional[List[str]]): Specific weight types to target (None for all weights)
        """
        super().__init__(corruption_rate)
        self.pattern = pattern
        self.target_layers = target_layers
        self.target_weights = target_weights
        
        if pattern not in ["random", "structured", "bit_flip"]:
            raise ValueError(f"Invalid corruption pattern: {pattern}")
    
    def inject(self, model: torch.nn.Module) -> None:
        """
        Inject the weight corruption fault into the model.
        
        Args:
            model: The model to inject the fault into
        """
        # Store original weights
        original_weights = {}
        # Determine model architecture and get layers
        if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
            # BERT model
            layers = model.bert.encoder.layer
            layer_mode = True
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT model
            layers = model.transformer.h
            layer_mode = True
        elif hasattr(model, "transformer") and isinstance(model.transformer, torch.nn.ModuleList):
            # Simple transformer
            layers = model.transformer
            layer_mode = True
        else:
            # Fallback: treat model as a flat module
            layers = [model]
            layer_mode = False
        # Iterate through model layers
        for i, layer in enumerate(layers):
            if self.target_layers is not None and i not in self.target_layers:
                continue
            # Corrupt weights in the layer
            for name, param in layer.named_parameters():
                if self.target_weights is not None and name not in self.target_weights:
                    continue
                if "weight" in name:
                    original_weights[f"{i}.{name}"] = param.data.clone()
                    self._corrupt_weights(param.data)
        if not layer_mode:
            # Also corrupt top-level parameters if not in layer mode
            for name, param in model.named_parameters():
                if self.target_weights is not None and name not in self.target_weights:
                    continue
                if "weight" in name:
                    original_weights[f"top.{name}"] = param.data.clone()
                    self._corrupt_weights(param.data)
        self.save_state({"weights": original_weights})
    
    def revert(self, model: torch.nn.Module) -> None:
        """
        Revert the weight corruption fault from the model.
        
        Args:
            model: The model to revert the fault from
        """
        if self.get_state() is not None:
            original_weights = self.get_state()["weights"]
            # Determine model architecture and get layers
            if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
                layers = model.bert.encoder.layer
                layer_mode = True
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layers = model.transformer.h
                layer_mode = True
            elif hasattr(model, "transformer") and isinstance(model.transformer, torch.nn.ModuleList):
                layers = model.transformer
                layer_mode = True
            else:
                layers = [model]
                layer_mode = False
            # Restore original weights
            for i, layer in enumerate(layers):
                if self.target_layers is not None and i not in self.target_layers:
                    continue
                for name, param in layer.named_parameters():
                    key = f"{i}.{name}"
                    if key in original_weights:
                        param.data = original_weights[key]
            if not layer_mode:
                for name, param in model.named_parameters():
                    key = f"top.{name}"
                    if key in original_weights:
                        param.data = original_weights[key]
    
    def _corrupt_weights(self, weights: torch.Tensor) -> None:
        """
        Corrupt weights based on the specified pattern.
        
        Args:
            weights: Weight tensor to corrupt
        """
        if self.pattern == "random":
            self._random_corruption(weights)
        elif self.pattern == "structured":
            self._structured_corruption(weights)
        elif self.pattern == "bit_flip":
            self._bit_flip_corruption(weights)
    
    def _random_corruption(self, weights: torch.Tensor) -> None:
        """Apply random corruption to weights."""
        mask = torch.rand_like(weights) < self.severity
        noise = torch.randn_like(weights) * self.severity
        weights[mask] += noise[mask]
    
    def _structured_corruption(self, weights: torch.Tensor) -> None:
        """Apply structured corruption to weights (corrupting entire rows/columns)."""
        if len(weights.shape) == 2:
            # For 2D weights, corrupt entire rows or columns
            if np.random.random() < 0.5:
                # Corrupt rows
                num_rows = int(weights.shape[0] * self.severity)
                rows = np.random.choice(weights.shape[0], num_rows, replace=False)
                weights[rows] += torch.randn_like(weights[rows]) * self.severity
            else:
                # Corrupt columns
                num_cols = int(weights.shape[1] * self.severity)
                cols = np.random.choice(weights.shape[1], num_cols, replace=False)
                weights[:, cols] += torch.randn_like(weights[:, cols]) * self.severity
        else:
            # For other shapes, fall back to random corruption
            self._random_corruption(weights)
    
    def _bit_flip_corruption(self, weights: torch.Tensor) -> None:
        """Apply bit-flip corruption to weights."""
        # Get the number of bits to flip based on severity
        total_bits = weights.numel() * 32  # 32 bits per float
        num_bits_to_flip = int(total_bits * self.severity)
        
        if num_bits_to_flip == 0:
            return  # No bits to flip
        
        # Store original weights for comparison
        original_weights = weights.clone()
        
        # Flatten weights for easier manipulation
        flat_weights = weights.flatten()
        
        # Convert to int32 for bit manipulation (avoid uint32 promotion issues)
        weights_int = flat_weights.view(torch.int32)
        
        # For IEEE 754 float32: 1 sign bit + 8 exponent bits + 23 mantissa bits
        # We'll only flip bits in the mantissa (bits 0-22) to avoid infinite values
        mantissa_bits = 23
        total_mantissa_bits = weights.numel() * mantissa_bits
        
        # Calculate how many mantissa bits to flip
        mantissa_bits_to_flip = int(total_mantissa_bits * self.severity)
        
        if mantissa_bits_to_flip == 0:
            return
        
        # Randomly select mantissa bits to flip
        mantissa_bit_indices = torch.randperm(total_mantissa_bits, dtype=torch.long)[:mantissa_bits_to_flip]
        
        for mantissa_bit_idx in mantissa_bit_indices:
            # Calculate which weight and which bit within that weight's mantissa
            weight_idx = mantissa_bit_idx // mantissa_bits
            bit_pos = mantissa_bit_idx % mantissa_bits  # This is 0-22 (mantissa bits)
            
            # Flip the specific mantissa bit
            weights_int[weight_idx] = weights_int[weight_idx] ^ (1 << bit_pos)
        
        # Convert back to float (no need to copy since we modified in place)
        # The view operation automatically handles the conversion 