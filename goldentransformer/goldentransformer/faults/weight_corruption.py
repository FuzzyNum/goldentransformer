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
        """Apply bit-flip corruption to weights, guaranteed to flip bits in the binary representation."""
        device = weights.device
        dtype = weights.dtype
        weights_cpu = weights.detach().cpu().contiguous()
        arr = weights_cpu.numpy().view(np.int32)
        shape = weights_cpu.shape
        mantissa_bits = 23

        # Save original for diagnostics
        original = weights_cpu.clone()

        # Create a mask for which weights should be corrupted (probability per weight)
        corruption_mask = np.random.rand(*shape) < self.severity
        corrupted_indices = np.where(corruption_mask)
        num_selected = len(corrupted_indices[0])
        num_bits_flipped = 0

        for idx in zip(*corrupted_indices):
            flat_idx = np.ravel_multi_index(idx, shape)
            bit_pos = np.random.randint(0, mantissa_bits)
            before = arr.flat[flat_idx]
            arr.flat[flat_idx] ^= (1 << bit_pos)
            after = arr.flat[flat_idx]
            if before != after:
                num_bits_flipped += 1

        # Copy the modified values back to the original tensor
        weights.data.copy_(torch.from_numpy(arr.view(np.float32)).to(device).type(dtype))

        # Diagnostics
        mean_abs_diff = (weights_cpu - original).abs().mean().item()
        print(f"[BitFlip] Selected weights: {num_selected}, Bits flipped: {num_bits_flipped}, Mean abs diff: {mean_abs_diff}") 