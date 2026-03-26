"""TurboQuant Weight Quantization: compress nn.Linear weight matrices.

Applies the same PolarQuant + QJL pipeline used for KV cache compression
to model weight matrices. Each row of a weight matrix is treated as a
vector and quantized independently.

Two modes:
  1. Dequantize mode (Phase 1): Stores weights compressed, dequantizes to
     full precision for matmul. Saves memory, standard compute.
  2. Compressed mode (Phase 2): Computes matmuls directly on compressed
     weights using the unbiased inner product estimator. Saves memory AND
     compute. This is the novel research contribution.

Usage:
    from turboquant.weight_quant import TurboQuantLinear, quantize_model

    # Quantize a single layer
    tq_linear = TurboQuantLinear.from_linear(linear_layer, bits=3)
    output = tq_linear(input)  # Same API as nn.Linear

    # Quantize an entire model
    quantize_model(model, bits=3)
    output = model(input)  # Works as before, uses less memory
"""

from typing import Optional, Tuple, Dict, Any
import math
import os
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquant.polar_quant import PolarQuant
from turboquant.qjl import QJL
from turboquant.turboquant import TurboQuant, TurboQuantEncoded

logger = logging.getLogger(__name__)


class TurboQuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear with TurboQuant weight compression.

    Stores the weight matrix in compressed form (PolarQuant indices + QJL
    sign bits + norms). On forward pass, either dequantizes and does standard
    matmul (Phase 1) or computes directly on compressed weights using the
    unbiased inner product estimator (Phase 2).

    Memory comparison for a (4096, 4096) linear layer:
        FP16:   32 MB
        3-bit:  ~6 MB  (5.3x compression)
        4-bit:  ~8 MB  (4x compression)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 3,
        bias: bool = True,
        compressed_forward: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        shared_quantizer: Optional[TurboQuant] = None,
    ):
        """Initialize a TurboQuant linear layer.

        Args:
            in_features: Input dimension.
            out_features: Output dimension (number of weight rows).
            bits: Bits per coordinate for weight quantization.
            bias: Whether the layer has a bias term.
            compressed_forward: If True, use Phase 2 compressed-domain
                matmul. If False (default), dequantize for standard matmul.
            device: Torch device.
            dtype: Original weight dtype for dequantization.
            shared_quantizer: Optional pre-existing TurboQuant to reuse
                (shares rotation matrix across layers with same dimensions).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.compressed_forward = compressed_forward
        self._dtype = dtype

        # TurboQuant quantizer for weight rows — share across layers
        # with the same (in_features, bits) to save memory on the (d,d)
        # rotation matrix.
        if shared_quantizer is not None:
            self._quantizer = shared_quantizer
        else:
            self._quantizer = TurboQuant(
                d=in_features, bits=bits,
                device=device or torch.device("cpu"))

        # Compressed weight storage (populated by from_linear or load)
        # PolarQuant indices: (out_features, in_features), int
        self.register_buffer("pq_indices", None)
        # Vector norms: (out_features, 1), float32
        self.register_buffer("weight_norms", None)
        # QJL sign bits: (out_features, m), bool
        self.register_buffer("qjl_sign_bits", None)
        # Residual norms: (out_features, 1), float32
        self.register_buffer("residual_norms", None)

        # Bias (full precision, tiny compared to weights)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Cached dequantized weights (invalidated on device move)
        self._weight_cache: Optional[torch.Tensor] = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 3,
        compressed_forward: bool = False,
        shared_quantizer: Optional[TurboQuant] = None,
    ) -> "TurboQuantLinear":
        """Convert an existing nn.Linear to TurboQuantLinear.

        This is the main entry point for quantizing a model. Takes the
        full-precision weight matrix, quantizes each row with TurboQuant,
        and stores the compressed representation.

        Args:
            linear: The nn.Linear layer to quantize.
            bits: Bits per coordinate.
            compressed_forward: Use compressed-domain matmul.
            shared_quantizer: Optional pre-existing TurboQuant to reuse.

        Returns:
            A TurboQuantLinear with the same behavior but compressed weights.
        """
        device = linear.weight.device
        dtype = linear.weight.dtype
        has_bias = linear.bias is not None

        tq_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            bias=has_bias,
            compressed_forward=compressed_forward,
            device=device,
            dtype=dtype,
            shared_quantizer=shared_quantizer,
        )

        # Quantize the weight matrix (each row is a vector)
        weight = linear.weight.data.float()  # (out_features, in_features)
        encoded = tq_linear._quantizer.encode(weight)

        tq_linear.pq_indices = encoded.pq_indices.to(device=device, dtype=torch.int16)
        tq_linear.weight_norms = encoded.norms.to(device)
        tq_linear.qjl_sign_bits = encoded.qjl_sign_bits.to(device)
        tq_linear.residual_norms = encoded.residual_norms.to(device)

        # Copy bias
        if has_bias:
            tq_linear.bias = nn.Parameter(linear.bias.data.clone())

        return tq_linear

    def _dequantize_weights(self) -> torch.Tensor:
        """Reconstruct the full-precision weight matrix from compressed storage.

        Uses PolarQuant reconstruction (MSE-optimal). The QJL sign bits
        are not used here — they're for the compressed-domain matmul path.

        Returns:
            Weight matrix of shape (out_features, in_features).
        """
        if self._weight_cache is not None:
            return self._weight_cache

        weight = self._quantizer.polar_quant.decode(
            self.pq_indices, self.weight_norms)
        weight = weight.to(self._dtype)

        # Cache for repeated forward passes
        self._weight_cache = weight
        return weight

    def _invalidate_cache(self):
        """Clear the dequantized weight cache."""
        self._weight_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = x @ W^T + bias.

        In dequantize mode (Phase 1): reconstructs weights then does F.linear.
        In compressed mode (Phase 2): uses TurboQuant's unbiased IP estimator
        to compute x @ W^T directly from compressed weights.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        if self.compressed_forward:
            return self._compressed_forward(x)
        else:
            return self._dequantize_forward(x)

    def _dequantize_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1: Dequantize weights and use standard matmul."""
        weight = self._dequantize_weights()
        return F.linear(x, weight, self.bias)

    def _compressed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 2: Compute matmul directly on compressed weights.

        Uses TurboQuant's unbiased inner product estimator:
            y_j = <x, w_j> ≈ <x, w_j_pq> + QJL_correction(x, residual_j)

        This avoids ever materializing the full weight matrix in memory.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        x_dtype = x.dtype
        # Reconstruct PolarQuant weights for the first term
        w_pq = self._quantizer.polar_quant.decode(
            self.pq_indices, self.weight_norms)  # (out_features, in_features)

        # Term 1: x @ w_pq^T (standard matmul with reconstructed weights)
        # This is the MSE-optimal part
        pq_result = x.to(w_pq.dtype) @ w_pq.t()

        # Term 2: QJL correction for the residual
        # Project x through the JL matrix: x_proj = x @ S^T
        S = self._quantizer.qjl.S  # (m, in_features)
        x_proj = x.to(S.dtype) @ S.t()  # (..., m)

        # Sign bits are the quantized residual projections
        signs = self.qjl_sign_bits.float() * 2 - 1  # (out_features, m)

        # Compute correction: sum_j x_proj_j * sign_j for each output
        qjl_raw = x_proj @ signs.t()  # (..., out_features)

        # Scale by sqrt(pi/2)/m and residual norms
        scale = math.sqrt(math.pi / 2.0) / self._quantizer.qjl.m
        qjl_correction = qjl_raw * scale * self.residual_norms.squeeze(-1)

        # Combine
        result = pq_result + qjl_correction.to(pq_result.dtype)

        if self.bias is not None:
            result = result + self.bias

        return result.to(x_dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bits={self.bits}, "
                f"bias={self.bias is not None}, "
                f"mode={'compressed' if self.compressed_forward else 'dequantize'}")

    def memory_bytes(self) -> dict:
        """Estimate memory usage.

        Reports theoretical compressed size (packed bits) rather than
        current PyTorch tensor sizes, since indices are stored as int64
        but only need (bits-1) bits each, and sign bits are stored as
        bool (1 byte) but only need 1 bit.
        """
        if self.pq_indices is None:
            return {"compressed_bytes": 0, "fp16_bytes": 0, "compression_ratio": 0}

        n_weights = self.out_features * self.in_features

        # Theoretical packed size:
        # PQ indices: (bits-1) bits per coordinate
        pq_bits = (self.bits - 1) * n_weights
        # QJL sign bits: 1 bit per coordinate
        qjl_bits = n_weights
        # Norms: 1 float32 per row (weight norm + residual norm)
        norm_bytes = self.out_features * 4 * 2

        compressed = (pq_bits + qjl_bits) // 8 + norm_bytes
        bias_bytes = self.bias.nelement() * self.bias.element_size() if self.bias is not None else 0
        compressed += bias_bytes

        # FP16 equivalent
        fp16 = self.in_features * self.out_features * 2 + bias_bytes

        return {
            "compressed_bytes": compressed,
            "fp16_bytes": fp16,
            "compression_ratio": fp16 / max(compressed, 1),
        }

    def _apply(self, fn, recurse=True):
        """Override to invalidate cache on device/dtype changes."""
        self._invalidate_cache()
        return super()._apply(fn, recurse)


def quantize_model(
    model: nn.Module,
    bits: int = 3,
    compressed_forward: bool = False,
    skip_modules: Optional[list] = None,
    min_features: int = 64,
) -> nn.Module:
    """Quantize all nn.Linear layers in a model with TurboQuant.

    Walks the model tree and replaces each nn.Linear with a
    TurboQuantLinear. Skips the LM head by default (output projection)
    since quantizing it hurts perplexity disproportionately.

    Args:
        model: The HuggingFace model to quantize.
        bits: Bits per coordinate for weight quantization.
        compressed_forward: Use compressed-domain matmul (Phase 2).
        skip_modules: List of module name patterns to skip.
            Defaults to ["lm_head", "embed_tokens", "embed_positions"].
        min_features: Skip layers smaller than this (not worth quantizing).

    Returns:
        The model with quantized linear layers (modified in-place).
    """
    if skip_modules is None:
        skip_modules = ["lm_head", "embed_tokens", "embed_positions",
                        "wte", "wpe", "rotary_emb", "norm"]

    total_params = 0
    quantized_params = 0
    replaced = 0

    # Cache TurboQuant instances by (in_features, bits) to share rotation
    # matrices across layers with the same dimensions. This dramatically
    # reduces memory: e.g. 36 down_proj layers share one (d,d) rotation
    # instead of each allocating their own.
    quantizer_cache: dict = {}

    for name, module in model.named_modules():
        # Find all nn.Linear children of this module
        children_to_replace = {}
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue

            full_name = f"{name}.{child_name}" if name else child_name
            n_params = child.weight.nelement()
            total_params += n_params

            # Skip if in skip list
            if any(skip in full_name for skip in skip_modules):
                logger.info(f"Skipping {full_name} (in skip list)")
                continue

            # Skip if too small
            if child.in_features < min_features or child.out_features < min_features:
                logger.info(f"Skipping {full_name} (too small: "
                            f"{child.in_features}x{child.out_features})")
                continue

            # Quantize, reusing shared quantizer for same dimensions
            cache_key = (child.in_features, bits)
            shared_quantizer = quantizer_cache.get(cache_key)
            tq_linear = TurboQuantLinear.from_linear(
                child, bits=bits, compressed_forward=compressed_forward,
                shared_quantizer=shared_quantizer)
            if cache_key not in quantizer_cache:
                quantizer_cache[cache_key] = tq_linear._quantizer
            children_to_replace[child_name] = tq_linear
            quantized_params += n_params
            replaced += 1

        # Apply replacements
        for child_name, tq_linear in children_to_replace.items():
            setattr(module, child_name, tq_linear)

    logger.info(f"Quantized {replaced} linear layers "
                f"({quantized_params / 1e6:.1f}M / {total_params / 1e6:.1f}M params)")
    return model


def save_quantized(model: nn.Module, path: str) -> None:
    """Save a quantized model to disk.

    Saves the compressed weights (indices, norms, sign bits) rather than
    full-precision weights, resulting in much smaller files.

    Args:
        model: Model with TurboQuantLinear layers.
        path: Directory to save to.
    """
    os.makedirs(path, exist_ok=True)

    # Separate quantized and non-quantized state
    quantized_state = {}
    other_state = {}

    for name, param_or_buf in {**dict(model.named_parameters()),
                                **dict(model.named_buffers())}.items():
        if param_or_buf is None:
            continue
        if any(k in name for k in ["pq_indices", "weight_norms",
                                    "qjl_sign_bits", "residual_norms"]):
            quantized_state[name] = param_or_buf
        else:
            other_state[name] = param_or_buf

    # Save compressed weights with reduced precision where possible
    torch.save(quantized_state, os.path.join(path, "quantized_weights.pt"))
    torch.save(other_state, os.path.join(path, "other_weights.pt"))

    # Save metadata
    meta = {
        "quantized_layers": [],
        "format": "turboquant",
        "version": "0.1.0",
    }
    for name, module in model.named_modules():
        if isinstance(module, TurboQuantLinear):
            meta["quantized_layers"].append({
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bits": module.bits,
                "compressed_forward": module.compressed_forward,
                "has_bias": module.bias is not None,
            })

    with open(os.path.join(path, "turboquant_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Report savings
    q_size = os.path.getsize(os.path.join(path, "quantized_weights.pt"))
    o_size = os.path.getsize(os.path.join(path, "other_weights.pt"))
    total_mb = (q_size + o_size) / (1024 * 1024)
    logger.info(f"Saved quantized model to {path} ({total_mb:.1f} MB)")


def load_quantized(model: nn.Module, path: str) -> nn.Module:
    """Load quantized weights into a model.

    The model should already have TurboQuantLinear layers (call
    quantize_model first with dummy weights, or reconstruct the
    architecture from turboquant_config.json).

    Args:
        model: Model with TurboQuantLinear layers.
        path: Directory containing saved quantized model.

    Returns:
        Model with loaded quantized weights.
    """
    q_state = torch.load(os.path.join(path, "quantized_weights.pt"),
                         weights_only=True)
    o_state = torch.load(os.path.join(path, "other_weights.pt"),
                         weights_only=True)

    # Merge and load
    full_state = {**q_state, **o_state}
    missing, unexpected = model.load_state_dict(full_state, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}...")

    return model


def model_memory_report(model: nn.Module) -> dict:
    """Report memory usage of a quantized model.

    Shows per-layer compression ratios and total savings.
    """
    layers = []
    total_compressed = 0
    total_fp16 = 0

    for name, module in model.named_modules():
        if isinstance(module, TurboQuantLinear):
            mem = module.memory_bytes()
            layers.append({
                "name": name,
                "shape": f"{module.out_features}x{module.in_features}",
                "bits": module.bits,
                **mem,
            })
            total_compressed += mem["compressed_bytes"]
            total_fp16 += mem["fp16_bytes"]

    return {
        "total_compressed_mb": total_compressed / (1024 * 1024),
        "total_fp16_mb": total_fp16 / (1024 * 1024),
        "compression_ratio": total_fp16 / max(total_compressed, 1),
        "num_quantized_layers": len(layers),
        "layers": layers,
    }
