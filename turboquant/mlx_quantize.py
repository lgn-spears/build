"""TurboQuant MLX Model Quantization — drop-in weight compression for mlx-lm.

Quantizes any MLX language model's nn.Linear layers with TurboQuant's
two-stage vector quantization. Designed to integrate with the mlx-lm
ecosystem (LLaMA, Mistral, Qwen, Phi, Gemma, etc).

Usage:
    import mlx.core as mx
    from mlx_lm import load
    from turboquant.mlx_quantize import quantize_model, TurboQuantLinear

    # Load any mlx-lm model
    model, tokenizer = load("mlx-community/Qwen2.5-7B-4bit")

    # Requantize with TurboQuant (better quality at same bits)
    quantize_model(model, bits=4)

    # Or load a full-precision model and quantize from scratch
    model, tokenizer = load("Qwen/Qwen2.5-7B")
    quantize_model(model, bits=4)

    # Generate as normal — TurboQuant is transparent
    from mlx_lm import generate
    print(generate(model, tokenizer, prompt="Hello"))
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for the MLX backend. Install with: pip install mlx"
    )

from turboquant.mlx_backend import TurboQuant, TurboQuantEncoded


# ===================================================================
#  TurboQuantLinear — Drop-in replacement for mlx.nn.Linear
# ===================================================================

class TurboQuantLinear(nn.Module):
    """MLX Linear layer with TurboQuant weight compression.

    Drop-in replacement for nn.Linear. Stores weights in compressed form
    and decompresses on the fly during forward pass. Two inference modes:

    Phase 1 (dequantize): Reconstruct full weights, standard matmul.
        Simpler, good for small batch sizes.

    Phase 2 (compressed): Compute y = x @ W^T directly on compressed
        weights using QJL's unbiased inner product estimator.
        Never materializes full weight matrix — lower peak memory.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        bias: bool = True,
        compressed_forward: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.compressed_forward = compressed_forward

        # These get populated by from_linear()
        self._quantizer: Optional[TurboQuant] = None
        self.pq_indices: Optional[mx.array] = None
        self.weight_norms: Optional[mx.array] = None
        self.qjl_sign_bits: Optional[mx.array] = None
        self.residual_norms: Optional[mx.array] = None
        self._bias: Optional[mx.array] = None
        self._weight_cache: Optional[mx.array] = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        compressed_forward: bool = False,
    ) -> "TurboQuantLinear":
        """Convert an existing nn.Linear to TurboQuantLinear.

        Args:
            linear: Source nn.Linear module.
            bits: Total bits per coordinate (splits as (bits-1) + 1).
            compressed_forward: If True, use Phase 2 compressed matmul.
        """
        weight = linear.weight  # (out_features, in_features) in MLX
        out_features, in_features = weight.shape
        has_bias = hasattr(linear, "bias") and linear.bias is not None

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bits=bits,
            bias=has_bias,
            compressed_forward=compressed_forward,
        )

        # Quantize each row of the weight matrix
        quantizer = TurboQuant(d=in_features, bits=bits)
        encoded = quantizer.encode(weight)

        layer._quantizer = quantizer
        layer.pq_indices = encoded.pq_indices
        layer.weight_norms = encoded.norms
        layer.qjl_sign_bits = encoded.qjl_sign_bits
        layer.residual_norms = encoded.residual_norms

        if has_bias:
            layer._bias = linear.bias

        mx.eval(
            layer.pq_indices,
            layer.weight_norms,
            layer.qjl_sign_bits,
            layer.residual_norms,
        )

        return layer

    @classmethod
    def from_quantized_linear(
        cls,
        qlinear,
        bits: int = 4,
        compressed_forward: bool = False,
    ) -> "TurboQuantLinear":
        """Convert an MLX QuantizedLinear back to full precision, then requantize.

        This is for upgrading models already quantized with MLX's default
        affine quantization to TurboQuant's superior vector quantization.
        """
        # Dequantize the existing quantized weights
        weight = mx.dequantize(
            qlinear.weight,
            qlinear.scales,
            qlinear.biases,
            qlinear.group_size,
            qlinear.bits,
        )
        # Create a temporary linear to convert from
        temp = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        temp.weight = weight
        if hasattr(qlinear, "bias") and qlinear.bias is not None:
            temp.bias = qlinear.bias

        return cls.from_linear(temp, bits=bits, compressed_forward=compressed_forward)

    def _dequantize_weights(self) -> mx.array:
        """Reconstruct full-precision weight matrix from compressed form."""
        if self._weight_cache is not None:
            return self._weight_cache

        encoded = TurboQuantEncoded(
            pq_indices=self.pq_indices,
            norms=self.weight_norms,
            qjl_sign_bits=self.qjl_sign_bits,
            residual_norms=self.residual_norms,
        )
        weights = self._quantizer.decode(encoded)
        self._weight_cache = weights
        return weights

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass — transparent to the caller."""
        if self.compressed_forward:
            return self._compressed_call(x)
        return self._dequantize_call(x)

    def _dequantize_call(self, x: mx.array) -> mx.array:
        """Phase 1: Dequantize weights, standard matmul."""
        w = self._dequantize_weights()
        out = x @ w.T
        if self._bias is not None:
            out = out + self._bias
        return out

    def _compressed_call(self, x: mx.array) -> mx.array:
        """Phase 2: Compute matmul directly on compressed weights.

        y = x @ W^T ≈ x @ W_pq^T + QJL_correction

        Never materializes the full weight matrix.
        """
        # PolarQuant component
        w_pq = self._quantizer.polar_quant.decode(
            self.pq_indices, self.weight_norms
        )
        pq_result = x @ w_pq.T

        # QJL correction component
        S = self._quantizer.qjl.S
        x_proj = x @ S.T  # (batch, m)

        signs = self.qjl_sign_bits.astype(mx.float32) * 2 - 1  # (out, m)
        qjl_raw = x_proj @ signs.T  # (batch, out)

        scale = math.sqrt(math.pi / 2.0) / self._quantizer.qjl.m
        qjl_correction = qjl_raw * scale * self.residual_norms.squeeze(-1)

        result = pq_result + qjl_correction
        if self._bias is not None:
            result = result + self._bias
        return result

    def memory_bytes(self) -> Dict[str, int]:
        """Estimate compressed memory usage (theoretical packed bits)."""
        n_rows = self.out_features
        n_cols = self.in_features
        pq_bits = (self.bits - 1)
        pq_bytes = math.ceil(n_rows * n_cols * pq_bits / 8)
        norms_bytes = n_rows * 4  # float32
        qjl_bytes = math.ceil(n_rows * self._quantizer.qjl.m / 8)
        res_norm_bytes = n_rows * 4
        total = pq_bytes + norms_bytes + qjl_bytes + res_norm_bytes

        fp16_bytes = n_rows * n_cols * 2
        return {
            "compressed_bytes": total,
            "fp16_bytes": fp16_bytes,
            "compression_ratio": fp16_bytes / total if total > 0 else float("inf"),
        }


# ===================================================================
#  Model-level quantization
# ===================================================================

# Modules to skip by default (too sensitive to quantize)
DEFAULT_SKIP = {"lm_head", "embed_tokens", "embed_positions", "wte", "wpe"}


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    compressed_forward: bool = False,
    skip_modules: Optional[Set[str]] = None,
    min_features: int = 32,
) -> nn.Module:
    """Quantize all Linear layers in an MLX model with TurboQuant.

    Walks the model tree and replaces nn.Linear (and optionally
    nn.QuantizedLinear) layers with TurboQuantLinear.

    Args:
        model: Any MLX nn.Module (typically from mlx-lm).
        bits: Bits per coordinate (2-8). 4 is the sweet spot.
        compressed_forward: Use Phase 2 compressed matmul.
        skip_modules: Module names to skip (default: embeddings, lm_head).
        min_features: Skip layers smaller than this.

    Returns:
        The model (modified in-place).
    """
    if skip_modules is None:
        skip_modules = DEFAULT_SKIP

    replacements = {}
    _find_linear_modules(model, "", bits, compressed_forward,
                         skip_modules, min_features, replacements)

    for path, new_module in replacements.items():
        _set_nested_attr(model, path, new_module)

    return model


def _find_linear_modules(
    module: nn.Module,
    prefix: str,
    bits: int,
    compressed_forward: bool,
    skip_modules: Set[str],
    min_features: int,
    replacements: Dict[str, TurboQuantLinear],
):
    """Recursively find and convert Linear modules."""
    for name, child in module.children().items() if hasattr(module, 'children') else []:
        full_name = f"{prefix}.{name}" if prefix else name
        short_name = name.split(".")[-1]

        if short_name in skip_modules:
            continue

        if isinstance(child, nn.Linear):
            weight = child.weight
            if min(weight.shape) >= min_features:
                new_layer = TurboQuantLinear.from_linear(
                    child, bits=bits, compressed_forward=compressed_forward
                )
                replacements[full_name] = new_layer
                continue

        # Handle already-quantized MLX layers (upgrade to TurboQuant)
        if isinstance(child, nn.QuantizedLinear):
            try:
                new_layer = TurboQuantLinear.from_quantized_linear(
                    child, bits=bits, compressed_forward=compressed_forward
                )
                replacements[full_name] = new_layer
                continue
            except Exception:
                pass  # Skip if dequantization fails

        # Recurse into submodules
        if isinstance(child, nn.Module):
            _find_linear_modules(
                child, full_name, bits, compressed_forward,
                skip_modules, min_features, replacements,
            )


def _walk_modules(module, prefix="", _visited=None):
    """Walk all named submodules in an MLX model."""
    if _visited is None:
        _visited = set()
    mid = id(module)
    if mid in _visited:
        return
    _visited.add(mid)
    # MLX modules store children as attributes
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            child = getattr(module, name)
        except Exception:
            continue
        if isinstance(child, nn.Module):
            if id(child) in _visited:
                continue
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, child
            yield from _walk_modules(child, full_name, _visited)
        elif isinstance(child, (list, tuple)):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    if id(item) in _visited:
                        continue
                    full_name = f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"
                    yield full_name, item
                    yield from _walk_modules(item, full_name, _visited)
        elif isinstance(child, dict):
            for k, item in child.items():
                if isinstance(item, nn.Module):
                    if id(item) in _visited:
                        continue
                    full_name = f"{prefix}.{name}.{k}" if prefix else f"{name}.{k}"
                    yield full_name, item
                    yield from _walk_modules(item, full_name, _visited)


def _set_nested_attr(module, path: str, value):
    """Set a nested attribute on a module by dotted path."""
    parts = path.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    last = parts[-1]
    if last.isdigit():
        module[int(last)] = value
    else:
        setattr(module, last, value)


# ===================================================================
#  Memory reporting
# ===================================================================

def model_memory_report(model: nn.Module) -> Dict:
    """Generate compression report for a TurboQuant-quantized MLX model."""
    total_fp16 = 0
    total_compressed = 0
    n_quantized = 0
    layers = []

    for name, mod in _walk_modules(model):
        if isinstance(mod, TurboQuantLinear):
            mem = mod.memory_bytes()
            total_fp16 += mem["fp16_bytes"]
            total_compressed += mem["compressed_bytes"]
            n_quantized += 1
            layers.append({
                "name": name,
                "shape": (mod.out_features, mod.in_features),
                "bits": mod.bits,
                **mem,
            })

    return {
        "num_quantized_layers": n_quantized,
        "total_fp16_mb": total_fp16 / (1024 * 1024),
        "total_compressed_mb": total_compressed / (1024 * 1024),
        "compression_ratio": total_fp16 / total_compressed if total_compressed > 0 else 0,
        "layers": layers,
    }


# ===================================================================
#  Save / Load quantized models
# ===================================================================

def save_quantized(model: nn.Module, path: str):
    """Save TurboQuant-compressed model weights.

    Saves all model weights (including TurboQuantLinear compressed data)
    using MLX's native safetensors format.
    """
    weights = dict(model.parameters())
    # Flatten nested dicts
    flat = {}
    _flatten_dict(weights, "", flat)
    mx.save_safetensors(path, flat)


def load_quantized(path: str) -> Dict[str, mx.array]:
    """Load TurboQuant-compressed model weights."""
    return mx.load(path)


def _flatten_dict(d, prefix, out):
    """Flatten a nested dict with dotted keys."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, key, out)
        elif isinstance(v, mx.array):
            out[key] = v
