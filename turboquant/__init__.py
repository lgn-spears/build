"""TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

A two-stage vector quantization algorithm combining:
  1. PolarQuant - Random rotation + Lloyd-Max scalar quantization
  2. QJL - 1-bit Quantized Johnson-Lindenstrauss residual correction

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", ICLR 2026.
"""

from turboquant.lloyd_max import LloydMaxQuantizer, compute_beta_codebook
from turboquant.polar_quant import PolarQuant
from turboquant.qjl import QJL
from turboquant.turboquant import TurboQuant
from turboquant.kv_cache import TurboQuantKVCache
from turboquant.weight_quant import (
    TurboQuantLinear, quantize_model, save_quantized,
    load_quantized, model_memory_report,
)

# HuggingFace integration (optional, requires transformers)
try:
    from turboquant.hf_cache import TurboQuantCache
except ImportError:
    TurboQuantCache = None

# MLX backend (optional, requires mlx — Apple Silicon only)
try:
    from turboquant import mlx_backend as mlx
    from turboquant.mlx_quantize import (
        TurboQuantLinear as MLXTurboQuantLinear,
        quantize_model as mlx_quantize_model,
        model_memory_report as mlx_memory_report,
    )
except ImportError:
    mlx = None
    MLXTurboQuantLinear = None
    mlx_quantize_model = None
    mlx_memory_report = None

__version__ = "0.1.0"
__all__ = [
    "LloydMaxQuantizer",
    "compute_beta_codebook",
    "PolarQuant",
    "QJL",
    "TurboQuant",
    "TurboQuantKVCache",
    "TurboQuantCache",
    "TurboQuantLinear",
    "quantize_model",
    "save_quantized",
    "load_quantized",
    "model_memory_report",
    # MLX backend
    "mlx",
    "MLXTurboQuantLinear",
    "mlx_quantize_model",
    "mlx_memory_report",
]
