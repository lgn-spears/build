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
]
