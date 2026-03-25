# TurboQuant

**Online Vector Quantization with Near-optimal Distortion Rate**

A PyTorch implementation of TurboQuant, the two-stage vector quantization algorithm from [Zandieh et al. (ICLR 2026)](https://arxiv.org/abs/2504.19874).

TurboQuant compresses high-dimensional vectors with near-optimal distortion by combining:

1. **PolarQuant** (Stage 1): Random rotation + Lloyd-Max scalar quantization on the resulting Beta-distributed coordinates. Uses `b-1` bits per coordinate for MSE-optimal compression.

2. **QJL** (Stage 2): 1-bit Quantized Johnson-Lindenstrauss transform applied to the residual, providing an unbiased inner product estimator.

## Key Results

- **6x+ KV cache compression** with zero accuracy loss at 3.5 bits/channel
- **8x speedup** in attention computation (4-bit on H100)
- **Unbiased** inner product estimation (TurboQuant_prod)
- **Near-optimal**: within ~2.7x of information-theoretic lower bound
- **Data-oblivious**: no calibration or fine-tuning needed

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from turboquant import TurboQuant, TurboQuantKVCache

# Basic vector quantization
d, bits = 128, 3
tq = TurboQuant(d=d, bits=bits)

x = torch.randn(100, d)
encoded = tq.encode(x)
x_hat = tq.decode(encoded)  # MSE-optimal reconstruction

# Unbiased inner product estimation
queries = torch.randn(10, d)
scores = tq.estimate_inner_product(queries, encoded)

# KV cache for transformer attention
cache = TurboQuantKVCache(head_dim=128, n_kv_heads=8, key_bits=3)
cache.update(key_states, value_states, layer_idx=0)
output = cache.attend(query_states, layer_idx=0)
```

## Architecture

```
turboquant/
  lloyd_max.py    # Lloyd-Max quantizer for Beta distribution (Eq. 4)
  polar_quant.py  # PolarQuant: random rotation + scalar quantization (Algorithm 1)
  qjl.py          # QJL: 1-bit JL transform (Definition 1)
  turboquant.py   # TurboQuant: two-stage PolarQuant + QJL (Algorithm 2)
  kv_cache.py     # KV cache integration for LLM attention
```

## Algorithm Overview

From the paper:

**Algorithm 1 (TurboQuant_mse):**
1. Generate random rotation matrix Pi via QR decomposition
2. Construct codebook by solving continuous 1-D k-means (Lloyd-Max) on Beta distribution
3. Quantize: `y = Pi * x`, then `idx_j = argmin |y_j - c_k|`
4. Dequantize: `x_hat = Pi^T * [c_{idx_1}, ..., c_{idx_d}]`

**Algorithm 2 (TurboQuant_prod):**
1. Apply TurboQuant_mse with bit-width `b-1`
2. Compute residual `r = x - DeQuant_mse(idx)`
3. Apply QJL: `qjl = sign(S * r)`, store `||r||`
4. Dequantize: `x_hat = x_mse + sqrt(pi/2)/d * ||r|| * S^T * qjl`

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Running Demo

```bash
python examples/demo.py
```

## References

- Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Zandieh, Daliri, Han. "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead", AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- Han, Kacham, Karbasi, Mirrokni, Zandieh. "PolarQuant: Quantizing KV Caches with Polar Transformation", AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
