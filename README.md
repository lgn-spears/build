# TurboQuant

**The thing behind the thing for LLM compression on Apple Silicon.**

First pip-installable implementation of TurboQuant — the two-stage vector quantizer from [Zandieh et al. (ICLR 2026)](https://arxiv.org/abs/2504.19874). Compresses KV caches, quantizes weights, and does it all natively on Metal via MLX. No calibration. No fine-tuning. Just math that works.

Two stages. One idea: make vectors smaller without making them dumber.

1. **PolarQuant** — Random rotation + Lloyd-Max scalar quantization. Takes your high-dimensional vectors, spins them into a Beta distribution, and finds the MSE-optimal codebook. `b-1` bits per coordinate. Clean.

2. **QJL** — 1-bit Quantized Johnson-Lindenstrauss on the residual. The correction term that makes inner products unbiased. One extra bit, massive accuracy recovery.

## What it actually does

- **4x compression** at 4-bit, **5x+** at 3-bit on real models (Qwen 0.5B–7B)
- **Drop-in HuggingFace cache** — swap `DynamicCache` for `TurboQuantCache`, done
- **Weight quantization** — replace `nn.Linear` layers, never materialize full weights again
- **Compressed-domain matmul** — compute `y = x @ W^T` directly on quantized weights. The weight matrix never comes back to life. Novel research contribution.
- **Native MLX backend** — runs on Apple Silicon Metal, not a CPU afterthought
- **Data-oblivious** — no calibration set, no fine-tuning pass, no begging the model to cooperate
- **Near-optimal** — within ~2.7x of the information-theoretic lower bound. That's the floor. We're close to it.

## Get started

```bash
pip install -e ".[dev]"        # PyTorch backend
pip install -e ".[dev,mlx]"    # + MLX on Apple Silicon
```

```python
import torch
from turboquant import TurboQuant, TurboQuantKVCache

# Compress vectors
tq = TurboQuant(d=128, bits=3)
encoded = tq.encode(torch.randn(100, 128))
x_hat = tq.decode(encoded)

# Estimate inner products without decompressing
scores = tq.estimate_inner_product(queries, encoded)

# Drop into HuggingFace generation
from turboquant.hf_cache import TurboQuantCache
cache = TurboQuantCache(model.config, key_bits=4)
output = model.generate(**inputs, past_key_values=cache)
```

## MLX (Apple Silicon)

```python
from turboquant.mlx_backend import TurboQuant as TurboQuantMLX
from turboquant.mlx_quantize import quantize_model, TurboQuantLinear

# Weight quantization on MLX models
quantize_model(model, bits=4)

# Or go full send — compressed-domain forward pass
quantize_model(model, bits=4, compressed_forward=True)
# ^ never builds the full weight matrix. Just vibes and math.
```

## Architecture

```
turboquant/
  lloyd_max.py      # Lloyd-Max quantizer for Beta distribution
  polar_quant.py    # PolarQuant: rotation + scalar quantization
  qjl.py            # QJL: 1-bit JL transform
  turboquant.py     # Two-stage: PolarQuant + QJL
  kv_cache.py       # KV cache for transformer attention
  hf_cache.py       # HuggingFace DynamicCache drop-in
  mlx_backend.py    # Native MLX/Metal implementation
  mlx_quantize.py   # MLX weight quantization + nn.Linear replacement
  weight_quant.py   # PyTorch weight quantization
```

## Tests

```bash
# Offline suite (fast, no model downloads)
pytest tests/test_turboquant.py tests/test_qa_comprehensive.py tests/test_extensions.py tests/test_weight_quant.py -v

# MLX backend
pytest tests/test_mlx_backend.py -v

# End-to-end with real models (slow, downloads Qwen)
pytest tests/test_e2e_real_model.py -v -s

# The demo
python examples/mlx_demo.py
```

## How it works (for the curious)

**Stage 1 — PolarQuant (Algorithm 1):**
1. Generate random orthogonal matrix via QR
2. Build codebook by solving continuous 1-D k-means on the Beta distribution
3. Rotate vector, quantize each coordinate to nearest centroid
4. Dequantize: inverse rotation on the codebook values

**Stage 2 — TurboQuant (Algorithm 2):**
1. Run PolarQuant at `b-1` bits
2. Compute residual (what PolarQuant missed)
3. Apply QJL: random projection → sign bits. Store residual norm.
4. Reconstruct: PolarQuant output + scaled sign-corrected projection

The key insight: Stage 2 costs exactly 1 bit per coordinate but makes inner product estimation unbiased. That's the engine that makes attention work at low bit-widths.

## References

- Zandieh, Daliri, Hadian, Mirrokni. *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*, ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Zandieh, Daliri, Han. *"QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead"*, AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- Han, Kacham, Karbasi, Mirrokni, Zandieh. *"PolarQuant: Quantizing KV Caches with Polar Transformation"*, AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
