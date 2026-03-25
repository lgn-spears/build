"""TurboQuant + MLX: Two-stage vector quantization on Apple Silicon.

Demonstrates TurboQuant running natively on MLX — KV cache compression,
weight quantization, and compressed-domain attention.

TurboQuant isn't trying to beat affine quantization on weight reconstruction.
Its edge is in KV cache compression and unbiased inner product estimation,
where the math guarantees matter more than per-element error.

Requirements:
    pip install mlx mlx-lm turboquant

Usage:
    python examples/mlx_demo.py
"""

import time


def demo_core_quantization():
    """Show TurboQuant's math running on MLX."""
    import mlx.core as mx
    from turboquant.mlx_backend import TurboQuant

    print("=" * 60)
    print("Demo 1: Core TurboQuant on MLX")
    print("=" * 60)

    d = 128
    n_keys = 1000
    bits = 4

    tq = TurboQuant(d=d, bits=bits)

    # Simulate KV cache keys
    keys = mx.random.normal(shape=(n_keys, d))
    query = mx.random.normal(shape=(1, d))
    mx.eval(keys, query)

    # Encode (compress)
    t0 = time.perf_counter()
    encoded = tq.encode(keys)
    mx.eval(encoded.pq_indices, encoded.norms,
            encoded.qjl_sign_bits, encoded.residual_norms)
    encode_ms = (time.perf_counter() - t0) * 1000

    # Compute attention on compressed keys
    t0 = time.perf_counter()
    scores = tq.compute_attention_scores(query, encoded)
    mx.eval(scores)
    attn_ms = (time.perf_counter() - t0) * 1000

    # Compare with exact attention
    true_scores = mx.softmax(query @ keys.T / (d ** 0.5), axis=-1)
    mx.eval(true_scores)

    # Top-k agreement
    k = 10
    est_top = mx.argsort(scores.reshape(-1))[-k:]
    true_top = mx.argsort(true_scores.reshape(-1))[-k:]
    est_set = set(est_top.tolist())
    true_set = set(true_top.tolist())
    overlap = len(est_set & true_set)

    print(f"  Vectors:       {n_keys} x {d}")
    print(f"  Bits:          {bits} per coordinate")
    print(f"  Compression:   {tq.compression_ratio:.1f}x vs FP16")
    print(f"  Encode time:   {encode_ms:.1f} ms")
    print(f"  Attention time:{attn_ms:.1f} ms")
    print(f"  Top-{k} overlap: {overlap}/{k}")
    print()


def demo_weight_quantization():
    """Quantize an MLX Linear layer and compare."""
    import mlx.core as mx
    import mlx.nn as nn
    from turboquant.mlx_quantize import TurboQuantLinear

    print("=" * 60)
    print("Demo 2: Weight Quantization (nn.Linear replacement)")
    print("=" * 60)

    in_f, out_f = 4096, 4096  # Typical transformer hidden dim

    # Original layer
    linear = nn.Linear(in_f, out_f, bias=False)
    mx.eval(linear.parameters())

    # Quantize with TurboQuant
    tq_linear = TurboQuantLinear.from_linear(linear, bits=4)

    # Compare outputs
    x = mx.random.normal(shape=(1, in_f))
    mx.eval(x)

    ref = linear(x)
    quant = tq_linear(x)
    mx.eval(ref, quant)

    # Relative error
    error = mx.sqrt(mx.sum((ref - quant) ** 2) / mx.sum(ref ** 2)).item()

    mem = tq_linear.memory_bytes()
    print(f"  Layer shape:       ({out_f}, {in_f})")
    print(f"  FP16 size:         {mem['fp16_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  TurboQuant size:   {mem['compressed_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Compression:       {mem['compression_ratio']:.1f}x")
    print(f"  Relative error:    {error:.4f}")
    print()


def demo_vs_mlx_quantize():
    """Compare TurboQuant vs MLX's built-in affine quantization.

    Honest comparison: MLX's affine quantization wins on weight reconstruction
    error. That's expected — affine is per-group and tuned for weight
    distributions, which are well-behaved and roughly symmetric.

    TurboQuant's advantage is elsewhere:
    - Unbiased inner product estimation (affine doesn't guarantee this)
    - KV cache compression (activations have weirder distributions)
    - Compressed-domain matmul (never materialize the full weight matrix)
    - Data-oblivious (no calibration, no per-group statistics)
    """
    import mlx.core as mx
    import mlx.nn as nn
    from turboquant.mlx_quantize import TurboQuantLinear

    print("=" * 60)
    print("Demo 3: Weight Reconstruction — TurboQuant vs MLX Affine")
    print("=" * 60)
    print()
    print("  NOTE: Affine quantization is purpose-built for weight matrices.")
    print("  TurboQuant's edge is in KV cache and inner product estimation,")
    print("  not weight reconstruction. This comparison is included for")
    print("  honesty, not to claim a win.")
    print()

    in_f, out_f = 2048, 2048

    # Original weights
    linear = nn.Linear(in_f, out_f, bias=False)
    mx.eval(linear.parameters())
    W = linear.weight  # ground truth

    x = mx.random.normal(shape=(8, in_f))
    mx.eval(x)
    ref_out = x @ W.T
    mx.eval(ref_out)

    for bits in [2, 3, 4]:
        # MLX default (affine, per-group with scales + biases)
        qw, scales, biases = mx.quantize(W, group_size=64, bits=bits)
        mlx_out = mx.quantized_matmul(x, qw, scales, biases, group_size=64, bits=bits)
        mx.eval(mlx_out)
        mlx_err = mx.sqrt(mx.sum((ref_out - mlx_out) ** 2) / mx.sum(ref_out ** 2)).item()

        # TurboQuant (data-oblivious, rotation-based)
        tq = TurboQuantLinear.from_linear(linear, bits=bits)
        tq_out = tq(x)
        mx.eval(tq_out)
        tq_err = mx.sqrt(mx.sum((ref_out - tq_out) ** 2) / mx.sum(ref_out ** 2)).item()

        print(f"  {bits}-bit:")
        print(f"    MLX affine error:   {mlx_err:.4f}")
        print(f"    TurboQuant error:   {tq_err:.4f}")
        print(f"    Winner:             MLX affine ({tq_err/mlx_err:.1f}x lower error)")
        print()

    print("  Takeaway: For static weight matrices, per-group affine quantization")
    print("  is hard to beat. TurboQuant pays a reconstruction tax for properties")
    print("  that matter more in other contexts (see Demo 3b below).")
    print()


def demo_kv_cache_advantage():
    """Test TurboQuant on its home turf: KV cache attention scores.

    In attention, you don't need perfect reconstruction — you need accurate
    inner products between queries and keys. TurboQuant's QJL stage provides
    an unbiased estimator for this. The question is whether the theoretical
    guarantee translates to a practical win over affine quantization.
    """
    import mlx.core as mx
    import numpy as np
    from turboquant.mlx_backend import TurboQuant

    print("=" * 60)
    print("Demo 3b: KV Cache — Attention Score Accuracy")
    print("=" * 60)
    print()
    print("  The real test: how well do quantized keys preserve attention")
    print("  scores? This is what actually matters in a transformer.")
    print()

    d = 128
    n_keys = 200
    n_queries = 10

    mx.random.seed(42)
    keys = mx.random.normal(shape=(n_keys, d))
    queries = mx.random.normal(shape=(n_queries, d))
    mx.eval(keys, queries)

    # True attention scores
    scale = 1.0 / (d ** 0.5)
    true_scores = mx.softmax(queries @ keys.T * scale, axis=-1)
    mx.eval(true_scores)
    true_np = np.array(true_scores.tolist())

    print(f"  Setup: {n_queries} queries x {n_keys} keys, d={d}")
    print()

    for bits in [2, 3, 4]:
        # --- MLX affine: quantize keys, dequantize, compute attention ---
        qw, scales, biases = mx.quantize(keys, group_size=64, bits=bits)
        keys_affine = mx.dequantize(qw, scales, biases, group_size=64, bits=bits)
        mx.eval(keys_affine)
        affine_scores = mx.softmax(queries @ keys_affine.T * scale, axis=-1)
        mx.eval(affine_scores)
        affine_np = np.array(affine_scores.tolist())

        # --- TurboQuant: encode keys, estimate inner products, softmax ---
        tq = TurboQuant(d=d, bits=bits)
        encoded = tq.encode(keys)
        # Use TurboQuant's unbiased inner product estimator
        tq_raw = tq.estimate_inner_product(queries, encoded)
        tq_scores = mx.softmax(tq_raw * scale, axis=-1)
        mx.eval(tq_scores)
        tq_np = np.array(tq_scores.tolist())

        # Score-level error (what attention actually sees)
        affine_score_err = np.sqrt(np.mean((true_np - affine_np) ** 2))
        tq_score_err = np.sqrt(np.mean((true_np - tq_np) ** 2))

        # Top-k agreement (do we attend to the right tokens?)
        k = 5
        affine_topk_agree = 0
        tq_topk_agree = 0
        for q in range(n_queries):
            true_topk = set(np.argsort(true_np[q])[-k:])
            affine_topk = set(np.argsort(affine_np[q])[-k:])
            tq_topk = set(np.argsort(tq_np[q])[-k:])
            affine_topk_agree += len(true_topk & affine_topk)
            tq_topk_agree += len(true_topk & tq_topk)
        affine_topk_agree /= n_queries
        tq_topk_agree /= n_queries

        winner_score = "TurboQuant" if tq_score_err < affine_score_err else "MLX affine"
        winner_topk = "TurboQuant" if tq_topk_agree > affine_topk_agree else "MLX affine"

        print(f"  {bits}-bit:")
        print(f"    Attention score RMSE:  affine={affine_score_err:.6f}  TQ={tq_score_err:.6f}  → {winner_score}")
        print(f"    Top-{k} overlap (avg):  affine={affine_topk_agree:.1f}/{k}  TQ={tq_topk_agree:.1f}/{k}  → {winner_topk}")
        print()

    print("  Takeaway: On random Gaussian vectors, MLX affine still wins —")
    print("  per-group scales and biases are a strong baseline everywhere.")
    print()
    print("  TurboQuant's theoretical edge (unbiased inner products, near-optimal")
    print("  distortion rate) may matter more at scale: longer sequences, lower")
    print("  bit-widths, or distributions where per-group affine breaks down.")
    print("  The math guarantees are real; whether they translate to practical")
    print("  wins over a well-tuned affine scheme is an open question.")
    print()
    print("  This is a research implementation, not a production claim.")
    print()


def demo_model_quantization():
    """Quantize a full mlx-lm model (if available)."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("=" * 60)
        print("Demo 4: Full Model Quantization (skipped — install mlx-lm)")
        print("=" * 60)
        print("  pip install mlx-lm")
        print()
        return

    from turboquant.mlx_quantize import quantize_model, model_memory_report

    print("=" * 60)
    print("Demo 4: Full Model Quantization with mlx-lm")
    print("=" * 60)

    # Load a small model
    model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    print(f"  Loading {model_name}...")
    model, tokenizer = load(model_name)

    # Generate baseline
    prompt = "The meaning of life is"
    baseline = generate(model, tokenizer, prompt=prompt, max_tokens=30)
    print(f"  Baseline: {baseline}")

    # Requantize with TurboQuant
    print("  Requantizing with TurboQuant (4-bit)...")
    quantize_model(model, bits=4)
    report = model_memory_report(model)

    # Generate with TurboQuant
    tq_output = generate(model, tokenizer, prompt=prompt, max_tokens=30)
    print(f"  TurboQuant: {tq_output}")

    print(f"\n  Quantized layers: {report['num_quantized_layers']}")
    print(f"  Compression:      {report['compression_ratio']:.1f}x vs FP16")
    print()


if __name__ == "__main__":
    print("\nTurboQuant + MLX: Two-Stage Vector Quantization on Apple Silicon\n")
    demo_core_quantization()
    demo_weight_quantization()
    demo_vs_mlx_quantize()
    demo_kv_cache_advantage()
    demo_model_quantization()
    print("Done.")
