"""TurboQuant + MLX: Next-gen quantization on Apple Silicon.

Demonstrates TurboQuant's two-stage vector quantization running natively
on MLX. This is a drop-in upgrade over MLX's default affine quantization,
delivering better quality at the same bit rate (or same quality at fewer bits).

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
    """Compare TurboQuant vs MLX's built-in affine quantization."""
    import mlx.core as mx
    import mlx.nn as nn
    from turboquant.mlx_quantize import TurboQuantLinear

    print("=" * 60)
    print("Demo 3: TurboQuant vs MLX Default Quantization")
    print("=" * 60)

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
        # MLX default (affine)
        qw, scales, biases = mx.quantize(W, group_size=64, bits=bits)
        mlx_out = mx.quantized_matmul(x, qw, scales, biases, group_size=64, bits=bits)
        mx.eval(mlx_out)
        mlx_err = mx.sqrt(mx.sum((ref_out - mlx_out) ** 2) / mx.sum(ref_out ** 2)).item()

        # TurboQuant
        tq = TurboQuantLinear.from_linear(linear, bits=bits)
        tq_out = tq(x)
        mx.eval(tq_out)
        tq_err = mx.sqrt(mx.sum((ref_out - tq_out) ** 2) / mx.sum(ref_out ** 2)).item()

        winner = "TurboQuant" if tq_err < mlx_err else "MLX affine"
        improvement = ((mlx_err - tq_err) / mlx_err * 100) if mlx_err > 0 else 0

        print(f"  {bits}-bit:")
        print(f"    MLX affine error:   {mlx_err:.4f}")
        print(f"    TurboQuant error:   {tq_err:.4f}")
        print(f"    Winner:             {winner} ({abs(improvement):.1f}% {'better' if improvement > 0 else 'worse'})")
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
    demo_model_quantization()
    print("Done.")
