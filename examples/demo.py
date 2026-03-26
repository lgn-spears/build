#!/usr/bin/env python3
"""TurboQuant demonstration: vector quantization and KV cache compression.

This script demonstrates:
  1. Basic vector quantization with distortion analysis
  2. Inner product preservation (unbiased estimation)
  3. KV cache compression for attention computation
  4. Compression ratio and memory savings analysis
"""

import torch
import numpy as np
import time


def demo_basic_quantization():
    """Demonstrate TurboQuant on random vectors with distortion analysis."""
    from turboquant import TurboQuant

    print("=" * 70)
    print("1. BASIC VECTOR QUANTIZATION")
    print("=" * 70)

    d = 128          # Vector dimension (typical attention head dim)
    n_vectors = 1000
    bits = 3          # Total bits per coordinate

    # Generate random vectors (simulating key embeddings)
    torch.manual_seed(0)
    x = torch.randn(n_vectors, d)

    # Initialize TurboQuant
    tq = TurboQuant(d=d, bits=bits)

    # Encode
    t0 = time.time()
    encoded = tq.encode(x)
    encode_time = time.time() - t0

    # Decode (PolarQuant reconstruction)
    x_hat = tq.decode(encoded)

    # Compute MSE distortion
    mse = ((x - x_hat) ** 2).mean().item()
    relative_mse = mse / (x ** 2).mean().item()

    print(f"  Dimension:           {d}")
    print(f"  Vectors:             {n_vectors}")
    print(f"  Bits per coordinate: {bits} ({bits-1} PolarQuant + 1 QJL)")
    print(f"  Encode time:         {encode_time*1000:.1f} ms")
    print(f"  MSE:                 {mse:.6f}")
    print(f"  Relative MSE:        {relative_mse:.4f} ({relative_mse*100:.2f}%)")
    print(f"  Compression ratio:   {tq.compression_ratio:.1f}x vs FP32")
    print()


def demo_inner_product_preservation():
    """Show that TurboQuant provides unbiased inner product estimation."""
    from turboquant import TurboQuant

    print("=" * 70)
    print("2. INNER PRODUCT PRESERVATION (UNBIASED ESTIMATION)")
    print("=" * 70)

    d = 128
    n_queries = 50
    n_keys = 200

    torch.manual_seed(42)
    queries = torch.randn(n_queries, d)
    keys = torch.randn(n_keys, d)

    # True inner products
    true_ip = queries @ keys.t()

    for bits in [2, 3, 4]:
        tq = TurboQuant(d=d, bits=bits)
        encoded_keys = tq.encode(keys)
        estimated_ip = tq.estimate_inner_product(
            queries.unsqueeze(0), encoded_keys).squeeze(0)

        # Check bias and accuracy
        error = estimated_ip - true_ip
        mean_error = error.mean().item()
        rmse = error.pow(2).mean().sqrt().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            true_ip.flatten().unsqueeze(0),
            estimated_ip.flatten().unsqueeze(0)
        ).item()

        print(f"  {bits}-bit TurboQuant:")
        print(f"    Mean error (bias):        {mean_error:+.6f}  (should be ~0)")
        print(f"    RMSE:                     {rmse:.4f}")
        print(f"    Cosine similarity:        {cosine_sim:.6f}")
        print()


def demo_kv_cache():
    """Demonstrate KV cache compression for attention computation."""
    from turboquant import TurboQuantKVCache

    print("=" * 70)
    print("3. KV CACHE COMPRESSION FOR ATTENTION")
    print("=" * 70)

    batch_size = 2
    n_heads = 8
    seq_len = 512
    head_dim = 64
    query_len = 1  # Single-token decoding

    torch.manual_seed(123)
    keys = torch.randn(batch_size, n_heads, seq_len, head_dim)
    values = torch.randn(batch_size, n_heads, seq_len, head_dim)
    queries = torch.randn(batch_size, n_heads, query_len, head_dim)

    # Reference: full-precision attention
    scale = 1.0 / (head_dim ** 0.5)
    ref_scores = (queries @ keys.transpose(-2, -1)) * scale
    ref_weights = torch.softmax(ref_scores, dim=-1, dtype=torch.float32)
    ref_output = (ref_weights.to(values.dtype) @ values)

    for key_bits in [3, 4]:
        cache = TurboQuantKVCache(
            head_dim=head_dim, n_kv_heads=n_heads,
            key_bits=key_bits)

        # Populate cache
        cache.update(keys, values, layer_idx=0)

        # Compute attention
        quant_output = cache.attend(queries, layer_idx=0)

        # Compare
        output_error = (quant_output - ref_output).pow(2).mean().sqrt().item()
        output_cosine = torch.nn.functional.cosine_similarity(
            ref_output.flatten().unsqueeze(0),
            quant_output.flatten().unsqueeze(0)
        ).item()

        mem = cache.memory_usage_bytes(layer_idx=0)

        print(f"  {key_bits}-bit keys (FP16 values):")
        print(f"    Output RMSE:          {output_error:.6f}")
        print(f"    Output cosine sim:    {output_cosine:.6f}")
        print(f"    Memory compression:   {mem['compression_ratio']:.1f}x")
        print(f"    Key memory:           {mem['key_bytes'] / 1024:.1f} KB")
        print(f"    Value memory:         {mem['value_bytes'] / 1024:.1f} KB")
        print()


def demo_vector_search():
    """Demonstrate TurboQuant for approximate nearest neighbor search."""
    from turboquant import TurboQuant

    print("=" * 70)
    print("4. VECTOR SEARCH (APPROXIMATE NEAREST NEIGHBOR)")
    print("=" * 70)

    d = 200       # GloVe-like dimensions
    n_database = 10000
    n_queries = 100
    k = 10        # top-k retrieval

    torch.manual_seed(7)
    database = torch.randn(n_database, d)
    queries = torch.randn(n_queries, d)

    # True top-k by inner product
    true_scores = queries @ database.t()
    _, true_topk = true_scores.topk(k, dim=-1)

    for bits in [2, 3, 4]:
        tq = TurboQuant(d=d, bits=bits)
        encoded_db = tq.encode(database)
        approx_scores = tq.estimate_inner_product(
            queries.unsqueeze(0), encoded_db).squeeze(0)
        _, approx_topk = approx_scores.topk(k, dim=-1)

        # Recall@k: fraction of true top-k recovered
        recall = 0.0
        for i in range(n_queries):
            true_set = set(true_topk[i].tolist())
            approx_set = set(approx_topk[i].tolist())
            recall += len(true_set & approx_set) / k
        recall /= n_queries

        print(f"  {bits}-bit TurboQuant: Recall@{k} = {recall:.4f}")

    print()


def demo_compression_analysis():
    """Analyze memory savings at different bit widths."""
    from turboquant import TurboQuantKVCache

    print("=" * 70)
    print("5. COMPRESSION ANALYSIS")
    print("=" * 70)

    head_dim = 128
    n_kv_heads = 8
    seq_len = 4096

    print(f"  Config: head_dim={head_dim}, n_kv_heads={n_kv_heads}, "
          f"seq_len={seq_len}")
    print()

    fp16_kv_bytes = seq_len * n_kv_heads * head_dim * 2 * 2  # keys + values
    print(f"  FP16 baseline:     {fp16_kv_bytes / 1024:.0f} KB per layer")
    print()

    dummy_k = torch.randn(1, n_kv_heads, seq_len, head_dim)
    dummy_v = torch.randn(1, n_kv_heads, seq_len, head_dim)

    for key_bits, val_bits in [(3, None), (3, 4), (4, None), (4, 4)]:
        cache = TurboQuantKVCache(
            head_dim=head_dim, n_kv_heads=n_kv_heads,
            key_bits=key_bits, value_bits=val_bits)
        cache.update(dummy_k, dummy_v, layer_idx=0)
        mem = cache.memory_usage_bytes(layer_idx=0)

        val_str = f"{val_bits}-bit" if val_bits else "FP16"
        print(f"  Keys={key_bits}-bit, Values={val_str}: "
              f"{mem['total_bytes'] / 1024:.0f} KB  "
              f"({mem['compression_ratio']:.1f}x compression)")

    print()


if __name__ == "__main__":
    print()
    print("  TurboQuant: Online Vector Quantization with Near-optimal")
    print("  Distortion Rate (ICLR 2026)")
    print()
    print("  Two-stage algorithm: PolarQuant (MSE-optimal) + QJL (unbiased IP)")
    print()

    demo_basic_quantization()
    demo_inner_product_preservation()
    demo_kv_cache()
    demo_vector_search()
    demo_compression_analysis()

    print("=" * 70)
    print("Done! All demos completed successfully.")
    print("=" * 70)
