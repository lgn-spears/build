"""Tests for TurboQuant components."""

import pytest
import torch
import numpy as np
import math


class TestLloydMaxQuantizer:
    """Tests for the Lloyd-Max scalar quantizer."""

    def test_codebook_shape(self):
        from turboquant.lloyd_max import LloydMaxQuantizer
        q = LloydMaxQuantizer(bits=2, d=64)
        assert q.codebook.shape == (4,)
        assert q.boundaries.shape == (3,)

    def test_codebook_sorted(self):
        from turboquant.lloyd_max import LloydMaxQuantizer
        q = LloydMaxQuantizer(bits=3, d=128)
        diffs = q.codebook[1:] - q.codebook[:-1]
        assert (diffs > 0).all(), "Codebook must be sorted ascending"

    def test_codebook_in_range(self):
        from turboquant.lloyd_max import LloydMaxQuantizer
        q = LloydMaxQuantizer(bits=3, d=128)
        assert q.codebook.min() >= -1.0
        assert q.codebook.max() <= 1.0

    def test_codebook_symmetric(self):
        """Beta distribution is symmetric, so codebook should be ~symmetric."""
        from turboquant.lloyd_max import LloydMaxQuantizer
        q = LloydMaxQuantizer(bits=3, d=128)
        # Centroids should be approximately symmetric around 0
        for i in range(q.n_levels):
            j = q.n_levels - 1 - i
            assert abs(q.codebook[i].item() + q.codebook[j].item()) < 0.01

    def test_quantize_dequantize(self):
        from turboquant.lloyd_max import LloydMaxQuantizer
        q = LloydMaxQuantizer(bits=3, d=64)
        x = torch.linspace(-0.5, 0.5, 100)
        indices = q.quantize(x)
        assert indices.min() >= 0
        assert indices.max() < q.n_levels
        x_hat = q.dequantize(indices)
        # Dequantized values should be codebook entries
        for val in x_hat.unique():
            assert val in q.codebook

    def test_quantize_reduces_mse(self):
        """Quantized values should be closer to codebook entries than random."""
        from turboquant.lloyd_max import LloydMaxQuantizer
        q = LloydMaxQuantizer(bits=3, d=128)
        x = torch.randn(1000) * 0.1  # Concentrated near 0 like Beta
        x_hat = q.quantize_dequantize(x)
        mse = ((x - x_hat) ** 2).mean()
        # Random assignment would be much worse
        random_hat = q.codebook[torch.randint(0, q.n_levels, (1000,))]
        random_mse = ((x - random_hat) ** 2).mean()
        assert mse < random_mse


class TestPolarQuant:
    """Tests for PolarQuant (random rotation + scalar quantization)."""

    def test_rotation_orthogonal(self):
        from turboquant.polar_quant import generate_random_rotation
        d = 64
        R = generate_random_rotation(d, torch.device("cpu"))
        # R @ R^T should be identity
        eye = R @ R.t()
        assert torch.allclose(eye, torch.eye(d), atol=1e-5)

    def test_hadamard_orthogonal(self):
        from turboquant.polar_quant import generate_random_hadamard
        d = 64  # power of 2
        H = generate_random_hadamard(d, torch.device("cpu"))
        eye = H @ H.t()
        assert torch.allclose(eye, torch.eye(d), atol=1e-5)

    def test_encode_decode_shape(self):
        from turboquant import PolarQuant
        d = 64
        pq = PolarQuant(d=d, bits=3)
        x = torch.randn(10, d)
        indices, norms = pq.encode(x)
        assert indices.shape == (10, d)
        assert norms.shape == (10, 1)
        x_hat = pq.decode(indices, norms)
        assert x_hat.shape == (10, d)

    def test_norm_preservation(self):
        """Reconstruction should approximately preserve norms."""
        from turboquant import PolarQuant
        d = 128
        pq = PolarQuant(d=d, bits=4)
        x = torch.randn(100, d)
        indices, norms = pq.encode(x)
        x_hat = pq.decode(indices, norms)
        ratio = x_hat.norm(dim=-1) / x.norm(dim=-1)
        # Norms should be within ~20% for 4-bit
        assert (ratio - 1.0).abs().mean() < 0.2

    def test_quantization_error_decreases_with_bits(self):
        from turboquant import PolarQuant
        d = 128
        x = torch.randn(100, d)
        errors = []
        for bits in [2, 3, 4]:
            pq = PolarQuant(d=d, bits=bits)
            indices, norms = pq.encode(x)
            x_hat = pq.decode(indices, norms)
            mse = ((x - x_hat) ** 2).mean().item()
            errors.append(mse)
        # Higher bits = lower error
        assert errors[0] > errors[1] > errors[2]


class TestQJL:
    """Tests for QJL (Quantized Johnson-Lindenstrauss)."""

    def test_encode_shape(self):
        from turboquant import QJL
        d = 64
        m = 128
        qjl = QJL(d=d, m=m)
        x = torch.randn(10, d)
        sign_bits, norms = qjl.encode(x)
        assert sign_bits.shape == (10, m)
        assert sign_bits.dtype == torch.bool
        assert norms.shape == (10, 1)

    def test_inner_product_unbiased(self):
        """QJL should provide approximately unbiased inner product estimates."""
        from turboquant import QJL
        d = 128
        m = 256
        n_trials = 50

        torch.manual_seed(0)
        q = torch.randn(d)
        k = torch.randn(d)
        true_ip = (q * k).sum().item()

        estimates = []
        for seed in range(n_trials):
            qjl = QJL(d=d, m=m, seed=seed)
            sign_bits, norms = qjl.encode(k.unsqueeze(0))
            est = qjl.decode_for_inner_product(
                q.unsqueeze(0).unsqueeze(0), sign_bits, norms)
            estimates.append(est.item())

        mean_est = np.mean(estimates)
        # Mean estimate should be close to true IP (unbiased)
        assert abs(mean_est - true_ip) < abs(true_ip) * 0.3, \
            f"Bias too high: est={mean_est:.4f}, true={true_ip:.4f}"

    def test_pack_unpack_bits(self):
        from turboquant.qjl import _pack_bits, _unpack_bits
        bits = torch.randint(0, 2, (5, 100)).bool()
        packed = _pack_bits(bits)
        unpacked = _unpack_bits(packed, 100)
        assert torch.equal(bits, unpacked)


class TestTurboQuant:
    """Tests for the full TurboQuant two-stage algorithm."""

    def test_encode_decode_shape(self):
        from turboquant import TurboQuant
        d = 64
        tq = TurboQuant(d=d, bits=3)
        x = torch.randn(10, d)
        encoded = tq.encode(x)
        assert encoded.pq_indices.shape == (10, d)
        assert encoded.norms.shape == (10, 1)
        assert encoded.qjl_sign_bits.shape == (10, d)
        assert encoded.residual_norms.shape == (10, 1)
        x_hat = tq.decode(encoded)
        assert x_hat.shape == (10, d)

    def test_inner_product_estimation(self):
        from turboquant import TurboQuant
        d = 128
        tq = TurboQuant(d=d, bits=3)
        torch.manual_seed(0)
        q = torch.randn(1, 5, d)   # 5 queries
        k = torch.randn(1, 20, d)  # 20 keys
        true_ip = q @ k.transpose(-2, -1)  # (1, 5, 20)
        encoded = tq.encode(k.squeeze(0))
        est_ip = tq.estimate_inner_product(q.squeeze(0), encoded)
        # Check shape
        assert est_ip.shape == (5, 20)
        # Check correlation with true IP
        cosine = torch.nn.functional.cosine_similarity(
            true_ip.flatten().unsqueeze(0),
            est_ip.flatten().unsqueeze(0)
        ).item()
        assert cosine > 0.8, f"Inner product correlation too low: {cosine}"

    def test_attention_scores(self):
        from turboquant import TurboQuant
        d = 64
        tq = TurboQuant(d=d, bits=3)
        q = torch.randn(2, 4, d)
        k = torch.randn(2, 10, d)
        attn, encoded = tq.quantize_and_score(q, k)
        # Attention weights should sum to 1
        assert attn.shape == (2, 4, 10)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_bits_requirement(self):
        from turboquant import TurboQuant
        with pytest.raises(ValueError):
            TurboQuant(d=64, bits=1)

    def test_higher_bits_lower_distortion(self):
        from turboquant import TurboQuant
        d = 128
        x = torch.randn(100, d)
        errors = []
        for bits in [2, 3, 4]:
            tq = TurboQuant(d=d, bits=bits)
            x_hat = tq.decode(tq.encode(x))
            mse = ((x - x_hat) ** 2).mean().item()
            errors.append(mse)
        assert errors[0] > errors[1] > errors[2]


class TestKVCache:
    """Tests for the KV cache integration."""

    def test_update_and_attend(self):
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(head_dim=64, n_kv_heads=4, key_bits=3)
        keys = torch.randn(1, 4, 16, 64)
        values = torch.randn(1, 4, 16, 64)
        queries = torch.randn(1, 4, 1, 64)
        cache.update(keys, values, layer_idx=0)
        output = cache.attend(queries, layer_idx=0)
        assert output.shape == (1, 4, 1, 64)

    def test_incremental_update(self):
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(head_dim=64, n_kv_heads=2, key_bits=3)
        # Add 10 tokens
        for _ in range(10):
            k = torch.randn(1, 2, 1, 64)
            v = torch.randn(1, 2, 1, 64)
            cache.update(k, v, layer_idx=0)
        assert cache.get_seq_len(layer_idx=0) == 10
        # Attend with 1 query
        q = torch.randn(1, 2, 1, 64)
        out = cache.attend(q, layer_idx=0)
        assert out.shape == (1, 2, 1, 64)

    def test_memory_compression(self):
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(head_dim=128, n_kv_heads=8, key_bits=3)
        k = torch.randn(1, 8, 100, 128)
        v = torch.randn(1, 8, 100, 128)
        cache.update(k, v, layer_idx=0)
        mem = cache.memory_usage_bytes(layer_idx=0)
        assert mem["compression_ratio"] > 1.0
        assert mem["total_bytes"] > 0

    def test_clear(self):
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(head_dim=64, n_kv_heads=2, key_bits=3)
        k = torch.randn(1, 2, 5, 64)
        v = torch.randn(1, 2, 5, 64)
        cache.update(k, v, layer_idx=0)
        assert cache.get_seq_len(0) == 5
        cache.clear(layer_idx=0)
        assert cache.get_seq_len(0) == 0

    def test_value_quantization(self):
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(
            head_dim=64, n_kv_heads=2, key_bits=3, value_bits=4)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        cache.update(k, v, layer_idx=0)
        q = torch.randn(1, 2, 1, 64)
        out = cache.attend(q, layer_idx=0)
        assert out.shape == (1, 2, 1, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
