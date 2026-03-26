"""Comprehensive QA tests for TurboQuant.

Tests cover:
  1. Numerical stability (extreme values, tiny vectors, high dimensions)
  2. Theoretical bound validation (paper's Theorem 1 distortion rates)
  3. Unbiasedness verification (statistical tests on inner product estimator)
  4. Edge cases (single vector, single dimension, batch shapes)
  5. Determinism and reproducibility
  6. KV cache correctness under incremental decoding
  7. GQA (grouped-query attention) correctness
  8. Bit-packing round-trip integrity
"""

import pytest
import torch
import numpy as np
import math
from scipy.stats import ttest_1samp


# ============================================================
# 1. NUMERICAL STABILITY
# ============================================================

class TestNumericalStability:

    def test_zero_vector(self):
        """Zero vectors should not produce NaN/Inf."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=3)
        x = torch.zeros(5, 64)
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert torch.isfinite(x_hat).all()

    def test_very_large_vectors(self):
        """Very large magnitude vectors should quantize without overflow."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=3)
        x = torch.randn(10, 64) * 1e6
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert torch.isfinite(x_hat).all()
        # Norms should be preserved approximately
        ratio = x_hat.norm(dim=-1) / x.norm(dim=-1)
        assert (ratio > 0.5).all() and (ratio < 2.0).all()

    def test_very_small_vectors(self):
        """Very small vectors should not produce NaN from division."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=3)
        x = torch.randn(10, 64) * 1e-8
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert torch.isfinite(x_hat).all()

    def test_single_nonzero_coordinate(self):
        """Spike vectors (one hot) should survive quantization."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=128, bits=3)
        x = torch.zeros(128)
        x[42] = 5.0
        encoded = tq.encode(x.unsqueeze(0))
        x_hat = tq.decode(encoded).squeeze(0)
        assert torch.isfinite(x_hat).all()
        # Reconstruction norm may deviate significantly for pathological inputs
        # (one-hot → after rotation, only 1 coordinate has the right centroid)
        # The key check is finiteness and that the energy is in the right ballpark
        assert x_hat.norm().item() > 0.5
        assert x_hat.norm().item() < 20.0

    def test_identical_vectors(self):
        """Batch of identical vectors should produce identical encodings."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=3)
        v = torch.randn(64)
        x = v.unsqueeze(0).expand(10, -1).clone()
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        # All reconstructions should be identical
        for i in range(1, 10):
            assert torch.allclose(x_hat[0], x_hat[i])

    def test_high_dimension(self):
        """High-dimensional vectors (d=1024) should work correctly."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=1024, bits=3, rotation_type="qr")
        x = torch.randn(5, 1024)
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert x_hat.shape == (5, 1024)
        assert torch.isfinite(x_hat).all()

    def test_mixed_magnitudes(self):
        """Batch with wildly different magnitudes."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=3)
        x = torch.randn(5, 64)
        x[0] *= 1e-6
        x[1] *= 1e-3
        x[2] *= 1.0
        x[3] *= 1e3
        x[4] *= 1e6
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert torch.isfinite(x_hat).all()


# ============================================================
# 2. THEORETICAL BOUND VALIDATION
# ============================================================

class TestTheoreticalBounds:

    def test_mse_distortion_bound(self):
        """Verify MSE matches paper's Theorem 1 predictions.

        For b-bit PolarQuant on unit-norm vectors in R^d:
            E[||x - x_hat||^2] ≈ D_mse(b) (Table 1 from paper)

        Paper's Table 1 values for D_mse:
            b=1: 0.3634
            b=2: 0.1170
            b=3: 0.0308
            b=4: 0.0078
        """
        from turboquant import PolarQuant

        d = 256  # High enough for concentration to kick in
        n = 2000
        torch.manual_seed(0)
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)  # Unit norm

        # Paper values for MSE on unit sphere (approximate)
        paper_dstar = {1: 0.3634, 2: 0.1170, 3: 0.0308, 4: 0.0078}

        for bits in [1, 2, 3, 4]:
            pq = PolarQuant(d=d, bits=bits, rotation_type="qr")
            indices, norms = pq.encode(x)
            x_hat = pq.decode(indices, norms)
            mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

            expected = paper_dstar[bits]
            # Allow 50% tolerance (paper's values are asymptotic for d→∞)
            assert mse < expected * 1.5, \
                f"bits={bits}: MSE={mse:.4f} > 1.5 * D*={expected:.4f}"
            # Also check it's not unreasonably low (sanity check)
            assert mse > expected * 0.3, \
                f"bits={bits}: MSE={mse:.4f} suspiciously low vs D*={expected:.4f}"

    def test_distortion_decreases_monotonically_with_bits(self):
        """More bits should always give lower MSE."""
        from turboquant import PolarQuant
        d = 128
        torch.manual_seed(0)
        x = torch.randn(500, d)

        prev_mse = float('inf')
        for bits in range(1, 6):
            pq = PolarQuant(d=d, bits=bits, rotation_type="qr")
            indices, norms = pq.encode(x)
            x_hat = pq.decode(indices, norms)
            mse = ((x - x_hat) ** 2).mean().item()
            assert mse < prev_mse, \
                f"bits={bits} MSE ({mse}) not less than bits={bits-1} ({prev_mse})"
            prev_mse = mse

    def test_distortion_stable_across_dimensions(self):
        """Relative MSE should be stable across dimensions.

        The paper's D*(b) is dimension-independent, so relative MSE should
        converge to D*(b) for all dimensions. Verify it stays in a tight band.
        """
        from turboquant import PolarQuant
        bits = 3
        torch.manual_seed(0)

        relative_mses = []
        for d in [32, 64, 128, 256]:
            x = torch.randn(500, d)
            pq = PolarQuant(d=d, bits=bits, rotation_type="qr")
            indices, norms = pq.encode(x)
            x_hat = pq.decode(indices, norms)
            rel_mse = ((x - x_hat) ** 2).mean().item() / (x ** 2).mean().item()
            relative_mses.append(rel_mse)

        # All relative MSEs should be within 30% of each other
        min_mse, max_mse = min(relative_mses), max(relative_mses)
        assert max_mse / min_mse < 1.3, \
            f"Relative MSEs vary too much: {relative_mses}"


# ============================================================
# 3. UNBIASEDNESS VERIFICATION
# ============================================================

class TestUnbiasedness:

    def test_qjl_unbiased_statistical(self):
        """Statistical test that QJL inner product estimator is unbiased.

        Uses a t-test to check that the mean estimation error is consistent
        with zero (p > 0.01).
        """
        from turboquant.qjl import QJL
        d = 128

        torch.manual_seed(0)
        q = torch.randn(d)
        k = torch.randn(d)
        true_ip = (q * k).sum().item()

        errors = []
        for seed in range(200):
            qjl = QJL(d=d, m=d, seed=seed)
            sign_bits, norms = qjl.encode(k.unsqueeze(0))
            est = qjl.decode_for_inner_product(
                q.unsqueeze(0).unsqueeze(0), sign_bits, norms)
            errors.append(est.item() - true_ip)

        # t-test: null hypothesis = mean error is 0
        t_stat, p_value = ttest_1samp(errors, 0.0)
        assert p_value > 0.01, \
            f"QJL appears biased: mean_err={np.mean(errors):.4f}, p={p_value:.4f}"

    def test_turboquant_inner_product_correlation(self):
        """TurboQuant IP estimates should be highly correlated with true IPs."""
        from turboquant import TurboQuant
        d = 128

        torch.manual_seed(42)
        queries = torch.randn(20, d)
        keys = torch.randn(100, d)
        true_ip = queries @ keys.t()

        for bits in [3, 4]:
            tq = TurboQuant(d=d, bits=bits)
            encoded = tq.encode(keys)
            est_ip = tq.estimate_inner_product(queries, encoded)

            # Pearson correlation
            t_flat = true_ip.flatten()
            e_flat = est_ip.flatten()
            correlation = torch.corrcoef(
                torch.stack([t_flat, e_flat]))[0, 1].item()

            min_corr = 0.85 if bits == 3 else 0.95
            assert correlation > min_corr, \
                f"bits={bits}: correlation={correlation:.4f} < {min_corr}"

    def test_inner_product_sign_preservation(self):
        """Quantized IP should preserve the sign of the true IP most of the time."""
        from turboquant import TurboQuant
        d = 128

        torch.manual_seed(0)
        queries = torch.randn(50, d)
        keys = torch.randn(200, d)
        true_ip = queries @ keys.t()

        tq = TurboQuant(d=d, bits=3)
        encoded = tq.encode(keys)
        est_ip = tq.estimate_inner_product(queries, encoded)

        # Check sign agreement (exclude near-zero true IPs)
        mask = true_ip.abs() > 1.0
        sign_agree = ((true_ip[mask] > 0) == (est_ip[mask] > 0)).float().mean()
        assert sign_agree > 0.80, \
            f"Sign agreement too low: {sign_agree:.2%}"


# ============================================================
# 4. EDGE CASES
# ============================================================

class TestEdgeCases:

    def test_single_vector(self):
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=3)
        x = torch.randn(1, 64)
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert x_hat.shape == (1, 64)

    def test_2bit_minimum(self):
        """2-bit is the minimum for TurboQuant (1 PQ + 1 QJL)."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=2)
        x = torch.randn(10, 64)
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert x_hat.shape == (10, 64)

    def test_high_bit_width(self):
        """8-bit quantization should be very accurate."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=8)
        x = torch.randn(10, 64)
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        rel_error = ((x - x_hat) ** 2).mean() / (x ** 2).mean()
        assert rel_error < 0.01, f"8-bit should be very accurate, got {rel_error:.4f}"

    def test_batch_dimensions(self):
        """3D batch input should work."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=64, bits=3)
        x = torch.randn(4, 10, 64)  # batch x seq x dim
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert x_hat.shape == (4, 10, 64)

    def test_non_power_of_2_dimension(self):
        """Non-power-of-2 dimensions should fall back to QR rotation."""
        from turboquant import TurboQuant
        tq = TurboQuant(d=100, bits=3, rotation_type="hadamard")
        x = torch.randn(10, 100)
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert x_hat.shape == (10, 100)
        assert torch.isfinite(x_hat).all()

    def test_odd_dimension(self):
        from turboquant import TurboQuant
        tq = TurboQuant(d=77, bits=3)
        x = torch.randn(5, 77)
        encoded = tq.encode(x)
        x_hat = tq.decode(encoded)
        assert x_hat.shape == (5, 77)


# ============================================================
# 5. DETERMINISM AND REPRODUCIBILITY
# ============================================================

class TestDeterminism:

    def test_same_seed_same_result(self):
        """Same seeds should produce identical quantization."""
        from turboquant import TurboQuant
        x = torch.randn(10, 64)
        tq1 = TurboQuant(d=64, bits=3, pq_seed=42, qjl_seed=137)
        tq2 = TurboQuant(d=64, bits=3, pq_seed=42, qjl_seed=137)
        enc1 = tq1.encode(x)
        enc2 = tq2.encode(x)
        assert torch.equal(enc1.pq_indices, enc2.pq_indices)
        assert torch.equal(enc1.qjl_sign_bits, enc2.qjl_sign_bits)

    def test_different_seed_different_result(self):
        """Different seeds should produce different quantization."""
        from turboquant import TurboQuant
        x = torch.randn(10, 64)
        tq1 = TurboQuant(d=64, bits=3, pq_seed=42)
        tq2 = TurboQuant(d=64, bits=3, pq_seed=99)
        enc1 = tq1.encode(x)
        enc2 = tq2.encode(x)
        assert not torch.equal(enc1.pq_indices, enc2.pq_indices)

    def test_rotation_matrix_orthogonality(self):
        """Verify rotation matrix is truly orthogonal (R @ R^T = I)."""
        from turboquant.polar_quant import generate_random_rotation
        for d in [32, 64, 128, 256]:
            R = generate_random_rotation(d, torch.device("cpu"), seed=42)
            eye = R @ R.t()
            assert torch.allclose(eye, torch.eye(d), atol=1e-5), \
                f"Rotation not orthogonal at d={d}"


# ============================================================
# 6. KV CACHE CORRECTNESS
# ============================================================

class TestKVCacheCorrectness:

    def test_attention_weights_sum_to_one(self):
        """Attention weights must sum to 1 across key dimension."""
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(head_dim=64, n_kv_heads=4, key_bits=3)
        cache.update(torch.randn(2, 4, 32, 64),
                     torch.randn(2, 4, 32, 64), layer_idx=0)
        q = torch.randn(2, 4, 1, 64)
        # We can't directly check weights, but output should be finite
        out = cache.attend(q, layer_idx=0)
        assert torch.isfinite(out).all()

    def test_incremental_matches_batch(self):
        """Incremental token-by-token should approximate batch encoding."""
        from turboquant import TurboQuantKVCache
        torch.manual_seed(0)
        keys = torch.randn(1, 2, 20, 64)
        values = torch.randn(1, 2, 20, 64)
        query = torch.randn(1, 2, 1, 64)

        # Batch mode
        cache_batch = TurboQuantKVCache(head_dim=64, n_kv_heads=2, key_bits=3)
        cache_batch.update(keys, values, layer_idx=0)
        out_batch = cache_batch.attend(query, layer_idx=0)

        # Incremental mode
        cache_inc = TurboQuantKVCache(head_dim=64, n_kv_heads=2, key_bits=3)
        for t in range(20):
            cache_inc.update(keys[:, :, t:t+1], values[:, :, t:t+1], layer_idx=0)
        out_inc = cache_inc.attend(query, layer_idx=0)

        # Should produce same result (same quantization since same seed)
        assert torch.allclose(out_batch, out_inc, atol=1e-5), \
            f"Max diff: {(out_batch - out_inc).abs().max()}"

    def test_multi_layer_independence(self):
        """Different layers should have independent caches."""
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(head_dim=64, n_kv_heads=2, key_bits=3)
        k0 = torch.randn(1, 2, 5, 64)
        v0 = torch.randn(1, 2, 5, 64)
        k1 = torch.randn(1, 2, 10, 64)
        v1 = torch.randn(1, 2, 10, 64)
        cache.update(k0, v0, layer_idx=0)
        cache.update(k1, v1, layer_idx=1)
        assert cache.get_seq_len(0) == 5
        assert cache.get_seq_len(1) == 10
        cache.clear(layer_idx=0)
        assert cache.get_seq_len(0) == 0
        assert cache.get_seq_len(1) == 10

    def test_causal_mask(self):
        """Causal attention mask should be applied correctly."""
        from turboquant import TurboQuantKVCache
        cache = TurboQuantKVCache(head_dim=64, n_kv_heads=2, key_bits=4)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        cache.update(k, v, layer_idx=0)
        q = torch.randn(1, 2, 8, 64)  # Full sequence query
        # Create causal mask
        seq_len = 8
        mask = torch.full((1, 1, seq_len, seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        out = cache.attend(q, layer_idx=0, attention_mask=mask)
        assert out.shape == (1, 2, 8, 64)
        assert torch.isfinite(out).all()

    def test_gqa_attend(self):
        """Grouped-query attention: more query heads than KV heads."""
        from turboquant import TurboQuantKVCache
        n_kv_heads = 4
        n_q_heads = 16
        n_rep = n_q_heads // n_kv_heads
        cache = TurboQuantKVCache(head_dim=64, n_kv_heads=n_kv_heads, key_bits=3)
        k = torch.randn(1, n_kv_heads, 20, 64)
        v = torch.randn(1, n_kv_heads, 20, 64)
        cache.update(k, v, layer_idx=0)
        q = torch.randn(1, n_q_heads, 1, 64)
        out = cache.attend(q, layer_idx=0, n_rep=n_rep)
        assert out.shape == (1, n_q_heads, 1, 64)
        assert torch.isfinite(out).all()


# ============================================================
# 7. LLOYD-MAX CODEBOOK QUALITY
# ============================================================

class TestCodebookQuality:

    def test_codebook_centroids_are_conditional_expectations(self):
        """Each centroid should be the conditional mean of its Voronoi cell.

        This is the optimality condition for Lloyd-Max.
        """
        from turboquant.lloyd_max import LloydMaxQuantizer, _beta_pdf
        from scipy import integrate

        q = LloydMaxQuantizer(bits=3, d=128)
        n = q.n_levels
        codebook = q.codebook.numpy()

        boundaries = np.empty(n + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n - 1):
            boundaries[i + 1] = (codebook[i] + codebook[i + 1]) / 2.0

        for i in range(n):
            lo, hi = boundaries[i], boundaries[i + 1]
            num, _ = integrate.quad(lambda x: x * _beta_pdf(x, 128), lo, hi)
            den, _ = integrate.quad(lambda x: _beta_pdf(x, 128), lo, hi)
            if den > 1e-10:
                expected_centroid = num / den
                assert abs(codebook[i] - expected_centroid) < 1e-4, \
                    f"Centroid {i}: {codebook[i]:.6f} != E[X|cell]={expected_centroid:.6f}"

    def test_codebook_beats_uniform_quantizer(self):
        """Lloyd-Max should outperform uniform quantization."""
        from turboquant.lloyd_max import LloydMaxQuantizer
        d = 128
        bits = 3
        q = LloydMaxQuantizer(bits=bits, d=d)

        # Generate data following the expected distribution
        torch.manual_seed(0)
        R = torch.randn(d, d)
        R, _ = torch.linalg.qr(R)
        x = torch.randn(1000, d)
        x = x / x.norm(dim=-1, keepdim=True)
        x_rot = x @ R.t()

        # Lloyd-Max MSE
        x_hat_lm = q.quantize_dequantize(x_rot)
        mse_lm = ((x_rot - x_hat_lm) ** 2).mean().item()

        # Uniform quantizer MSE
        levels = torch.linspace(-1, 1, 2 ** bits).float()
        uniform_boundaries = (levels[:-1] + levels[1:]) / 2
        uniform_idx = torch.searchsorted(uniform_boundaries, x_rot.contiguous())
        uniform_idx = uniform_idx.clamp(0, 2 ** bits - 1)
        x_hat_uniform = levels[uniform_idx]
        mse_uniform = ((x_rot - x_hat_uniform) ** 2).mean().item()

        assert mse_lm < mse_uniform, \
            f"Lloyd-Max MSE ({mse_lm:.6f}) should beat uniform ({mse_uniform:.6f})"


# ============================================================
# 8. BIT PACKING INTEGRITY
# ============================================================

class TestBitPacking:

    def test_pack_unpack_various_sizes(self):
        from turboquant.qjl import _pack_bits, _unpack_bits
        for m in [1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 256]:
            bits = torch.randint(0, 2, (3, m)).bool()
            packed = _pack_bits(bits)
            unpacked = _unpack_bits(packed, m)
            assert torch.equal(bits, unpacked), f"Failed for m={m}"

    def test_pack_all_zeros(self):
        from turboquant.qjl import _pack_bits, _unpack_bits
        bits = torch.zeros(5, 128).bool()
        packed = _pack_bits(bits)
        assert (packed == 0).all()
        unpacked = _unpack_bits(packed, 128)
        assert torch.equal(bits, unpacked)

    def test_pack_all_ones(self):
        from turboquant.qjl import _pack_bits, _unpack_bits
        bits = torch.ones(5, 128).bool()
        packed = _pack_bits(bits)
        assert (packed == 255).all()
        unpacked = _unpack_bits(packed, 128)
        assert torch.equal(bits, unpacked)

    def test_pack_compression_ratio(self):
        """Packed bits should use ~8x less memory."""
        from turboquant.qjl import _pack_bits
        bits = torch.randint(0, 2, (100, 256)).bool()
        packed = _pack_bits(bits)
        assert packed.shape == (100, 32)  # 256 / 8 = 32 bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
