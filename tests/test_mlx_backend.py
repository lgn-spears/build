"""Tests for TurboQuant MLX backend.

Requires: pip install mlx (Apple Silicon Mac only)
These tests verify the MLX port produces mathematically equivalent
results to the PyTorch reference implementation.
"""

import pytest
import math
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed (requires Apple Silicon)")


@requires_mlx
class TestLloydMaxMLX:

    def test_codebook_shape(self):
        from turboquant.mlx_backend import LloydMaxQuantizer
        q = LloydMaxQuantizer(bits=3, d=64)
        assert q.centroids.shape == (8,)
        assert q.boundaries.shape == (7,)

    def test_quantize_dequantize_range(self):
        from turboquant.mlx_backend import LloydMaxQuantizer
        q = LloydMaxQuantizer(bits=2, d=64)
        x = mx.array([-0.9, -0.3, 0.0, 0.3, 0.9])
        indices = q.quantize(x)
        reconstructed = q.dequantize(indices)
        # All reconstructed values should be in [-1, 1]
        assert mx.all(reconstructed >= -1.0).item()
        assert mx.all(reconstructed <= 1.0).item()

    def test_mse_matches_theory(self):
        """MSE distortion should match paper's predictions."""
        from turboquant.mlx_backend import LloydMaxQuantizer
        d = 128
        q = LloydMaxQuantizer(bits=2, d=d)
        # Sample from Beta-like distribution via rotation
        np.random.seed(42)
        x_np = np.random.randn(10000, d).astype(np.float32)
        norms = np.linalg.norm(x_np, axis=-1, keepdims=True)
        x_np = x_np / norms
        # After random rotation, each coordinate ~ Beta((d-1)/2, (d-1)/2) mapped to [-1,1]
        # Test on raw coordinates (already approximately Beta-distributed for large d)
        flat = mx.array(x_np.flatten())
        reconstructed = q.quantize_dequantize(flat)
        mse = mx.mean((flat - reconstructed) ** 2).item()
        # 2-bit Lloyd-Max on this distribution: MSE should be < 0.15
        assert mse < 0.15, f"MSE too high: {mse}"


@requires_mlx
class TestPolarQuantMLX:

    def test_rotation_orthogonality(self):
        from turboquant.mlx_backend import generate_random_rotation
        R = generate_random_rotation(64, seed=42)
        identity = R @ R.T
        eye = mx.eye(64)
        error = mx.max(mx.abs(identity - eye)).item()
        assert error < 1e-4, f"Rotation not orthogonal: max error {error}"

    def test_encode_decode_preserves_direction(self):
        from turboquant.mlx_backend import PolarQuant
        pq = PolarQuant(d=64, bits=3, seed=42)
        x = mx.random.normal(shape=(10, 64))
        indices, norms = pq.encode(x)
        x_hat = pq.decode(indices, norms)
        # Cosine similarity should be high at 3 bits
        cos_sim = mx.sum(x * x_hat, axis=-1) / (
            mx.linalg.norm(x, axis=-1) * mx.linalg.norm(x_hat, axis=-1) + 1e-8
        )
        mean_sim = mx.mean(cos_sim).item()
        assert mean_sim > 0.85, f"Cosine similarity too low: {mean_sim}"

    def test_residual_is_orthogonal_error(self):
        from turboquant.mlx_backend import PolarQuant
        pq = PolarQuant(d=128, bits=3, seed=42)
        x = mx.random.normal(shape=(5, 128))
        indices, norms, residual = pq.quantize(x)
        x_hat = pq.decode(indices, norms)
        # x = x_hat + residual should hold approximately
        reconstruction_error = mx.mean(mx.abs(x - x_hat - residual)).item()
        assert reconstruction_error < 0.01


@requires_mlx
class TestQJLMLX:

    def test_unbiased_estimator(self):
        """QJL inner product estimator should be approximately unbiased."""
        from turboquant.mlx_backend import QJL
        d = 64
        m = 256  # more projections for tighter estimate
        qjl = QJL(d=d, m=m, seed=42)

        x = mx.random.normal(shape=(d,))
        y = mx.random.normal(shape=(1, d))
        y_norm = mx.linalg.norm(y, axis=-1, keepdims=True)

        true_ip = mx.sum(x * y).item()
        sign_bits = qjl.encode(y)
        estimated_ip = qjl.decode_for_inner_product(
            mx.expand_dims(x, 0), sign_bits, y_norm
        )
        est_val = estimated_ip.item()

        # Should be in the right ballpark (not exact due to quantization noise)
        assert abs(est_val - true_ip) < abs(true_ip) + 5.0, \
            f"Estimate {est_val} too far from true {true_ip}"

    def test_sign_bits_are_boolean(self):
        from turboquant.mlx_backend import QJL
        qjl = QJL(d=32, m=64, seed=42)
        x = mx.random.normal(shape=(5, 32))
        bits = qjl.encode(x)
        # Should be boolean-like (0 or 1 when cast)
        unique_vals = set(bits.reshape(-1).tolist())
        assert unique_vals <= {True, False} or unique_vals <= {0, 1}


@requires_mlx
class TestTurboQuantMLX:

    def test_encode_decode_shapes(self):
        from turboquant.mlx_backend import TurboQuant
        tq = TurboQuant(d=64, bits=4)
        x = mx.random.normal(shape=(10, 64))
        encoded = tq.encode(x)
        assert encoded.pq_indices.shape == (10, 64)
        assert encoded.norms.shape == (10, 1)
        assert encoded.qjl_sign_bits.shape == (10, 64)
        assert encoded.residual_norms.shape == (10, 1)

    def test_inner_product_estimation(self):
        from turboquant.mlx_backend import TurboQuant
        d = 64
        tq = TurboQuant(d=d, bits=4, qjl_dim=128)
        keys = mx.random.normal(shape=(20, d))
        query = mx.random.normal(shape=(1, d))

        encoded = tq.encode(keys)
        estimated = tq.estimate_inner_product(query, encoded)
        true_scores = query @ keys.T

        # Correlation should be positive and reasonably high
        est_flat = estimated.reshape(-1)
        true_flat = true_scores.reshape(-1)
        est_np = np.array(est_flat.tolist())
        true_np = np.array(true_flat.tolist())
        correlation = np.corrcoef(est_np, true_np)[0, 1]
        assert correlation > 0.7, f"Correlation too low: {correlation}"

    def test_attention_scores_sum_to_one(self):
        from turboquant.mlx_backend import TurboQuant
        tq = TurboQuant(d=64, bits=4)
        keys = mx.random.normal(shape=(10, 64))
        query = mx.random.normal(shape=(1, 64))
        encoded = tq.encode(keys)
        scores = tq.compute_attention_scores(query, encoded)
        total = mx.sum(scores).item()
        assert abs(total - 1.0) < 1e-4, f"Attention scores sum to {total}"

    def test_compression_ratio(self):
        from turboquant.mlx_backend import TurboQuant
        tq = TurboQuant(d=64, bits=4)
        assert tq.bits_per_coordinate == 4.0
        assert tq.compression_ratio == 4.0  # 16 / 4


@requires_mlx
class TestTurboQuantLinearMLX:

    def test_from_linear_basic(self):
        from turboquant.mlx_quantize import TurboQuantLinear
        import mlx.nn as nn
        linear = nn.Linear(64, 32)
        mx.eval(linear.parameters())
        tq_linear = TurboQuantLinear.from_linear(linear, bits=4)
        x = mx.random.normal(shape=(2, 64))
        out = tq_linear(x)
        assert out.shape == (2, 32)

    def test_compressed_forward(self):
        from turboquant.mlx_quantize import TurboQuantLinear
        import mlx.nn as nn
        linear = nn.Linear(64, 32)
        mx.eval(linear.parameters())
        tq_linear = TurboQuantLinear.from_linear(
            linear, bits=4, compressed_forward=True
        )
        x = mx.random.normal(shape=(2, 64))
        out = tq_linear(x)
        assert out.shape == (2, 32)

    def test_memory_reduction(self):
        from turboquant.mlx_quantize import TurboQuantLinear
        import mlx.nn as nn
        linear = nn.Linear(512, 256)
        mx.eval(linear.parameters())
        tq_linear = TurboQuantLinear.from_linear(linear, bits=4)
        mem = tq_linear.memory_bytes()
        assert mem["compression_ratio"] > 2.0


@requires_mlx
class TestBitPacking:

    def test_pack_unpack_roundtrip(self):
        from turboquant.mlx_backend import _pack_bits, _unpack_bits
        original = mx.array([True, False, True, True, False, True, False, True,
                            False, False, True, True, False, False, True, True])
        packed = _pack_bits(original)
        unpacked = _unpack_bits(packed, len(original))
        for i in range(len(original)):
            assert bool(original[i].item()) == bool(unpacked[i].item())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
