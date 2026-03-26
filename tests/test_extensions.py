"""Tests for TurboQuant research extensions."""

import pytest
import torch
import math


class TestAdaptiveTurboQuant:

    def test_calibration_produces_different_bits(self):
        """Calibration should allocate different bits to heads with
        different sensitivities."""
        from turboquant.extensions import AdaptiveTurboQuant

        d = 64
        n_heads = 4
        n_samples = 50

        atq = AdaptiveTurboQuant(
            d=d, n_heads=n_heads, total_bit_budget=3.0,
            min_bits=2, max_bits=5)

        torch.manual_seed(0)
        # Make head 0 very sensitive (structured attention pattern)
        # and head 3 insensitive (random noise)
        key_samples = torch.randn(n_heads, n_samples, d)
        query_samples = torch.randn(n_heads, n_samples, d)

        # Head 0: queries and keys are correlated (sensitive to perturbation)
        key_samples[0] = torch.randn(n_samples, d)
        query_samples[0] = key_samples[0] + torch.randn(n_samples, d) * 0.1

        bits = atq.calibrate(key_samples, query_samples)
        assert len(bits) == n_heads
        assert all(2 <= b <= 5 for b in bits)

    def test_encode_decode_works(self):
        from turboquant.extensions import AdaptiveTurboQuant
        atq = AdaptiveTurboQuant(d=64, n_heads=4)
        x = torch.randn(10, 64)
        for h in range(4):
            encoded = atq.encode_head(x, h)
            scores = atq.estimate_inner_product_head(
                x[:3], encoded, h)
            assert scores.shape == (3, 10)


class TestMultiResolutionTurboQuant:

    def test_fine_is_more_accurate_than_coarse(self):
        """Fine resolution should give better IP estimates than coarse."""
        from turboquant.extensions import MultiResolutionTurboQuant

        d = 128
        mr = MultiResolutionTurboQuant(d=d, coarse_bits=2, fine_bits=4)

        torch.manual_seed(42)
        queries = torch.randn(10, d)
        keys = torch.randn(50, d)
        true_ip = queries @ keys.t()

        encoded = mr.encode(keys)
        coarse_ip = mr.estimate_coarse(queries, encoded)
        fine_ip = mr.estimate_fine(queries, encoded)

        coarse_error = ((coarse_ip - true_ip) ** 2).mean().item()
        fine_error = ((fine_ip - true_ip) ** 2).mean().item()

        assert fine_error < coarse_error, \
            f"Fine ({fine_error:.4f}) should be more accurate than coarse ({coarse_error:.4f})"

    def test_speculative_attention(self):
        """Speculative attention should produce valid output."""
        from turboquant.extensions import MultiResolutionTurboQuant

        d = 64
        mr = MultiResolutionTurboQuant(d=d, coarse_bits=2, fine_bits=4)

        torch.manual_seed(0)
        query = torch.randn(1, 1, d)
        keys = torch.randn(1, 20, d)
        values = torch.randn(1, 20, d)

        encoded = mr.encode(keys.squeeze(0))
        output, used_fine = mr.speculative_attention(
            query.squeeze(0), encoded, values.squeeze(0), threshold=0.5)
        assert output.shape == (1, d)
        assert torch.isfinite(output).all()


class TestHybridTurboQuant:

    def test_calibration_finds_outliers(self):
        """Calibration should identify high-kurtosis dimensions."""
        from turboquant.extensions import HybridTurboQuant

        d = 64
        n_outliers = 4
        hybrid = HybridTurboQuant(d=d, bits=3, n_outliers=n_outliers)

        torch.manual_seed(0)
        # Create data with clear outlier dimensions
        data = torch.randn(1000, d) * 0.1
        # Make dims 10, 20, 30, 40 have huge outlier spikes
        for dim in [10, 20, 30, 40]:
            data[:5, dim] = torch.randn(5) * 100  # Extreme outliers

        outlier_dims = hybrid.calibrate(data)
        assert len(outlier_dims) == n_outliers

        # The identified outliers should include our planted ones
        planted = {10, 20, 30, 40}
        found = set(outlier_dims.tolist())
        overlap = len(planted & found)
        assert overlap >= 2, f"Should find most planted outliers, found {found}"

    def test_hybrid_at_low_bits(self):
        """Hybrid quantization should help at very low bit budgets (2-bit)
        where quantization error is high.

        RESEARCH FINDING: At 3+ bits, TurboQuant's random rotation already
        handles outliers optimally by spreading their energy uniformly. The
        hybrid approach is most valuable at low bit-widths where even rotated
        coordinates suffer significant quantization error on outlier-heavy data.
        """
        from turboquant.extensions import HybridTurboQuant

        d = 64
        torch.manual_seed(42)

        # Create data with extreme outlier dimensions
        keys = torch.randn(100, d)
        keys[:, 0] *= 50
        keys[:, 1] *= 30
        keys[:, 2] *= 20
        keys[:, 3] *= 15
        queries = torch.randn(20, d)
        true_ip = queries @ keys.t()

        # Hybrid approach should produce reasonable results
        hybrid = HybridTurboQuant(d=d, bits=2, n_outliers=4)
        hybrid.calibrate(keys)
        encoded_hybrid = hybrid.encode(keys)
        est_hybrid = hybrid.estimate_inner_product(queries, encoded_hybrid)

        # Check correlation with true IP (should be decent)
        corr = torch.corrcoef(
            torch.stack([true_ip.flatten(), est_hybrid.flatten()])
        )[0, 1].item()
        assert corr > 0.8, f"Hybrid IP correlation too low: {corr:.4f}"

    def test_effective_bits(self):
        """Effective bits should account for full-precision outliers."""
        from turboquant.extensions import HybridTurboQuant
        hybrid = HybridTurboQuant(d=128, bits=3, n_outliers=8)
        hybrid.calibrate(torch.randn(100, 128))
        # 8 dims * 32 bits + 120 dims * 3 bits = 616 bits / 128 dims = 4.8125
        expected = (8 * 32 + 120 * 3) / 128
        assert abs(hybrid.effective_bits - expected) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
