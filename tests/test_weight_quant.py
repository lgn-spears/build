"""Tests for TurboQuant weight quantization."""

import pytest
import torch
import torch.nn as nn
import math
import os
import tempfile


class TestTurboQuantLinear:

    def test_from_linear_shape(self):
        """Quantized layer should produce same output shape."""
        from turboquant.weight_quant import TurboQuantLinear
        linear = nn.Linear(128, 64)
        tq = TurboQuantLinear.from_linear(linear, bits=3)
        x = torch.randn(2, 10, 128)
        out = tq(x)
        assert out.shape == (2, 10, 64)

    def test_from_linear_no_bias(self):
        from turboquant.weight_quant import TurboQuantLinear
        linear = nn.Linear(128, 64, bias=False)
        tq = TurboQuantLinear.from_linear(linear, bits=3)
        x = torch.randn(5, 128)
        out = tq(x)
        assert out.shape == (5, 64)

    def test_dequantize_forward_approximates_original(self):
        """Dequantize mode should approximate the original linear layer."""
        from turboquant.weight_quant import TurboQuantLinear
        torch.manual_seed(0)
        linear = nn.Linear(128, 64)
        x = torch.randn(10, 128)

        ref = linear(x)
        tq = TurboQuantLinear.from_linear(linear, bits=4)
        tq_out = tq(x)

        # Should be correlated
        cosine = nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            tq_out.flatten().unsqueeze(0)
        ).item()
        assert cosine > 0.9, f"Dequantize output too different: cosine={cosine}"

    def test_compressed_forward_approximates_original(self):
        """Compressed mode (Phase 2) should also approximate the original."""
        from turboquant.weight_quant import TurboQuantLinear
        torch.manual_seed(0)
        linear = nn.Linear(128, 64)
        x = torch.randn(10, 128)

        ref = linear(x)
        tq = TurboQuantLinear.from_linear(linear, bits=4, compressed_forward=True)
        tq_out = tq(x)

        cosine = nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            tq_out.flatten().unsqueeze(0)
        ).item()
        assert cosine > 0.85, f"Compressed output too different: cosine={cosine}"

    def test_compressed_vs_dequantize_correlation(self):
        """Phase 2 and Phase 1 should produce correlated results."""
        from turboquant.weight_quant import TurboQuantLinear
        torch.manual_seed(42)
        linear = nn.Linear(128, 64)
        x = torch.randn(10, 128)

        tq_deq = TurboQuantLinear.from_linear(linear, bits=3, compressed_forward=False)
        tq_comp = TurboQuantLinear.from_linear(linear, bits=3, compressed_forward=True)

        out_deq = tq_deq(x)
        out_comp = tq_comp(x)

        cosine = nn.functional.cosine_similarity(
            out_deq.flatten().unsqueeze(0),
            out_comp.flatten().unsqueeze(0)
        ).item()
        assert cosine > 0.85, f"Phase 1 vs Phase 2 too different: cosine={cosine}"

    def test_higher_bits_more_accurate(self):
        """More bits should give closer approximation to original."""
        from turboquant.weight_quant import TurboQuantLinear
        torch.manual_seed(0)
        linear = nn.Linear(128, 64)
        x = torch.randn(10, 128)
        ref = linear(x)

        errors = []
        for bits in [2, 3, 4]:
            tq = TurboQuantLinear.from_linear(linear, bits=bits)
            tq_out = tq(x)
            mse = ((ref - tq_out) ** 2).mean().item()
            errors.append(mse)

        assert errors[0] > errors[1] > errors[2], \
            f"Expected decreasing errors, got {errors}"

    def test_memory_savings(self):
        """Quantized layer should use less memory than FP16."""
        from turboquant.weight_quant import TurboQuantLinear
        linear = nn.Linear(1024, 1024)
        tq = TurboQuantLinear.from_linear(linear, bits=3)
        mem = tq.memory_bytes()
        assert mem["compression_ratio"] > 1.0
        assert mem["compressed_bytes"] < mem["fp16_bytes"]

    def test_extra_repr(self):
        from turboquant.weight_quant import TurboQuantLinear
        linear = nn.Linear(128, 64)
        tq = TurboQuantLinear.from_linear(linear, bits=3)
        s = repr(tq)
        assert "128" in s
        assert "64" in s
        assert "bits=3" in s


class TestQuantizeModel:

    def _make_tiny_model(self):
        """Create a tiny model for testing."""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        return model

    def test_quantize_replaces_linear(self):
        from turboquant.weight_quant import quantize_model, TurboQuantLinear
        model = self._make_tiny_model()
        quantize_model(model, bits=3, skip_modules=[])
        # All linear layers should be replaced
        for module in model.modules():
            assert not (isinstance(module, nn.Linear)
                        and not isinstance(module, TurboQuantLinear))

    def test_quantized_model_forward(self):
        from turboquant.weight_quant import quantize_model
        model = self._make_tiny_model()
        x = torch.randn(5, 128)

        # Forward before quantization
        ref = model(x)

        # Quantize and forward
        quantize_model(model, bits=4, skip_modules=[])
        quant_out = model(x)

        assert quant_out.shape == ref.shape
        cosine = nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            quant_out.flatten().unsqueeze(0)
        ).item()
        assert cosine > 0.8, f"Quantized model output diverged: cosine={cosine}"

    def test_skip_modules(self):
        """Layers in skip list should not be quantized."""
        from turboquant.weight_quant import quantize_model, TurboQuantLinear

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Linear(64, 128)
                self.hidden = nn.Linear(128, 128)
                self.lm_head = nn.Linear(128, 64)

            def forward(self, x):
                return self.lm_head(self.hidden(self.embed_tokens(x)))

        model = TinyLM()
        quantize_model(model, bits=3)

        # embed_tokens and lm_head should NOT be quantized
        assert isinstance(model.embed_tokens, nn.Linear)
        assert not isinstance(model.embed_tokens, TurboQuantLinear)
        assert isinstance(model.lm_head, nn.Linear)
        assert not isinstance(model.lm_head, TurboQuantLinear)
        # hidden should be quantized
        assert isinstance(model.hidden, TurboQuantLinear)

    def test_compressed_forward_model(self):
        """Phase 2 compressed forward on full model."""
        from turboquant.weight_quant import quantize_model
        model = self._make_tiny_model()
        x = torch.randn(5, 128)
        quantize_model(model, bits=3, compressed_forward=True, skip_modules=[])
        out = model(x)
        assert out.shape == (5, 64)
        assert torch.isfinite(out).all()


class TestSaveLoad:

    def test_save_load_roundtrip(self):
        from turboquant.weight_quant import (
            quantize_model, save_quantized, load_quantized, TurboQuantLinear
        )

        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        x = torch.randn(5, 128)

        quantize_model(model, bits=3, skip_modules=[])
        out_before = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_quantized(model, tmpdir)

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "quantized_weights.pt"))
            assert os.path.exists(os.path.join(tmpdir, "other_weights.pt"))
            assert os.path.exists(os.path.join(tmpdir, "turboquant_config.json"))

            # Load into fresh model
            model2 = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
            quantize_model(model2, bits=3, skip_modules=[])
            load_quantized(model2, tmpdir)
            out_after = model2(x)

            assert torch.allclose(out_before, out_after, atol=1e-5)


class TestMemoryReport:

    def test_report_structure(self):
        from turboquant.weight_quant import quantize_model, model_memory_report

        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        quantize_model(model, bits=3, skip_modules=[])
        report = model_memory_report(model)

        assert report["num_quantized_layers"] == 2
        assert report["compression_ratio"] > 1.0
        assert report["total_compressed_mb"] < report["total_fp16_mb"]
        assert len(report["layers"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
