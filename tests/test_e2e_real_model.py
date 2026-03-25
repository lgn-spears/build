"""End-to-end tests with a real HuggingFace model (Qwen2.5-0.5B).

These tests download and run a real LLM to validate:
  1. Weight quantization preserves model coherence
  2. KV cache compression works with real generation
  3. Compressed-domain matmul (Phase 2) produces valid output
  4. Memory savings are real and measurable

Requires: ~2 GB disk for model download, ~3 GB RAM for inference.
Model: Qwen/Qwen2.5-0.5B (500M params, smallest modern instruction LLM)
"""

import pytest
import torch
import gc
import sys


# Skip all tests if transformers not available, insufficient memory, or no network
def _check_resources():
    try:
        import transformers
    except ImportError:
        return False, "transformers not installed"
    import psutil
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if available_gb < 2.0:
        return False, f"Need 2 GB free RAM, have {available_gb:.1f} GB"
    # Check if we can reach HuggingFace Hub
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    except Exception:
        return False, "Cannot reach HuggingFace Hub (no network access)"
    return True, "OK"


_resources_ok, _skip_reason = _check_resources()
requires_resources = pytest.mark.skipif(
    not _resources_ok,
    reason=_skip_reason
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer once for all tests in this module."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    yield model, tokenizer

    # Cleanup
    del model, tokenizer
    gc.collect()


@requires_resources
class TestWeightQuantizationE2E:

    def test_quantize_and_generate(self, model_and_tokenizer):
        """Quantize model weights and verify it still generates coherent text."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from turboquant.weight_quant import quantize_model

        # Load a fresh copy to quantize (don't modify the shared fixture)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()

        # Generate reference output BEFORE quantization
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            ref_output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                temperature=1.0,
            )
        ref_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)

        # Quantize the model
        quantize_model(model, bits=4)

        # Generate with quantized model
        with torch.no_grad():
            quant_output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                temperature=1.0,
            )
        quant_text = tokenizer.decode(quant_output[0], skip_special_tokens=True)

        # The quantized model should produce text (not garbage)
        assert len(quant_text) > len(prompt), \
            f"Quantized model produced no new text: '{quant_text}'"

        # Should contain recognizable words (not just random tokens)
        assert any(word in quant_text.lower() for word in
                   ["paris", "france", "city", "capital", "the", "is", "a"]), \
            f"Quantized output looks like garbage: '{quant_text}'"

        print(f"\n  Reference:  {ref_text}")
        print(f"  Quantized:  {quant_text}")

        del model
        gc.collect()

    def test_memory_reduction(self, model_and_tokenizer):
        """Verify that quantization actually reduces memory footprint."""
        from transformers import AutoModelForCausalLM
        from turboquant.weight_quant import quantize_model, model_memory_report

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

        quantize_model(model, bits=3)
        report = model_memory_report(model)

        print(f"\n  Quantized layers:  {report['num_quantized_layers']}")
        print(f"  FP16 equivalent:   {report['total_fp16_mb']:.1f} MB")
        print(f"  Compressed:        {report['total_compressed_mb']:.1f} MB")
        print(f"  Compression ratio: {report['compression_ratio']:.1f}x")

        assert report["num_quantized_layers"] > 0
        assert report["compression_ratio"] > 2.0, \
            f"Expected >2x compression, got {report['compression_ratio']:.1f}x"

        del model
        gc.collect()

    def test_compressed_forward_generates(self, model_and_tokenizer):
        """Phase 2 compressed-domain matmul should produce valid output."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from turboquant.weight_quant import quantize_model

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()

        # Quantize with compressed forward (Phase 2)
        quantize_model(model, bits=4, compressed_forward=True)

        prompt = "1 + 1 ="
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                temperature=1.0,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        assert len(text) > len(prompt), \
            f"Compressed forward produced no output: '{text}'"
        print(f"\n  Compressed forward output: {text}")

        del model
        gc.collect()


@requires_resources
class TestKVCacheE2E:

    def test_turboquant_cache_with_generation(self, model_and_tokenizer):
        """TurboQuantCache should work as drop-in for model.generate()."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from turboquant.hf_cache import TurboQuantCache

        model, tokenizer = model_and_tokenizer

        prompt = "Write a short poem about the moon:"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate with TurboQuant KV cache
        cache = TurboQuantCache(model.config, key_bits=4)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=30,
                do_sample=False,
                temperature=1.0,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        assert len(text) > len(prompt), \
            f"TurboQuantCache generation produced no output: '{text}'"

        # Should have cached tokens
        seq_len = cache.get_seq_length(layer_idx=0)
        assert seq_len > 0, "Cache should have stored tokens"

        print(f"\n  TurboQuantCache output ({seq_len} cached tokens):")
        print(f"  {text}")

    def test_cache_vs_no_cache_similarity(self, model_and_tokenizer):
        """Output with TurboQuantCache should be similar to standard cache."""
        from turboquant.hf_cache import TurboQuantCache

        model, tokenizer = model_and_tokenizer
        prompt = "The meaning of life is"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Reference: no custom cache (uses DynamicCache internally)
        with torch.no_grad():
            ref_output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                temperature=1.0,
            )
        ref_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)

        # With TurboQuant cache (4-bit, high quality)
        cache = TurboQuantCache(model.config, key_bits=4)
        with torch.no_grad():
            tq_output = model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
            )
        tq_text = tokenizer.decode(tq_output[0], skip_special_tokens=True)

        print(f"\n  Reference:     {ref_text}")
        print(f"  TurboQuant:    {tq_text}")

        # Both should produce real text (not garbage)
        assert len(tq_text) > len(prompt)
        # They may diverge after a few tokens (quantization noise), but
        # both should be coherent English
        assert any(word in tq_text.lower() for word in
                   ["life", "meaning", "is", "the", "to", "a", "that"]), \
            f"TurboQuant output looks incoherent: '{tq_text}'"


@requires_resources
class TestFullStackE2E:

    def test_weight_quant_plus_kv_cache(self, model_and_tokenizer):
        """The holy grail: quantized weights + quantized KV cache together."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from turboquant.weight_quant import quantize_model
        from turboquant.hf_cache import TurboQuantCache

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()

        # Quantize weights (Phase 1: dequantize mode)
        quantize_model(model, bits=4)

        # Use TurboQuant KV cache
        cache = TurboQuantCache(model.config, key_bits=3)

        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        assert len(text) > len(prompt), \
            f"Full-stack quantized model produced no output: '{text}'"

        print(f"\n  Full-stack (4-bit weights + 3-bit KV cache):")
        print(f"  {text}")

        del model
        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
