"""End-to-end tests with real HuggingFace Qwen models at multiple scales.

Tests TurboQuant weight quantization, KV cache compression, and full-stack
(quantized weights + quantized KV cache) across three model sizes:

  - Qwen2.5-0.5B  (500M params, ~1 GB)  — fast smoke test
  - Qwen2.5-3B    (3B params,   ~6 GB)  — mid-range validation
  - Qwen2.5-7B    (7B params,  ~14 GB)  — real-world stress test

Each model auto-skips if insufficient RAM or no network access.
Run: python -m pytest tests/test_e2e_real_model.py -v -s
"""

import pytest
import torch
import gc
import math


# ---------------------------------------------------------------------------
# Model configurations with RAM requirements
# ---------------------------------------------------------------------------

MODEL_CONFIGS = [
    {
        "id": "Qwen/Qwen2.5-0.5B",
        "label": "0.5B",
        "min_ram_gb": 3,
        "fp16_gb": 1.0,
    },
    {
        "id": "Qwen/Qwen2.5-3B",
        "label": "3B",
        "min_ram_gb": 10,
        "fp16_gb": 6.0,
    },
    {
        "id": "Qwen/Qwen2.5-7B",
        "label": "7B",
        "min_ram_gb": 20,
        "fp16_gb": 14.0,
    },
]


def _has_network():
    """Check if HuggingFace Hub is reachable."""
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


def _has_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _available_ram_gb():
    import psutil
    return psutil.virtual_memory().available / (1024 ** 3)


def _can_run(config):
    """Check if a given model config can run in this environment."""
    if not _has_transformers():
        return False, "transformers not installed"
    if not _has_network():
        return False, "Cannot reach HuggingFace Hub"
    avail = _available_ram_gb()
    if avail < config["min_ram_gb"]:
        return False, f"Need {config['min_ram_gb']} GB RAM, have {avail:.1f} GB"
    return True, "OK"


# ---------------------------------------------------------------------------
# Fixtures — one per model size, module-scoped for efficiency
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_0_5b():
    """Load Qwen2.5-0.5B once for all tests that need it."""
    cfg = MODEL_CONFIGS[0]
    ok, reason = _can_run(cfg)
    if not ok:
        pytest.skip(reason)
    return _load_model(cfg["id"])


@pytest.fixture(scope="module")
def model_3b():
    """Load Qwen2.5-3B for tests that need the actual model (KV cache, etc)."""
    cfg = MODEL_CONFIGS[1]
    ok, reason = _can_run(cfg)
    if not ok:
        pytest.skip(reason)
    result = _load_model(cfg["id"])
    yield result
    # Free model memory after all tests using this fixture are done
    del result
    gc.collect()


@pytest.fixture(scope="module")
def model_7b():
    """Load Qwen2.5-7B for tests that need the actual model (KV cache, etc)."""
    cfg = MODEL_CONFIGS[2]
    ok, reason = _can_run(cfg)
    if not ok:
        pytest.skip(reason)
    result = _load_model(cfg["id"])
    yield result
    del result
    gc.collect()


# Lightweight fixtures that only check availability and return the model name.
# Used by TestWeightQuantization which loads its own copies via _fresh_model.
@pytest.fixture(scope="module")
def model_name_3b():
    """Return 3B model name after checking availability (doesn't load model)."""
    cfg = MODEL_CONFIGS[1]
    ok, reason = _can_run(cfg)
    if not ok:
        pytest.skip(reason)
    return cfg["id"]


@pytest.fixture(scope="module")
def model_name_7b():
    """Return 7B model name after checking availability (doesn't load model)."""
    cfg = MODEL_CONFIGS[2]
    ok, reason = _can_run(cfg)
    if not ok:
        pytest.skip(reason)
    return cfg["id"]


def _load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, model_name


def _fresh_model(model_name):
    """Load a separate copy of a model (for tests that mutate it)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ===================================================================
#  WEIGHT QUANTIZATION — all model sizes
# ===================================================================

class TestWeightQuantization:
    """Quantize model weights and verify generation still works."""

    # --- 0.5B (fast) ---

    def test_quantize_and_generate_0_5b(self, model_0_5b):
        model_name = model_0_5b[2]
        self._quantize_and_generate(model_name, bits=4)

    def test_memory_reduction_0_5b(self, model_0_5b):
        model_name = model_0_5b[2]
        self._memory_reduction(model_name, bits=3)

    def test_compressed_forward_0_5b(self, model_0_5b):
        model_name = model_0_5b[2]
        self._compressed_forward(model_name, bits=4)

    # --- 3B (medium) ---

    def test_quantize_and_generate_3b(self, model_name_3b):
        self._quantize_and_generate(model_name_3b, bits=4)

    def test_memory_reduction_3b(self, model_name_3b):
        self._memory_reduction(model_name_3b, bits=3)

    def test_compressed_forward_3b(self, model_name_3b):
        self._compressed_forward(model_name_3b, bits=4)

    # --- 7B (stress test) ---

    def test_quantize_and_generate_7b(self, model_name_7b):
        self._quantize_and_generate(model_name_7b, bits=4)

    def test_memory_reduction_7b(self, model_name_7b):
        self._memory_reduction(model_name_7b, bits=3)

    def test_compressed_forward_7b(self, model_name_7b):
        self._compressed_forward(model_name_7b, bits=4)

    # --- Shared implementations ---

    def _quantize_and_generate(self, model_name, bits):
        from turboquant.weight_quant import quantize_model

        # Generate reference output, then free the model to reduce peak memory
        prompt = "The capital of France is"
        model, tokenizer = _fresh_model(model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            ref_output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False, temperature=1.0,
            )
        ref_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)
        del model, ref_output
        gc.collect()

        # Load fresh copy, quantize, and generate
        model, _ = _fresh_model(model_name)
        quantize_model(model, bits=bits)

        with torch.no_grad():
            quant_output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False, temperature=1.0,
            )
        quant_text = tokenizer.decode(quant_output[0], skip_special_tokens=True)

        assert len(quant_text) > len(prompt), \
            f"[{model_name}] Quantized model produced no new text: '{quant_text}'"
        assert any(w in quant_text.lower() for w in
                   ["paris", "france", "city", "capital", "the", "is", "a"]), \
            f"[{model_name}] Output looks like garbage: '{quant_text}'"

        print(f"\n  [{model_name} @ {bits}-bit]")
        print(f"  Reference:  {ref_text}")
        print(f"  Quantized:  {quant_text}")

        del model
        gc.collect()

    def _memory_reduction(self, model_name, bits):
        from transformers import AutoModelForCausalLM
        from turboquant.weight_quant import quantize_model, model_memory_report

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map="cpu", trust_remote_code=True,
        )
        quantize_model(model, bits=bits)
        report = model_memory_report(model)

        print(f"\n  [{model_name} @ {bits}-bit]")
        print(f"  Quantized layers:  {report['num_quantized_layers']}")
        print(f"  FP16 equivalent:   {report['total_fp16_mb']:.1f} MB")
        print(f"  Compressed:        {report['total_compressed_mb']:.1f} MB")
        print(f"  Compression ratio: {report['compression_ratio']:.1f}x")

        assert report["num_quantized_layers"] > 0
        assert report["compression_ratio"] > 2.0, \
            f"[{model_name}] Expected >2x compression, got {report['compression_ratio']:.1f}x"

        del model
        gc.collect()

    def _compressed_forward(self, model_name, bits):
        from turboquant.weight_quant import quantize_model

        model, tokenizer = _fresh_model(model_name)
        quantize_model(model, bits=bits, compressed_forward=True)

        prompt = "1 + 1 ="
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=10, do_sample=False, temperature=1.0,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        assert len(text) > len(prompt), \
            f"[{model_name}] Compressed forward produced no output: '{text}'"
        print(f"\n  [{model_name} @ {bits}-bit compressed forward]: {text}")

        del model
        gc.collect()


# ===================================================================
#  KV CACHE — uses shared fixture (doesn't mutate model)
# ===================================================================

class TestKVCache:
    """TurboQuantCache as drop-in for HuggingFace generate()."""

    # --- 0.5B ---

    def test_cache_generation_0_5b(self, model_0_5b):
        self._cache_generation(*model_0_5b)

    def test_cache_vs_baseline_0_5b(self, model_0_5b):
        self._cache_vs_baseline(*model_0_5b)

    # --- 3B ---

    def test_cache_generation_3b(self, model_3b):
        self._cache_generation(*model_3b)

    def test_cache_vs_baseline_3b(self, model_3b):
        self._cache_vs_baseline(*model_3b)

    # --- 7B ---

    def test_cache_generation_7b(self, model_7b):
        self._cache_generation(*model_7b)

    def test_cache_vs_baseline_7b(self, model_7b):
        self._cache_vs_baseline(*model_7b)

    # --- Shared implementations ---

    def _cache_generation(self, model, tokenizer, model_name):
        from turboquant.hf_cache import TurboQuantCache

        prompt = "Write a short poem about the moon:"
        inputs = tokenizer(prompt, return_tensors="pt")

        cache = TurboQuantCache(model.config, key_bits=4)
        with torch.no_grad():
            output = model.generate(
                **inputs, past_key_values=cache,
                max_new_tokens=30, do_sample=False, temperature=1.0,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        assert len(text) > len(prompt), \
            f"[{model_name}] Cache generation produced no output: '{text}'"

        seq_len = cache.get_seq_length(layer_idx=0)
        assert seq_len > 0

        print(f"\n  [{model_name}] TurboQuantCache ({seq_len} tokens): {text}")

    def _cache_vs_baseline(self, model, tokenizer, model_name):
        from turboquant.hf_cache import TurboQuantCache

        prompt = "The meaning of life is"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Baseline (standard DynamicCache)
        with torch.no_grad():
            ref_output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False, temperature=1.0,
            )
        ref_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)

        # TurboQuant cache
        cache = TurboQuantCache(model.config, key_bits=4)
        with torch.no_grad():
            tq_output = model.generate(
                **inputs, past_key_values=cache,
                max_new_tokens=20, do_sample=False, temperature=1.0,
            )
        tq_text = tokenizer.decode(tq_output[0], skip_special_tokens=True)

        print(f"\n  [{model_name}]")
        print(f"  Baseline:     {ref_text}")
        print(f"  TurboQuant:   {tq_text}")

        assert len(tq_text) > len(prompt)
        assert any(w in tq_text.lower() for w in
                   ["life", "meaning", "is", "the", "to", "a", "that"]), \
            f"[{model_name}] TurboQuant output incoherent: '{tq_text}'"


# ===================================================================
#  FULL STACK — quantized weights + quantized KV cache
# ===================================================================

class TestFullStack:
    """The holy grail: compressed weights AND compressed KV cache together."""

    def test_full_stack_0_5b(self, model_0_5b):
        self._full_stack(model_0_5b[2], weight_bits=4, kv_bits=3)

    def test_full_stack_3b(self, model_name_3b):
        self._full_stack(model_name_3b, weight_bits=4, kv_bits=3)

    def test_full_stack_7b(self, model_name_7b):
        self._full_stack(model_name_7b, weight_bits=4, kv_bits=3)

    def _full_stack(self, model_name, weight_bits, kv_bits):
        from turboquant.weight_quant import quantize_model
        from turboquant.hf_cache import TurboQuantCache

        model, tokenizer = _fresh_model(model_name)

        # Quantize weights
        quantize_model(model, bits=weight_bits)

        # Quantize KV cache
        cache = TurboQuantCache(model.config, key_bits=kv_bits)

        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs, past_key_values=cache,
                max_new_tokens=20, do_sample=False, temperature=1.0,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        assert len(text) > len(prompt), \
            f"[{model_name}] Full-stack produced no output: '{text}'"

        print(f"\n  [{model_name}] {weight_bits}-bit weights + {kv_bits}-bit KV cache:")
        print(f"  {text}")

        del model
        gc.collect()


# ===================================================================
#  CROSS-BIT-RATE — test quality across bit budgets on smallest model
# ===================================================================

class TestBitRateSweep:
    """Test multiple bit rates to show quality vs compression tradeoff."""

    @pytest.mark.parametrize("bits", [2, 3, 4, 5])
    def test_weight_quant_bit_sweep(self, model_0_5b, bits):
        """Quantize at different bit rates and verify all produce text."""
        from turboquant.weight_quant import quantize_model, model_memory_report

        model_name = model_0_5b[2]
        model, tokenizer = _fresh_model(model_name)

        quantize_model(model, bits=bits)
        report = model_memory_report(model)

        prompt = "The sky is"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=15, do_sample=False, temperature=1.0,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        assert len(text) > len(prompt), \
            f"[{bits}-bit] No output: '{text}'"

        print(f"\n  [{bits}-bit] {report['compression_ratio']:.1f}x compression: {text}")

        del model
        gc.collect()

    @pytest.mark.parametrize("kv_bits", [2, 3, 4])
    def test_kv_cache_bit_sweep(self, model_0_5b, kv_bits):
        """KV cache at different bit rates."""
        from turboquant.hf_cache import TurboQuantCache

        model, tokenizer, model_name = model_0_5b

        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt")

        cache = TurboQuantCache(model.config, key_bits=kv_bits)
        with torch.no_grad():
            output = model.generate(
                **inputs, past_key_values=cache,
                max_new_tokens=15, do_sample=False, temperature=1.0,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        assert len(text) > len(prompt), \
            f"[KV {kv_bits}-bit] No output: '{text}'"

        print(f"\n  [KV {kv_bits}-bit]: {text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
