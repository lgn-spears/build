"""TurboQuant KV Cache for transformer attention.

Drop-in replacement for standard KV caches in transformer models. Quantizes
key vectors using TurboQuant while keeping value vectors at configurable
precision (full, or separately quantized with PolarQuant).

Supports:
  - Incremental (autoregressive) decoding: append one token at a time
  - Prefill: batch-encode all key/value pairs
  - Multi-head attention with per-head quantization
  - Grouped-query attention (GQA)

Usage:
    cache = TurboQuantKVCache(head_dim=128, n_kv_heads=8, key_bits=3)

    # During prefill
    cache.update(key_states, value_states, layer_idx=0)

    # During generation
    attn_output = cache.attend(query_states, layer_idx=0)
"""

from typing import Optional, Dict, Tuple, List
import math

import torch
import torch.nn.functional as F

from turboquant.turboquant import TurboQuant, TurboQuantEncoded
from turboquant.polar_quant import PolarQuant


class TurboQuantKVCache:
    """Quantized KV cache using TurboQuant for keys and optional quantization
    for values.

    Keys are quantized with TurboQuant (PolarQuant + QJL) for unbiased
    attention score computation. Values can be kept at full precision or
    quantized with PolarQuant for additional memory savings.
    """

    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        key_bits: int = 3,
        value_bits: Optional[int] = None,
        max_seq_len: int = 32768,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        rotation_type: str = "hadamard",
    ):
        """Initialize TurboQuant KV cache.

        Args:
            head_dim: Dimension of each attention head.
            n_kv_heads: Number of key/value heads.
            key_bits: Bits per coordinate for key quantization (>= 2).
            value_bits: Bits per coordinate for value quantization.
                None means full precision values.
            max_seq_len: Maximum sequence length for pre-allocation.
            device: Torch device.
            dtype: Data type for full-precision storage.
            rotation_type: Rotation type for PolarQuant ("hadamard" or "qr").
        """
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Create per-head TurboQuant quantizers for keys
        # All heads share the same random matrices for efficiency
        self.key_quantizer = TurboQuant(
            d=head_dim, bits=key_bits, device=self.device,
            rotation_type=rotation_type)

        # Optional value quantizer (PolarQuant only, no QJL needed)
        self.value_quantizer = None
        if value_bits is not None:
            self.value_quantizer = PolarQuant(
                d=head_dim, bits=value_bits, device=self.device,
                rotation_type=rotation_type, seed=314)

        # Storage: per-layer caches
        self._key_cache: Dict[int, TurboQuantEncoded] = {}
        self._value_cache: Dict[int, torch.Tensor] = {}
        self._value_indices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._seq_lens: Dict[int, int] = {}

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
        layer_idx: int
    ) -> None:
        """Add key/value states to the cache.

        Can be called incrementally (one token at a time) or in batch
        (prefill with full sequence).

        Args:
            key_states: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim).
            value_states: Value tensor of shape (batch, n_kv_heads, seq_len, head_dim).
            layer_idx: Transformer layer index.
        """
        batch_size, n_heads, seq_len, d = key_states.shape

        # Reshape for quantization: merge batch and heads
        keys_flat = key_states.reshape(-1, seq_len, d)
        encoded_keys = self.key_quantizer.encode(keys_flat)

        # Reshape back to (batch, n_heads, seq_len, ...)
        def reshape_encoded(enc: TurboQuantEncoded) -> TurboQuantEncoded:
            return TurboQuantEncoded(
                pq_indices=enc.pq_indices.reshape(batch_size, n_heads, seq_len, d),
                norms=enc.norms.reshape(batch_size, n_heads, seq_len, 1),
                qjl_sign_bits=enc.qjl_sign_bits.reshape(batch_size, n_heads, seq_len, -1),
                residual_norms=enc.residual_norms.reshape(batch_size, n_heads, seq_len, 1),
            )

        encoded_keys = reshape_encoded(encoded_keys)

        # Append to existing cache or create new
        if layer_idx in self._key_cache:
            existing = self._key_cache[layer_idx]
            self._key_cache[layer_idx] = TurboQuantEncoded(
                pq_indices=torch.cat([existing.pq_indices, encoded_keys.pq_indices], dim=2),
                norms=torch.cat([existing.norms, encoded_keys.norms], dim=2),
                qjl_sign_bits=torch.cat([existing.qjl_sign_bits, encoded_keys.qjl_sign_bits], dim=2),
                residual_norms=torch.cat([existing.residual_norms, encoded_keys.residual_norms], dim=2),
            )
        else:
            self._key_cache[layer_idx] = encoded_keys

        # Store values (full precision or quantized)
        if self.value_quantizer is not None:
            vals_flat = value_states.reshape(-1, seq_len, d)
            v_indices, v_norms = self.value_quantizer.encode(vals_flat)
            v_indices = v_indices.reshape(batch_size, n_heads, seq_len, d)
            v_norms = v_norms.reshape(batch_size, n_heads, seq_len, 1)
            if layer_idx in self._value_indices:
                ei, en = self._value_indices[layer_idx]
                self._value_indices[layer_idx] = (
                    torch.cat([ei, v_indices], dim=2),
                    torch.cat([en, v_norms], dim=2),
                )
            else:
                self._value_indices[layer_idx] = (v_indices, v_norms)
        else:
            if layer_idx in self._value_cache:
                self._value_cache[layer_idx] = torch.cat(
                    [self._value_cache[layer_idx], value_states], dim=2)
            else:
                self._value_cache[layer_idx] = value_states

        self._seq_lens[layer_idx] = self._seq_lens.get(layer_idx, 0) + seq_len

    def attend(
        self, query_states: torch.Tensor, layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        n_rep: int = 1,
    ) -> torch.Tensor:
        """Compute attention output using quantized KV cache.

        Args:
            query_states: Query tensor (batch, n_q_heads, seq_q, head_dim).
            layer_idx: Transformer layer index.
            attention_mask: Optional mask (batch, 1, seq_q, seq_k).
            n_rep: GQA repetition factor (n_q_heads // n_kv_heads).

        Returns:
            Attention output of shape (batch, n_q_heads, seq_q, head_dim).
        """
        encoded_keys = self._key_cache[layer_idx]
        scale = 1.0 / math.sqrt(self.head_dim)

        # Handle GQA by repeating KV heads
        if n_rep > 1:
            encoded_keys = _repeat_kv_encoded(encoded_keys, n_rep)

        # Compute attention scores using TurboQuant
        attn_scores = self.key_quantizer.estimate_inner_product(
            query_states, encoded_keys)
        attn_scores = attn_scores * scale

        # Apply attention mask (causal or padding)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)

        # Get value states
        if self.value_quantizer is not None:
            v_indices, v_norms = self._value_indices[layer_idx]
            if n_rep > 1:
                v_indices = _repeat_kv(v_indices, n_rep)
                v_norms = _repeat_kv(v_norms, n_rep)
            value_states = self.value_quantizer.decode(v_indices, v_norms)
        else:
            value_states = self._value_cache[layer_idx]
            if n_rep > 1:
                value_states = _repeat_kv(value_states, n_rep)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    def get_seq_len(self, layer_idx: int) -> int:
        """Get current sequence length for a layer."""
        return self._seq_lens.get(layer_idx, 0)

    def memory_usage_bytes(self, layer_idx: int) -> dict:
        """Estimate memory usage for a layer.

        Returns:
            Dict with key_bytes, value_bytes, total_bytes, and
            compression_ratio vs fp16.
        """
        seq_len = self._seq_lens.get(layer_idx, 0)
        if seq_len == 0:
            return {"key_bytes": 0, "value_bytes": 0,
                    "total_bytes": 0, "compression_ratio": 0}

        n_heads = self.n_kv_heads

        # Key storage: bits * d * seq_len * n_heads / 8 + norms
        key_bits_total = self.key_bits * self.head_dim * seq_len * n_heads
        key_norm_bytes = seq_len * n_heads * 4  # float32 norms
        key_residual_norm_bytes = seq_len * n_heads * 4
        key_bytes = key_bits_total // 8 + key_norm_bytes + key_residual_norm_bytes

        # Value storage
        if self.value_bits is not None:
            val_bits_total = self.value_bits * self.head_dim * seq_len * n_heads
            val_norm_bytes = seq_len * n_heads * 4
            val_bytes = val_bits_total // 8 + val_norm_bytes
        else:
            val_bytes = seq_len * n_heads * self.head_dim * 2  # fp16

        total = key_bytes + val_bytes
        fp16_total = seq_len * n_heads * self.head_dim * 2 * 2  # keys + values in fp16
        ratio = fp16_total / max(total, 1)

        return {
            "key_bytes": key_bytes,
            "value_bytes": val_bytes,
            "total_bytes": total,
            "compression_ratio": ratio,
        }

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """Clear cache for a specific layer or all layers."""
        if layer_idx is not None:
            self._key_cache.pop(layer_idx, None)
            self._value_cache.pop(layer_idx, None)
            self._value_indices.pop(layer_idx, None)
            self._seq_lens.pop(layer_idx, None)
        else:
            self._key_cache.clear()
            self._value_cache.clear()
            self._value_indices.clear()
            self._seq_lens.clear()


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention.

    (batch, n_kv_heads, seq, dim) -> (batch, n_kv_heads * n_rep, seq, dim)
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, dim = x.shape
    x = x.unsqueeze(2).expand(batch, n_kv_heads, n_rep, seq_len, dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, dim)


def _repeat_kv_encoded(enc: TurboQuantEncoded, n_rep: int) -> TurboQuantEncoded:
    """Repeat encoded KV cache for GQA."""
    if n_rep == 1:
        return enc
    return TurboQuantEncoded(
        pq_indices=_repeat_kv(enc.pq_indices, n_rep),
        norms=_repeat_kv(enc.norms, n_rep),
        qjl_sign_bits=_repeat_kv(enc.qjl_sign_bits, n_rep),
        residual_norms=_repeat_kv(enc.residual_norms, n_rep),
    )
