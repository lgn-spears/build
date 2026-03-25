"""TurboQuant MLX Backend — Apple Silicon native implementation.

Ports the full TurboQuant two-stage vector quantization pipeline to MLX,
Apple's ML framework for M-series chips. Uses unified memory for zero-copy
GPU/CPU sharing, which is exactly what makes large models fit on a MacBook.

Usage:
    import mlx.core as mx
    from turboquant.mlx_backend import TurboQuant

    tq = TurboQuant(d=128, bits=4)
    encoded = tq.encode(keys)                    # compress
    scores = tq.estimate_inner_product(q, encoded)  # compute on compressed
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    raise ImportError(
        "MLX is required for the MLX backend. Install with: pip install mlx"
    )

import numpy as np
from scipy import integrate
from scipy.stats import beta as beta_dist


# ===================================================================
#  Lloyd-Max Quantizer
# ===================================================================

def compute_beta_codebook(
    bits: int,
    d: int,
    max_iters: int = 200,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal Lloyd-Max codebook for Beta((d-1)/2, (d-1)/2) on [-1, 1].

    Returns (centroids, boundaries) as numpy arrays.
    """
    n_levels = 2 ** bits
    a = (d - 1) / 2.0
    b = a  # symmetric Beta

    # Beta distribution mapped to [-1, 1]: x = 2*u - 1 where u ~ Beta(a, b)
    def pdf(x):
        u = (x + 1) / 2
        return beta_dist.pdf(u, a, b) / 2

    # Initialize with uniform quantiles
    quantiles = np.linspace(0, 1, n_levels + 1)
    boundaries = 2 * beta_dist.ppf(quantiles[1:-1], a, b) - 1

    for _ in range(max_iters):
        # Compute centroids as conditional expectations
        centroids = np.zeros(n_levels)
        edges = np.concatenate([[-1.0], boundaries, [1.0]])

        for i in range(n_levels):
            lo, hi = edges[i], edges[i + 1]
            if hi - lo < 1e-15:
                centroids[i] = (lo + hi) / 2
                continue
            num, _ = integrate.quad(lambda x: x * pdf(x), lo, hi)
            den, _ = integrate.quad(pdf, lo, hi)
            centroids[i] = num / den if den > 1e-15 else (lo + hi) / 2

        # Update boundaries as midpoints of adjacent centroids
        new_boundaries = (centroids[:-1] + centroids[1:]) / 2

        if np.max(np.abs(new_boundaries - boundaries)) < tol:
            boundaries = new_boundaries
            break
        boundaries = new_boundaries

    return centroids.astype(np.float32), boundaries.astype(np.float32)


class LloydMaxQuantizer:
    """MLX-native Lloyd-Max scalar quantizer with precomputed codebook."""

    def __init__(self, bits: int, d: int):
        self.bits = bits
        self.d = d
        centroids_np, boundaries_np = compute_beta_codebook(bits, d)
        self.centroids = mx.array(centroids_np)
        self.boundaries = mx.array(boundaries_np)

    def quantize(self, x: mx.array) -> mx.array:
        """Map values to nearest codebook indices. O(n * levels) for MLX."""
        # MLX doesn't have searchsorted, but codebooks are tiny (max 128 entries)
        # so broadcasting comparison is fast
        # x: (...,), boundaries: (n_levels - 1,)
        expanded = mx.expand_dims(x, axis=-1)  # (..., 1)
        # Count how many boundaries each value exceeds
        indices = mx.sum(expanded > self.boundaries, axis=-1)
        return indices.astype(mx.int32)

    def dequantize(self, indices: mx.array) -> mx.array:
        """Map indices back to centroid values."""
        return self.centroids[indices]

    def quantize_dequantize(self, x: mx.array) -> mx.array:
        """Quantize then dequantize (for computing residuals)."""
        return self.dequantize(self.quantize(x))


# ===================================================================
#  PolarQuant — Random Rotation + Optimal Scalar Quantization
# ===================================================================

def generate_random_rotation(d: int, seed: int = 42) -> mx.array:
    """Generate random orthogonal matrix via QR decomposition.

    Uses numpy for QR (MLX doesn't support QR on Metal GPU),
    then converts to MLX array. This only runs once at init time.
    """
    rng = np.random.RandomState(seed)
    G = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(G)
    # Ensure proper rotation (det = +1)
    diag_sign = np.sign(np.diag(R))
    Q = Q * diag_sign
    return mx.array(Q)


class PolarQuant:
    """Stage 1: Random rotation → per-coordinate Lloyd-Max quantization."""

    def __init__(self, d: int, bits: int, seed: int = 42):
        self.d = d
        self.bits = bits
        self.R = generate_random_rotation(d, seed=seed)
        self.quantizer = LloydMaxQuantizer(bits, d)

    def _rotate(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Normalize and rotate vectors."""
        norms = mx.linalg.norm(x, axis=-1, keepdims=True)
        norms = mx.maximum(norms, mx.array(1e-8))
        x_normalized = x / norms
        x_rotated = x_normalized @ self.R.T
        x_rotated = mx.clip(x_rotated, -1.0, 1.0)
        return x_rotated, norms

    def _unrotate(self, x_rotated: mx.array, norms: mx.array) -> mx.array:
        """Inverse rotation and rescaling."""
        return (x_rotated @ self.R) * norms

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Encode vectors to (indices, norms)."""
        x_rotated, norms = self._rotate(x)
        indices = self.quantizer.quantize(x_rotated)
        return indices, norms

    def decode(self, indices: mx.array, norms: mx.array) -> mx.array:
        """Reconstruct vectors from indices and norms."""
        x_rotated = self.quantizer.dequantize(indices)
        return self._unrotate(x_rotated, norms)

    def quantize(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Full quantization returning (indices, norms, residual)."""
        x_rotated, norms = self._rotate(x)
        x_quantized = self.quantizer.quantize_dequantize(x_rotated)
        residual = self._unrotate(x_rotated - x_quantized, norms)
        indices = self.quantizer.quantize(x_rotated)
        return indices, norms, residual


# ===================================================================
#  QJL — Quantized Johnson-Lindenstrauss Transform
# ===================================================================

def _pack_bits(bits: mx.array) -> mx.array:
    """Pack boolean tensor into uint8 bytes."""
    flat = bits.reshape(-1)
    # Pad to multiple of 8
    remainder = flat.shape[0] % 8
    if remainder != 0:
        pad_size = 8 - remainder
        flat = mx.concatenate([flat, mx.zeros((pad_size,), dtype=flat.dtype)])
    flat = flat.reshape(-1, 8)
    powers = mx.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=mx.uint8)
    packed = mx.sum(flat.astype(mx.uint8) * powers, axis=-1)
    return packed


def _unpack_bits(packed: mx.array, m: int) -> mx.array:
    """Unpack uint8 bytes back to boolean tensor."""
    expanded = mx.expand_dims(packed, axis=-1)
    powers = mx.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=mx.uint8)
    bits = (expanded & powers) > 0
    return bits.reshape(-1)[:m]


class QJL:
    """Stage 2: 1-bit sign quantization of residuals via Gaussian projection.

    Uses the unbiased inner product estimator from Definition 1 of the paper:
        <x, y> ≈ (√(π/2) / m) · sign(Sx)^T · Sy · ||y||
    """

    def __init__(self, d: int, m: Optional[int] = None, seed: int = 123):
        self.d = d
        self.m = m or d
        key = mx.random.key(seed)
        self.S = mx.random.normal(shape=(self.m, d), key=key)
        mx.eval(self.S)

    def encode(self, x: mx.array) -> mx.array:
        """Project and sign-quantize: returns boolean tensor."""
        projected = x @ self.S.T  # (..., m)
        return projected > 0

    def encode_pack(self, x: mx.array) -> mx.array:
        """Memory-efficient packed bit encoding."""
        sign_bits = self.encode(x)
        original_shape = sign_bits.shape
        flat = sign_bits.reshape(-1)
        packed = _pack_bits(flat)
        return packed

    def decode_for_inner_product(
        self,
        query: mx.array,
        sign_bits: mx.array,
        key_norms: mx.array,
    ) -> mx.array:
        """Asymmetric inner product estimator (Definition 1).

        Args:
            query: (..., d) query vectors
            sign_bits: (..., m) boolean sign bits of projected keys
            key_norms: (..., 1) norms of original key vectors

        Returns:
            Estimated inner products.
        """
        # Project query through same Gaussian matrix
        q_proj = query @ self.S.T  # (..., m)

        # Convert sign bits to ±1
        signs = sign_bits.astype(mx.float32) * 2 - 1  # (..., m)

        # Raw dot product of projected query with sign bits
        if q_proj.ndim == 1 and signs.ndim == 2:
            raw_scores = q_proj @ signs.T
        elif q_proj.ndim == signs.ndim:
            raw_scores = mx.sum(q_proj * signs, axis=-1, keepdims=True)
        else:
            raw_scores = mx.matmul(q_proj, mx.swapaxes(signs, -2, -1))

        # Apply the unbiased scaling factor: √(π/2) / m
        scale = math.sqrt(math.pi / 2.0) / self.m

        scores = raw_scores * scale
        # Scale by key norms
        if key_norms.ndim < scores.ndim:
            key_norms = mx.swapaxes(key_norms, -2, -1)
        scores = scores * key_norms

        return scores


# ===================================================================
#  TurboQuant — Two-Stage Vector Quantizer
# ===================================================================

class TurboQuantEncoded(NamedTuple):
    """Compressed representation of a set of vectors."""
    pq_indices: mx.array      # (n, d) int32 — PolarQuant codebook indices
    norms: mx.array            # (n, 1) float32 — original vector norms
    qjl_sign_bits: mx.array    # (n, m) bool — QJL sign bits of residual
    residual_norms: mx.array   # (n, 1) float32 — norms of residual vectors


class TurboQuant:
    """Two-stage vector quantizer: PolarQuant + QJL.

    Stage 1 (PolarQuant): (b-1) bits per coordinate for MSE-optimal compression.
    Stage 2 (QJL): 1 bit per coordinate for unbiased inner product correction.
    Total: b bits per coordinate.
    """

    def __init__(
        self,
        d: int,
        bits: int = 4,
        qjl_dim: Optional[int] = None,
        pq_seed: int = 42,
        qjl_seed: int = 123,
    ):
        assert bits >= 2, "Need at least 2 bits (1 for PolarQuant + 1 for QJL)"
        self.d = d
        self.bits = bits
        self.polar_quant = PolarQuant(d, bits=bits - 1, seed=pq_seed)
        self.qjl = QJL(d, m=qjl_dim or d, seed=qjl_seed)

    @property
    def bits_per_coordinate(self) -> float:
        pq_bits = self.bits - 1
        qjl_bits = 1
        return pq_bits + qjl_bits

    @property
    def compression_ratio(self) -> float:
        return 16.0 / self.bits_per_coordinate  # vs FP16

    def encode(self, x: mx.array) -> TurboQuantEncoded:
        """Encode vectors using both stages."""
        # Stage 1: PolarQuant
        pq_indices, norms, residual = self.polar_quant.quantize(x)

        # Stage 2: QJL on residual
        residual_norms = mx.linalg.norm(residual, axis=-1, keepdims=True)
        residual_norms = mx.maximum(residual_norms, mx.array(1e-8))
        qjl_sign_bits = self.qjl.encode(residual)

        return TurboQuantEncoded(
            pq_indices=pq_indices,
            norms=norms,
            qjl_sign_bits=qjl_sign_bits,
            residual_norms=residual_norms,
        )

    def decode(self, encoded: TurboQuantEncoded) -> mx.array:
        """Reconstruct vectors (Stage 1 only — QJL is asymmetric)."""
        return self.polar_quant.decode(encoded.pq_indices, encoded.norms)

    def estimate_inner_product(
        self,
        query: mx.array,
        encoded_keys: TurboQuantEncoded,
    ) -> mx.array:
        """Unbiased inner product estimation using both stages.

        This is the key innovation: combining PolarQuant's reconstruction
        with QJL's unbiased residual correction.
        """
        # Stage 1: PolarQuant reconstruction inner product
        keys_pq = self.polar_quant.decode(
            encoded_keys.pq_indices, encoded_keys.norms
        )
        pq_scores = mx.matmul(query, mx.swapaxes(keys_pq, -2, -1))

        # Stage 2: QJL residual correction
        qjl_correction = self.qjl.decode_for_inner_product(
            query, encoded_keys.qjl_sign_bits, encoded_keys.residual_norms
        )

        return pq_scores + qjl_correction

    def compute_attention_scores(
        self,
        query: mx.array,
        encoded_keys: TurboQuantEncoded,
        scale: Optional[float] = None,
    ) -> mx.array:
        """Compute scaled softmax attention scores on compressed keys."""
        if scale is None:
            scale = 1.0 / math.sqrt(self.d)
        raw_scores = self.estimate_inner_product(query, encoded_keys)
        return mx.softmax(raw_scores * scale, axis=-1)
