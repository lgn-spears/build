"""TurboQuant: Two-stage vector quantizer with near-optimal distortion rate.

Combines PolarQuant (MSE-optimal scalar quantization after random rotation)
with QJL (1-bit residual correction) to achieve both:
  - Near-optimal MSE distortion (from PolarQuant)
  - Unbiased inner product estimation (from QJL residual correction)

The bit budget is split as: (b-1) bits for PolarQuant + 1 bit for QJL,
where b is the total bits per coordinate.

Reference: Zandieh, Kacham, Han, Daliri, Gottesbüren, Jayaram, Mirrokni.
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
ICLR 2026.
"""

from typing import Optional, Tuple, NamedTuple

import torch
import math

from turboquant.polar_quant import PolarQuant
from turboquant.qjl import QJL


class TurboQuantEncoded(NamedTuple):
    """Encoded representation from TurboQuant.

    Attributes:
        pq_indices: PolarQuant codebook indices, shape (..., d), int dtype.
        norms: Original vector norms, shape (..., 1), float.
        qjl_sign_bits: QJL sign bits for residual, shape (..., m), bool.
        residual_norms: Norms of the PolarQuant residuals, shape (..., 1), float.
    """
    pq_indices: torch.Tensor
    norms: torch.Tensor
    qjl_sign_bits: torch.Tensor
    residual_norms: torch.Tensor


class TurboQuant:
    """TurboQuant: Near-optimal online vector quantizer.

    Two-stage algorithm:
      Stage 1 (PolarQuant): Uses (bits - 1) bits per coordinate for MSE-optimal
          quantization via random rotation + Lloyd-Max scalar quantization.
      Stage 2 (QJL): Uses 1 bit per coordinate to sign-quantize the JL
          projection of the residual, providing unbiased inner product estimation.

    Total storage per vector: (bits - 1) * d bits for PolarQuant indices
                             + 1 * d bits for QJL sign bits
                             + overhead for norms (negligible, amortized)
                             = bits * d bits total

    The near-optimality guarantee states that TurboQuant's distortion is within
    a constant factor (~2.7) of the information-theoretic lower bound.
    """

    def __init__(self, d: int, bits: int = 3,
                 device: Optional[torch.device] = None,
                 rotation_type: str = "hadamard",
                 pq_seed: int = 42,
                 qjl_seed: int = 137,
                 qjl_projection_dim: Optional[int] = None):
        """Initialize TurboQuant.

        Args:
            d: Vector dimension.
            bits: Total bits per coordinate (>= 2). Split as (bits-1) for
                PolarQuant + 1 for QJL.
            device: Torch device.
            rotation_type: Rotation matrix type for PolarQuant
                ("hadamard" or "qr").
            pq_seed: Random seed for PolarQuant rotation.
            qjl_seed: Random seed for QJL projection.
            qjl_projection_dim: QJL projection dimension. Defaults to d.
        """
        if bits < 2:
            raise ValueError("TurboQuant requires at least 2 bits "
                             "(1 for PolarQuant + 1 for QJL)")

        self.d = d
        self.bits = bits
        self.device = device or torch.device("cpu")

        # Stage 1: PolarQuant with (bits - 1) bits per coordinate
        self.pq_bits = bits - 1
        self.polar_quant = PolarQuant(
            d=d, bits=self.pq_bits, device=self.device,
            rotation_type=rotation_type, seed=pq_seed)

        # Stage 2: QJL with 1 bit per projected dimension
        self.qjl_dim = qjl_projection_dim if qjl_projection_dim else d
        self.qjl = QJL(d=d, m=self.qjl_dim, device=self.device, seed=qjl_seed)

    def encode(self, x: torch.Tensor) -> TurboQuantEncoded:
        """Encode vectors using the two-stage TurboQuant pipeline.

        Stage 1: Apply PolarQuant to get MSE-optimal quantized representation.
        Stage 2: Compute residual and encode it with QJL sign bits.

        Args:
            x: Input vectors of shape (..., d).

        Returns:
            TurboQuantEncoded containing indices, norms, sign bits,
            and residual norms.
        """
        # Stage 1: PolarQuant quantization
        pq_indices, norms, residual = self.polar_quant.quantize(x)

        # Stage 2: QJL on the residual
        qjl_sign_bits, residual_norms = self.qjl.encode(residual)

        return TurboQuantEncoded(
            pq_indices=pq_indices,
            norms=norms,
            qjl_sign_bits=qjl_sign_bits,
            residual_norms=residual_norms,
        )

    def decode(self, encoded: TurboQuantEncoded) -> torch.Tensor:
        """Decode (reconstruct) vectors from TurboQuant encoding.

        Note: This gives the PolarQuant reconstruction only. For inner product
        estimation, use `estimate_inner_product` which incorporates the QJL
        correction for unbiased results.

        Args:
            encoded: TurboQuantEncoded from encode().

        Returns:
            Reconstructed vectors of shape (..., d).
        """
        return self.polar_quant.decode(encoded.pq_indices, encoded.norms)

    def estimate_inner_product(
        self, query: torch.Tensor,
        encoded_keys: TurboQuantEncoded
    ) -> torch.Tensor:
        """Estimate inner products <query, key> using both stages.

        Combines the PolarQuant reconstruction with the QJL residual correction
        to produce an unbiased inner product estimator:

            <q, k> ≈ <q, k_pq> + QJL_estimate(<q, residual>)

        where k_pq is the PolarQuant reconstruction and residual = k - k_pq.

        Args:
            query: Query vectors of shape (..., n_queries, d).
            encoded_keys: Encoded key vectors from encode().

        Returns:
            Inner product estimates of shape (..., n_queries, n_keys).
        """
        # Term 1: Inner product with PolarQuant reconstruction
        keys_reconstructed = self.decode(encoded_keys)
        pq_scores = torch.matmul(
            query, keys_reconstructed.transpose(-2, -1))

        # Term 2: QJL correction for the residual
        qjl_scores = self.qjl.decode_for_inner_product(
            query=query,
            sign_bits=encoded_keys.qjl_sign_bits,
            key_norms=encoded_keys.residual_norms,
        )

        return pq_scores + qjl_scores

    def compute_attention_scores(
        self, query: torch.Tensor,
        encoded_keys: TurboQuantEncoded,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """Compute scaled attention scores for transformer attention.

        Equivalent to softmax(Q @ K^T / sqrt(d_k)) but using quantized keys.

        Args:
            query: Query tensor of shape (batch, n_heads, seq_q, d).
            encoded_keys: Encoded key tensor.
            scale: Attention scale factor. Defaults to 1/sqrt(d).

        Returns:
            Attention weights of shape (batch, n_heads, seq_q, seq_k).
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.d)

        scores = self.estimate_inner_product(query, encoded_keys)
        scores = scores * scale
        return torch.softmax(scores, dim=-1)

    def quantize_and_score(
        self, query: torch.Tensor,
        key: torch.Tensor,
        scale: Optional[float] = None
    ) -> Tuple[torch.Tensor, TurboQuantEncoded]:
        """Convenience: quantize keys and compute attention scores in one call.

        Args:
            query: Query tensor (..., seq_q, d).
            key: Key tensor (..., seq_k, d).
            scale: Attention scale factor.

        Returns:
            Tuple of (attention_weights, encoded_keys).
        """
        encoded_keys = self.encode(key)
        attn_weights = self.compute_attention_scores(
            query, encoded_keys, scale)
        return attn_weights, encoded_keys

    @property
    def bits_per_coordinate(self) -> int:
        """Total bits stored per coordinate."""
        return self.bits

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs 32-bit float."""
        # Account for norm overhead (1 float per vector, amortized over d)
        effective_bits = self.bits + 32.0 / self.d  # norm overhead
        return 32.0 / effective_bits

    def to(self, device: torch.device) -> "TurboQuant":
        """Move all tensors to a new device."""
        self.device = device
        self.polar_quant = self.polar_quant.to(device)
        self.qjl = self.qjl.to(device)
        return self
