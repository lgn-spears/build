"""QJL: Quantized Johnson-Lindenstrauss transform for 1-bit residual correction.

Stage 2 of TurboQuant. QJL applies a Johnson-Lindenstrauss random projection
followed by sign-bit quantization. The key property is that this provides an
*unbiased* estimator of inner products, correcting the bias introduced by
MSE-optimal quantizers.

From the paper (Definition 1):
  Q_qjl(x) = sign(S · x)           where S has i.i.d. N(0,1) entries
  Q⁻¹_qjl(z) = √(π/2) / d · S^T · z

The asymmetric estimator satisfies E[<y, Q⁻¹(Q(x))>] = <y, x> for unit x.
For non-unit vectors, we store ||x|| and scale accordingly.

Reference: Zandieh, Daliri, Han. "QJL: 1-Bit Quantized JL Transform for KV
Cache Quantization with Zero Overhead", AAAI 2025.
"""

from typing import Optional, Tuple

import torch
import math


def generate_gaussian_matrix(d: int, m: int, device: torch.device,
                             dtype: torch.dtype = torch.float32,
                             seed: Optional[int] = None) -> torch.Tensor:
    """Generate a Gaussian random projection matrix S with i.i.d. N(0,1) entries.

    This is the S matrix from the paper's Definition 1. NOT scaled by 1/sqrt(m).

    Args:
        d: Input dimension.
        m: Projection dimension (number of sign bits to store).
        device: Torch device.
        dtype: Floating point type.
        seed: Optional random seed.

    Returns:
        Gaussian matrix of shape (m, d) with N(0,1) entries.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    S = torch.randn(m, d, generator=gen, dtype=dtype)
    return S.to(device)


class QJL:
    """Quantized Johnson-Lindenstrauss transform for 1-bit vector compression.

    Projects vectors using a random Gaussian matrix S and stores only sign bits.
    The dequantization map is Q⁻¹(z) = √(π/2) / m · S^T · z, which gives an
    unbiased inner product estimator when one vector (query) is kept at full
    precision and the other (key) is sign-quantized.

    For unit vector x and any y:
        E[<y, Q⁻¹(Q(x))>] = <y, x>
        Var[<y, Q⁻¹(Q(x))>] <= (π/2) · ||y||² / m
    """

    def __init__(self, d: int, m: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 seed: int = 137):
        """Initialize QJL.

        Args:
            d: Input vector dimension.
            m: Projection dimension (number of sign bits per vector).
                Defaults to d (used as residual stage in TurboQuant).
            device: Torch device.
            seed: Random seed for reproducible projection matrix.
        """
        self.d = d
        self.m = m if m is not None else d
        self.device = device or torch.device("cpu")
        self.seed = seed

        # Generate Gaussian projection matrix S with i.i.d. N(0,1) entries
        self.S = generate_gaussian_matrix(
            d, self.m, self.device, seed=seed)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode vectors to sign bits.

        Q_qjl(x) = sign(S · x)

        We also store ||x|| to handle non-unit vectors.

        Args:
            x: Input vectors of shape (..., d).

        Returns:
            Tuple of (sign_bits, norms):
              - sign_bits: Boolean tensor of shape (..., m), True = positive
              - norms: Float tensor of shape (..., 1), the L2 norms
        """
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # Project: z = x @ S^T = S·x for each vector, shape (..., m)
        projected = x @ self.S.t().to(x.dtype)
        # Sign quantization
        sign_bits = projected > 0
        return sign_bits, norms

    def decode_for_inner_product(
        self, query: torch.Tensor,
        sign_bits: torch.Tensor,
        key_norms: torch.Tensor
    ) -> torch.Tensor:
        """Compute asymmetric inner product estimate: <query, key>.

        For a key k with norm γ = ||k||, the estimator is:
            <q, k> ≈ γ · √(π/2) / m · Σ_i (s_i · q) · sign(s_i · k)

        This is unbiased: E[estimator] = <q, k>.

        Args:
            query: Full-precision query vectors (..., n_queries, d).
            sign_bits: Encoded key sign bits (..., n_keys, m).
            key_norms: Key norms (..., n_keys, 1).

        Returns:
            Inner product estimates of shape (..., n_queries, n_keys).
        """
        # Project query at full precision: q_proj = S · q, shape (..., n_queries, m)
        q_projected = query @ self.S.t().to(query.dtype)

        # Convert sign bits to +1/-1
        signs = sign_bits.float() * 2 - 1  # (..., n_keys, m)

        # Asymmetric dot product: sum over projection dimension
        # q_projected: (..., n_queries, m)
        # signs: (..., n_keys, m)
        # Result: (..., n_queries, n_keys)
        raw_scores = torch.matmul(
            q_projected,  # (..., n_queries, m)
            signs.transpose(-2, -1)  # (..., m, n_keys)
        )

        # Apply the √(π/2) / m scaling factor from the dequantization formula
        scale = math.sqrt(math.pi / 2.0) / self.m
        scores = raw_scores * scale

        # Apply key norm correction for non-unit vectors
        # key_norms shape: (..., n_keys, 1) -> (..., 1, n_keys)
        norm_factor = key_norms.squeeze(-1).unsqueeze(-2)
        scores = scores * norm_factor

        return scores

    def encode_pack(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode vectors to packed sign bits (8 bits per byte).

        More memory-efficient storage for large-scale use.

        Args:
            x: Input vectors of shape (..., d).

        Returns:
            Tuple of (packed_bits, norms):
              - packed_bits: uint8 tensor of shape (..., ceil(m/8))
              - norms: Float tensor of shape (..., 1)
        """
        sign_bits, norms = self.encode(x)
        packed = _pack_bits(sign_bits)
        return packed, norms

    def to(self, device: torch.device) -> "QJL":
        """Move all tensors to a new device."""
        self.device = device
        self.S = self.S.to(device)
        return self


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack boolean tensor into uint8 bytes along the last dimension.

    Args:
        bits: Boolean tensor of shape (..., m).

    Returns:
        uint8 tensor of shape (..., ceil(m/8)).
    """
    *batch_shape, m = bits.shape
    # Pad to multiple of 8
    pad_size = (8 - m % 8) % 8
    if pad_size > 0:
        bits = torch.nn.functional.pad(bits, (0, pad_size), value=False)

    bits_reshaped = bits.reshape(*batch_shape, -1, 8)
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                          dtype=torch.uint8, device=bits.device)
    packed = (bits_reshaped.byte() * powers).sum(dim=-1).byte()
    return packed


def _unpack_bits(packed: torch.Tensor, m: int) -> torch.Tensor:
    """Unpack uint8 bytes back to boolean tensor.

    Args:
        packed: uint8 tensor of shape (..., ceil(m/8)).
        m: Original number of bits.

    Returns:
        Boolean tensor of shape (..., m).
    """
    *batch_shape, n_bytes = packed.shape
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                          dtype=torch.uint8, device=packed.device)
    unpacked = (packed.unsqueeze(-1) & powers) > 0
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :m]
