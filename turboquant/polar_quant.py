"""PolarQuant: Random rotation + optimal scalar quantization.

Stage 1 of TurboQuant. The key insight is that after multiplying a vector by a
random orthogonal matrix, each coordinate follows a Beta distribution that is
independent of the input. This allows applying a precomputed Lloyd-Max scalar
quantizer to each coordinate independently while achieving near-optimal MSE.

The random rotation also eliminates the need for per-block normalization
constants, removing the memory overhead that plagues traditional quantizers.

Reference: Han, Kacham, Karbasi, Mirrokni, Zandieh. "PolarQuant: Quantizing
KV Caches with Polar Transformation", AISTATS 2026.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from turboquant.lloyd_max import LloydMaxQuantizer


def generate_random_rotation(d: int, device: torch.device,
                             dtype: torch.dtype = torch.float32,
                             seed: Optional[int] = None) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition.

    Uses the standard method: draw a d x d Gaussian matrix and compute
    its QR factorization. The Q factor is a uniformly random orthogonal matrix
    (Haar measure on O(d)).

    Args:
        d: Matrix dimension.
        device: Torch device.
        dtype: Floating point dtype.
        seed: Optional random seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (d, d).
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Generate on CPU for reproducibility, then move
    G = torch.randn(d, d, generator=gen, dtype=dtype)
    Q, R = torch.linalg.qr(G)
    # Ensure proper rotation (det = +1) by fixing sign convention
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_random_hadamard(d: int, device: torch.device,
                             dtype: torch.dtype = torch.float32,
                             seed: Optional[int] = None) -> torch.Tensor:
    """Generate a randomized Hadamard-like rotation matrix.

    For dimensions that are powers of 2, uses a fast Hadamard construction
    with random sign flips. Falls back to QR for other dimensions.

    This is more computationally efficient than a full QR decomposition
    and provides the same incoherence properties.

    Args:
        d: Matrix dimension.
        device: Torch device.
        dtype: Floating point dtype.
        seed: Optional random seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (d, d).
    """
    if d & (d - 1) != 0:
        # Not a power of 2, fall back to QR
        return generate_random_rotation(d, device, dtype, seed)

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Random sign-flip diagonal
    signs = torch.randint(0, 2, (d,), generator=gen, dtype=dtype) * 2 - 1

    # Build Hadamard matrix recursively
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    H = H / (d ** 0.5)  # Normalize to orthogonal

    # Apply random signs: D @ H where D = diag(signs)
    result = signs.unsqueeze(1) * H
    return result.to(device)


class PolarQuant:
    """PolarQuant: MSE-optimal vector quantizer via random rotation.

    Applies a random orthogonal rotation to input vectors, then quantizes
    each coordinate independently using a Lloyd-Max quantizer optimized for
    the resulting Beta distribution.

    This achieves near-optimal MSE distortion without any per-block
    normalization overhead.
    """

    def __init__(self, d: int, bits: int,
                 device: Optional[torch.device] = None,
                 rotation_type: str = "hadamard",
                 seed: int = 42):
        """Initialize PolarQuant.

        Args:
            d: Vector dimension.
            bits: Quantization bits per coordinate.
            device: Torch device.
            rotation_type: "hadamard" (fast) or "qr" (general).
            seed: Random seed for reproducible rotation matrix.
        """
        self.d = d
        self.bits = bits
        self.device = device or torch.device("cpu")
        self.seed = seed

        # Generate random rotation matrix (shared across all vectors)
        if rotation_type == "hadamard":
            self.rotation = generate_random_hadamard(
                d, self.device, seed=seed)
        else:
            self.rotation = generate_random_rotation(
                d, self.device, seed=seed)

        # Build Lloyd-Max quantizer for the Beta distribution
        self.quantizer = LloydMaxQuantizer(bits, d, self.device)

    def _rotate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate vectors and extract norms.

        Args:
            x: Input tensor of shape (..., d).

        Returns:
            Tuple of (rotated_normalized, norms) where:
              - rotated_normalized has shape (..., d) with values in ~[-1, 1]
              - norms has shape (..., 1)
        """
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_normalized = x / norms
        # Apply rotation: x_rot = x_normalized @ R^T
        x_rotated = x_normalized @ self.rotation.t().to(x.dtype)
        return x_rotated, norms

    def _unrotate(self, x_rotated: torch.Tensor,
                  norms: torch.Tensor) -> torch.Tensor:
        """Inverse rotation and rescale.

        Args:
            x_rotated: Rotated tensor of shape (..., d).
            norms: Original norms of shape (..., 1).

        Returns:
            Reconstructed tensor of shape (..., d).
        """
        x_reconstructed = x_rotated @ self.rotation.to(x_rotated.dtype)
        return x_reconstructed * norms

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize vectors to codebook indices.

        Args:
            x: Input vectors of shape (..., d).

        Returns:
            Tuple of (indices, norms):
              - indices: Integer tensor of shape (..., d), values in [0, 2^bits)
              - norms: Float tensor of shape (..., 1)
        """
        x_rotated, norms = self._rotate(x)
        # Clamp to valid range for the quantizer
        x_clamped = x_rotated.clamp(-1.0, 1.0)
        indices = self.quantizer.quantize(x_clamped)
        return indices, norms

    def decode(self, indices: torch.Tensor,
               norms: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from codebook indices and norms.

        Args:
            indices: Codebook indices of shape (..., d).
            norms: Vector norms of shape (..., 1).

        Returns:
            Reconstructed vectors of shape (..., d).
        """
        x_dequantized = self.quantizer.dequantize(indices)
        return self._unrotate(x_dequantized, norms)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full quantization pipeline returning indices, norms, and residual.

        Args:
            x: Input vectors of shape (..., d).

        Returns:
            Tuple of (indices, norms, residual):
              - indices: Codebook indices of shape (..., d)
              - norms: Original norms of shape (..., 1)
              - residual: Quantization error x - x_hat, shape (..., d)
        """
        x_rotated, norms = self._rotate(x)
        x_clamped = x_rotated.clamp(-1.0, 1.0)
        indices = self.quantizer.quantize(x_clamped)
        x_dequantized = self.quantizer.dequantize(indices)
        # Residual in rotated space, then transform back
        residual_rotated = x_clamped - x_dequantized
        residual = residual_rotated @ self.rotation.to(x.dtype) * norms
        return indices, norms, residual

    def to(self, device: torch.device) -> "PolarQuant":
        """Move all tensors to a new device."""
        self.device = device
        self.rotation = self.rotation.to(device)
        self.quantizer = self.quantizer.to(device)
        return self
