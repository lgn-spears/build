"""Lloyd-Max optimal scalar quantizer for Beta-distributed coordinates.

After random rotation of a unit-norm vector in R^d, each coordinate follows
a Beta distribution concentrated around 0. The Lloyd-Max algorithm finds
optimal centroids (codebook) that minimize MSE for this distribution.

The codebook is precomputed once and reused for all subsequent quantizations.
"""

import math
from functools import lru_cache
from typing import Optional

import torch
import numpy as np
from scipy import integrate
from scipy.stats import beta as beta_dist


def _beta_pdf(x: float, d: int) -> float:
    """PDF of a single coordinate of a uniformly random unit vector in R^d.

    After applying a random rotation to a vector on the unit sphere S^{d-1},
    each coordinate z_i = (Rx)_i / ||x|| follows a distribution with PDF
    proportional to (1 - z^2)^{(d-3)/2} on [-1, 1].

    This is a scaled Beta((d-1)/2, (d-1)/2) distribution shifted to [-1, 1].
    """
    alpha = (d - 1) / 2.0
    # Transform from [-1, 1] to [0, 1]: t = (x + 1) / 2
    t = (x + 1.0) / 2.0
    if t <= 0 or t >= 1:
        return 0.0
    return beta_dist.pdf(t, alpha, alpha) / 2.0


def _beta_cdf(x: float, d: int) -> float:
    """CDF of the coordinate distribution."""
    alpha = (d - 1) / 2.0
    t = (x + 1.0) / 2.0
    t = max(0.0, min(1.0, t))
    return beta_dist.cdf(t, alpha, alpha)


def compute_beta_codebook(bits: int, d: int, max_iters: int = 200,
                          tol: float = 1e-10) -> np.ndarray:
    """Compute optimal Lloyd-Max codebook for Beta-distributed coordinates.

    Solves the continuous 1-D k-means problem on [-1, 1] with the Beta PDF
    that arises from random rotation in d dimensions.

    Args:
        bits: Number of quantization bits (codebook size = 2^bits).
        d: Dimension of the vectors (determines Beta distribution shape).
        max_iters: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        Sorted array of 2^bits centroids in [-1, 1].
    """
    n_levels = 2 ** bits
    alpha = (d - 1) / 2.0

    # Initialize centroids uniformly spaced using quantiles of the distribution
    quantiles = np.linspace(1.0 / (2 * n_levels), 1.0 - 1.0 / (2 * n_levels), n_levels)
    centroids = np.array([
        beta_dist.ppf(q, alpha, alpha) * 2.0 - 1.0 for q in quantiles
    ])

    for _ in range(max_iters):
        # Compute Voronoi boundaries (midpoints between consecutive centroids)
        boundaries = np.empty(n_levels + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_levels - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        # Update centroids to conditional expectations within each Voronoi cell
        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            # E[X | lo <= X <= hi] = integral(x * pdf(x)) / integral(pdf(x))
            num, _ = integrate.quad(lambda x: x * _beta_pdf(x, d), lo, hi)
            den, _ = integrate.quad(lambda x: _beta_pdf(x, d), lo, hi)
            if den > 1e-15:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = (lo + hi) / 2.0

        # Check convergence
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < tol:
            break

    return np.sort(centroids)


class LloydMaxQuantizer:
    """Precomputed Lloyd-Max scalar quantizer for Beta-distributed coordinates.

    Stores optimal codebooks for given bit-widths and dimensions, enabling
    fast quantization and dequantization of rotated vector coordinates.
    """

    def __init__(self, bits: int, d: int, device: Optional[torch.device] = None):
        """Initialize the quantizer with a precomputed codebook.

        Args:
            bits: Number of quantization bits per coordinate.
            d: Vector dimension (determines the Beta distribution shape).
            device: Torch device for the codebook tensors.
        """
        self.bits = bits
        self.d = d
        self.n_levels = 2 ** bits
        self.device = device or torch.device("cpu")

        # Compute optimal codebook via Lloyd-Max algorithm
        codebook_np = compute_beta_codebook(bits, d)
        self.codebook = torch.from_numpy(codebook_np).float().to(self.device)

        # Precompute boundaries for fast quantization
        boundaries_np = np.empty(self.n_levels - 1)
        for i in range(self.n_levels - 1):
            boundaries_np[i] = (codebook_np[i] + codebook_np[i + 1]) / 2.0
        self.boundaries = torch.from_numpy(boundaries_np).float().to(self.device)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map continuous values to quantization indices.

        Args:
            x: Tensor of coordinate values in [-1, 1].

        Returns:
            Integer tensor of codebook indices (same shape as x).
        """
        # Use searchsorted on boundaries for O(log n) quantization
        indices = torch.searchsorted(self.boundaries, x.contiguous())
        return indices.clamp(0, self.n_levels - 1).to(torch.int16)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Map quantization indices back to centroid values.

        Args:
            indices: Integer tensor of codebook indices.

        Returns:
            Tensor of centroid values (same shape as indices).
        """
        return self.codebook[indices.long()]

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and immediately dequantize (for computing residuals)."""
        return self.dequantize(self.quantize(x))

    def to(self, device: torch.device) -> "LloydMaxQuantizer":
        """Move quantizer tensors to a new device."""
        self.device = device
        self.codebook = self.codebook.to(device)
        self.boundaries = self.boundaries.to(device)
        return self
