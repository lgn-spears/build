"""Research extensions beyond the TurboQuant paper.

Three ideas that could improve TurboQuant:

1. AdaptiveTurboQuant: Use different bit allocations per layer/head based
   on measured sensitivity. Early layers and outlier-heavy heads get more bits.

2. MultiResolutionTurboQuant: Progressive encoding that allows partial
   decoding. First b1 bits give a coarse answer, adding b2 more bits refines
   it. Useful for early-exit attention or speculative decoding.

3. HybridTurboQuant: Keep a small number of "outlier" coordinates at full
   precision (like SqueezeLLM/KVQuant), quantize the rest with TurboQuant.
   Combines the best of both worlds.
"""

from typing import Optional, Tuple, Dict, List
import torch
import math

from turboquant.turboquant import TurboQuant, TurboQuantEncoded
from turboquant.polar_quant import PolarQuant
from turboquant.qjl import QJL


# ============================================================
# Extension 1: Adaptive Bit Allocation
# ============================================================

class AdaptiveTurboQuant:
    """TurboQuant with per-head adaptive bit allocation.

    Instead of using the same bit-width everywhere, this extension
    measures the "sensitivity" of each attention head during a short
    calibration phase and allocates more bits to sensitive heads.

    The insight: not all heads are equally important. Some heads
    attend to local patterns (low sensitivity, fewer bits ok) while
    others capture long-range dependencies (high sensitivity, need
    more bits).

    This is a novel extension — the original paper uses uniform bits.
    """

    def __init__(self, d: int, n_heads: int,
                 total_bit_budget: float = 3.0,
                 min_bits: int = 2, max_bits: int = 5,
                 device: Optional[torch.device] = None):
        """Initialize adaptive quantizer.

        Args:
            d: Head dimension.
            n_heads: Number of attention heads.
            total_bit_budget: Average bits per coordinate across all heads.
            min_bits: Minimum bits for any head.
            max_bits: Maximum bits for any head.
            device: Torch device.
        """
        self.d = d
        self.n_heads = n_heads
        self.total_bit_budget = total_bit_budget
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.device = device or torch.device("cpu")

        # Initially all heads get uniform bits (before calibration)
        self.head_bits = [round(total_bit_budget)] * n_heads
        self.quantizers: Dict[int, TurboQuant] = {}
        self._build_quantizers()

    def _build_quantizers(self):
        """Build per-head quantizers based on current bit allocation."""
        self.quantizers = {}
        for bits in set(self.head_bits):
            self.quantizers[bits] = TurboQuant(
                d=self.d, bits=bits, device=self.device)

    def calibrate(self, key_samples: torch.Tensor,
                  query_samples: torch.Tensor) -> List[int]:
        """Calibrate bit allocation based on attention sensitivity.

        Measures how much quantization error affects attention scores
        for each head, then allocates bits proportionally.

        Args:
            key_samples: (n_heads, n_samples, d) representative keys.
            query_samples: (n_heads, n_samples, d) representative queries.

        Returns:
            List of bit allocations per head.
        """
        n_heads, n_samples, d = key_samples.shape
        sensitivities = []

        for h in range(n_heads):
            keys = key_samples[h]
            queries = query_samples[h]

            # True attention scores
            true_scores = queries @ keys.t() / math.sqrt(d)
            true_attn = torch.softmax(true_scores, dim=-1)

            # Measure sensitivity: how much does 2-bit quantization
            # change the attention distribution?
            tq_low = TurboQuant(d=d, bits=2, device=self.device)
            encoded = tq_low.encode(keys)
            est_scores = tq_low.estimate_inner_product(queries, encoded)
            est_scores = est_scores / math.sqrt(d)
            est_attn = torch.softmax(est_scores, dim=-1)

            # KL divergence as sensitivity measure
            kl = (true_attn * (true_attn.clamp(min=1e-8).log()
                               - est_attn.clamp(min=1e-8).log())).sum(dim=-1)
            sensitivities.append(kl.mean().item())

        # Allocate bits proportionally to sensitivity
        total_budget = self.total_bit_budget * n_heads
        sensitivities = torch.tensor(sensitivities)
        sensitivities = sensitivities / sensitivities.sum()

        # Weighted allocation
        raw_bits = sensitivities * total_budget
        # Round and clamp
        self.head_bits = [
            max(self.min_bits, min(self.max_bits, round(b.item())))
            for b in raw_bits
        ]

        # Adjust to meet budget (greedy rebalancing)
        while sum(self.head_bits) > total_budget:
            # Remove a bit from the least sensitive head that's above min
            for idx in sensitivities.argsort():
                i = idx.item()
                if self.head_bits[i] > self.min_bits:
                    self.head_bits[i] -= 1
                    break
        while sum(self.head_bits) < total_budget:
            # Add a bit to the most sensitive head that's below max
            for idx in sensitivities.argsort(descending=True):
                i = idx.item()
                if self.head_bits[i] < self.max_bits:
                    self.head_bits[i] += 1
                    break

        self._build_quantizers()
        return self.head_bits

    def encode_head(self, x: torch.Tensor,
                    head_idx: int) -> TurboQuantEncoded:
        """Encode vectors for a specific head."""
        bits = self.head_bits[head_idx]
        return self.quantizers[bits].encode(x)

    def estimate_inner_product_head(
        self, query: torch.Tensor,
        encoded_keys: TurboQuantEncoded,
        head_idx: int
    ) -> torch.Tensor:
        """Estimate inner products for a specific head."""
        bits = self.head_bits[head_idx]
        return self.quantizers[bits].estimate_inner_product(query, encoded_keys)


# ============================================================
# Extension 2: Multi-Resolution Progressive Encoding
# ============================================================

class MultiResolutionTurboQuant:
    """Progressive TurboQuant with refinable encoding.

    Encodes vectors at multiple resolutions. The coarse encoding (fewer bits)
    can be decoded quickly for early-exit attention. If higher precision is
    needed, additional refinement bits can be applied without re-encoding.

    Use case: speculative decoding where you first compute attention with
    2-bit keys to check if the draft token is likely correct, then refine
    to 4-bit only for tokens that fail verification.

    This is novel — the original paper doesn't address progressive coding.
    """

    def __init__(self, d: int,
                 coarse_bits: int = 2,
                 fine_bits: int = 4,
                 device: Optional[torch.device] = None):
        """Initialize multi-resolution quantizer.

        Args:
            d: Vector dimension.
            coarse_bits: Bits for coarse (fast) encoding.
            fine_bits: Total bits for fine (accurate) encoding.
            device: Torch device.
        """
        self.d = d
        self.coarse_bits = coarse_bits
        self.fine_bits = fine_bits
        self.device = device or torch.device("cpu")

        # Coarse quantizer
        self.coarse = TurboQuant(d=d, bits=coarse_bits, device=self.device)
        # Fine quantizer (independent, higher fidelity)
        self.fine = TurboQuant(d=d, bits=fine_bits, device=self.device)

    def encode(self, x: torch.Tensor) -> dict:
        """Encode at both resolutions.

        Returns:
            Dict with 'coarse' and 'fine' TurboQuantEncoded objects.
        """
        return {
            "coarse": self.coarse.encode(x),
            "fine": self.fine.encode(x),
        }

    def estimate_coarse(self, query: torch.Tensor,
                        encoded: dict) -> torch.Tensor:
        """Fast, lower-quality inner product estimate."""
        return self.coarse.estimate_inner_product(query, encoded["coarse"])

    def estimate_fine(self, query: torch.Tensor,
                      encoded: dict) -> torch.Tensor:
        """Slower, higher-quality inner product estimate."""
        return self.fine.estimate_inner_product(query, encoded["fine"])

    def speculative_attention(
        self, query: torch.Tensor, encoded: dict,
        value_states: torch.Tensor,
        threshold: float = 0.9,
    ) -> Tuple[torch.Tensor, bool]:
        """Speculative attention with early exit.

        Computes coarse attention first. If the top-1 token is dominant
        (weight > threshold), returns early without computing fine attention.

        Args:
            query: (batch, 1, d) single-token query.
            encoded: Dict from encode().
            value_states: (batch, seq_len, d) value vectors.
            threshold: Confidence threshold for early exit.

        Returns:
            Tuple of (attention_output, used_fine_resolution).
        """
        scale = 1.0 / math.sqrt(self.d)

        # Try coarse first
        coarse_scores = self.estimate_coarse(query, encoded) * scale
        coarse_weights = torch.softmax(coarse_scores, dim=-1)

        # Check if confident enough for early exit
        max_weight = coarse_weights.max(dim=-1).values
        if (max_weight > threshold).all():
            output = coarse_weights @ value_states
            return output, False  # Didn't need fine resolution

        # Fall back to fine resolution
        fine_scores = self.estimate_fine(query, encoded) * scale
        fine_weights = torch.softmax(fine_scores, dim=-1)
        output = fine_weights @ value_states
        return output, True  # Used fine resolution


# ============================================================
# Extension 3: Hybrid Outlier-Aware Quantization
# ============================================================

class HybridTurboQuant:
    """TurboQuant with outlier channels preserved at full precision.

    Insight from KVQuant/SqueezeLLM: a few "outlier" coordinate dimensions
    carry disproportionate information. Keeping these at full precision and
    quantizing the rest with TurboQuant can dramatically improve quality
    at the same average bit budget.

    This extension detects outlier dimensions during calibration and stores
    them separately, applying TurboQuant only to the remaining dimensions.

    Novel combination — the original paper doesn't handle outliers.
    """

    def __init__(self, d: int, bits: int = 3,
                 n_outliers: int = 8,
                 device: Optional[torch.device] = None):
        """Initialize hybrid quantizer.

        Args:
            d: Vector dimension.
            bits: Bits for TurboQuant on non-outlier dimensions.
            n_outliers: Number of dimensions to keep at full precision.
            device: Torch device.
        """
        self.d = d
        self.bits = bits
        self.n_outliers = n_outliers
        self.device = device or torch.device("cpu")

        # Will be set during calibration
        self.outlier_dims: Optional[torch.Tensor] = None
        self.normal_dims: Optional[torch.Tensor] = None
        self.quantizer: Optional[TurboQuant] = None

    def calibrate(self, key_samples: torch.Tensor) -> torch.Tensor:
        """Identify outlier dimensions from calibration data.

        Outliers are detected by finding dimensions with highest variance
        (those most sensitive to quantization error).

        Args:
            key_samples: (..., d) representative key vectors.

        Returns:
            Indices of outlier dimensions.
        """
        # Flatten to (n, d)
        flat = key_samples.reshape(-1, self.d)
        # Find dimensions with highest kurtosis (heavy-tailed = outlier-prone)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp(min=1e-8)
        normalized = (flat - mean) / std
        kurtosis = (normalized ** 4).mean(dim=0) - 3.0  # excess kurtosis

        _, outlier_idx = kurtosis.topk(self.n_outliers)
        self.outlier_dims = outlier_idx.sort().values.to(self.device)

        all_dims = torch.arange(self.d, device=self.device)
        mask = torch.ones(self.d, dtype=torch.bool, device=self.device)
        mask[self.outlier_dims] = False
        self.normal_dims = all_dims[mask]

        # Build quantizer for non-outlier dimensions
        d_normal = self.d - self.n_outliers
        self.quantizer = TurboQuant(d=d_normal, bits=self.bits, device=self.device)

        return self.outlier_dims

    def encode(self, x: torch.Tensor) -> dict:
        """Encode vectors, separating outlier dimensions.

        Args:
            x: Input vectors (..., d).

        Returns:
            Dict with 'normal_encoded', 'outlier_values', and 'outlier_dims'.
        """
        if self.outlier_dims is None:
            raise RuntimeError("Must call calibrate() before encode()")

        outlier_vals = x[..., self.outlier_dims]  # Full precision
        normal_vals = x[..., self.normal_dims]    # To be quantized
        normal_encoded = self.quantizer.encode(normal_vals)

        return {
            "normal_encoded": normal_encoded,
            "outlier_values": outlier_vals,
            "outlier_dims": self.outlier_dims,
            "normal_dims": self.normal_dims,
        }

    def estimate_inner_product(self, query: torch.Tensor,
                               encoded: dict) -> torch.Tensor:
        """Estimate <query, key> using hybrid approach.

        Exact inner product for outlier dims + TurboQuant estimate for rest.
        """
        q_outlier = query[..., encoded["outlier_dims"]]
        q_normal = query[..., encoded["normal_dims"]]

        # Exact inner product for outlier dimensions
        ip_outlier = torch.matmul(
            q_outlier, encoded["outlier_values"].transpose(-2, -1))

        # TurboQuant estimate for normal dimensions
        ip_normal = self.quantizer.estimate_inner_product(
            q_normal, encoded["normal_encoded"])

        return ip_outlier + ip_normal

    @property
    def effective_bits(self) -> float:
        """Average bits per coordinate including outlier overhead."""
        outlier_bits = self.n_outliers * 32  # FP32 for outliers
        normal_bits = (self.d - self.n_outliers) * self.bits
        return (outlier_bits + normal_bits) / self.d
