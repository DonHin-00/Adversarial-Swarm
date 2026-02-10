import torch
import torch.nn.functional as F
from typing import Optional

class CompositeReward:
    """
    Computes a weighted sum of different reward components for HIVE-ZERO.

    Components:
        - Adversarial (Evasion)
        - Information Gain (Graph Entropy Reduction)
        - Stealth (Traffic Mimicry)
    """
    def __init__(self, w_adv: float = 1.0, w_info: float = 0.5, w_stealth: float = 0.8):
        self.w_adv = w_adv
        self.w_info = w_info
        self.w_stealth = w_stealth

    def _renormalize_distribution(self, dist: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Renormalize a probability distribution with safety checks.

        Args:
            dist: Input probability distribution tensor (assumed to be non-negative)
            epsilon: Small value to prevent division by zero

        Returns:
            Normalized distribution

        Note:
            This function assumes dist contains non-negative values. Negative values
            will be clamped to epsilon, which may not reflect the intended distribution.
            
            Two-step clamping strategy:
            1. Clamp individual values to prevent log(0) in KL divergence
            2. Clamp sum to prevent division by zero when normalizing
        """
        # Step 1: Clamp values to prevent log(0)
        dist_clamped = torch.clamp(dist, min=epsilon)
        # Step 2: Normalize with safe division
        return dist_clamped / torch.clamp(dist_clamped.sum(dim=-1, keepdim=True), min=epsilon)

    def calculate_adversarial_reward(self, sentinel_score: torch.Tensor) -> torch.Tensor:
        """
        R_adv: Maximize probability of evasion.

        Args:
            sentinel_score: P(Allowed) from SentinelAgent.
        """
        # Hardening: Clip to [0, 1] range
        return torch.clamp(sentinel_score, 0.0, 1.0)

    def calculate_info_gain_reward(self, prev_entropy: float, current_entropy: float) -> float:
        """
        R_info: Reduction in entropy of Knowledge Graph.

        Args:
            prev_entropy: Entropy before reconnaissance.
            current_entropy: Entropy after reconnaissance.
        """
        gain = prev_entropy - current_entropy
        return float(max(0.0, gain))

    def calculate_stealth_reward(self, traffic_dist: torch.Tensor, baseline_dist: torch.Tensor) -> torch.Tensor:
        """
        R_stealth: Minimize KL Divergence between generated traffic and baseline.

        Args:
            traffic_dist: Probability distribution of generated traffic.
            baseline_dist: Probability distribution of normal traffic.
        """
        if traffic_dist.shape != baseline_dist.shape:
            return torch.tensor(0.0, device=traffic_dist.device)

        # Renormalize distributions with safety checks
        traffic_dist_norm = self._renormalize_distribution(traffic_dist)
        baseline_dist_norm = self._renormalize_distribution(baseline_dist)

        # Maximize negative KL (minimize divergence)
        kl = F.kl_div(traffic_dist_norm.log(), baseline_dist_norm, reduction='batchmean')
        return -kl

    def compute(self, adv_score: torch.Tensor, info_gain: float,
                traffic_dist: Optional[torch.Tensor] = None,
                baseline_dist: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the final composite reward.
        """
        r_adv = self.calculate_adversarial_reward(adv_score)
        r_info = torch.tensor(info_gain, device=adv_score.device)

        total = (self.w_adv * r_adv) + (self.w_info * r_info)

        if traffic_dist is not None and baseline_dist is not None:
            r_stealth = self.calculate_stealth_reward(traffic_dist, baseline_dist)
            total = total + (self.w_stealth * r_stealth)

        return total
