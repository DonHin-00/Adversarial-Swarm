import torch
import torch.nn.functional as F  # noqa: N812


class CompositeReward:
    def __init__(self, w_adv: float = 1.0, w_info: float = 0.5, w_stealth: float = 0.8):
        self.w_adv = w_adv
        self.w_info = w_info
        self.w_stealth = w_stealth

    def calculate_adversarial_reward(self, sentinel_score: torch.Tensor) -> torch.Tensor:
        """
        R_adv: Maximize probability of evasion (1 - Sentinel Score)
        Sentinel Score: P(Blocked) [Index 0] or P(Allowed) [Index 1]?
        Sentinel output from forward() is typically P(Allowed) in our impl.
        Wait, Sentinel classifies [Blocked, Allowed].
        If Payload is Allowed, Score -> 1. We want to MAXIMIZE this.

        So R_adv = P(Allowed).
        """
        # Ensure input is prob, not logit
        return sentinel_score

    def calculate_info_gain_reward(self, prev_entropy: float, current_entropy: float) -> float:
        """
        R_info: Reduction in entropy of Knowledge Graph.
        Gain = Prev - Current.
        """
        gain = prev_entropy - current_entropy
        return max(0.0, gain)  # Reward positive gain only

    def calculate_stealth_reward(self, traffic_dist: torch.Tensor, baseline_dist: torch.Tensor) -> torch.Tensor:
        """
        R_stealth: Minimize KL Divergence between generated traffic and baseline.
        Reward = -KL(P || Q)
        """
        # Assume distributions are categorical or Gaussian params?
        # For prototype, assume simple MSE or negative distance if continuous
        # Or KL if prob distributions.

        # Hardening: Check shapes
        if traffic_dist.shape != baseline_dist.shape:
            # Fallback
            return torch.tensor(0.0)

        # KL Divergence (assuming log_probs input for P? or probs?)
        # Let's assume input is probs.
        kl = F.kl_div(traffic_dist.log(), baseline_dist, reduction="batchmean")
        return -kl  # Maximize negative KL (minimize divergence)

    def compute(self, adv_score, info_gain, traffic_dist, baseline_dist) -> torch.Tensor:
        r_adv = self.calculate_adversarial_reward(adv_score)
        r_info = info_gain
        r_stealth = self.calculate_stealth_reward(traffic_dist, baseline_dist)

        # Normalize?
        total = (self.w_adv * r_adv) + (self.w_info * r_info) + (self.w_stealth * r_stealth)
        return total
