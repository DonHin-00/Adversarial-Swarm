import torch
import torch.nn.functional as F


class CompositeReward:
    """
    Multi-objective reward function combining adversarial evasion, information
    gain, stealth (traffic mimicry), and temporal efficiency components.

    Attributes:
        w_adv:      Weight for adversarial evasion reward.
        w_info:     Weight for information-gain reward.
        w_stealth:  Weight for traffic-stealth reward.
        w_temporal: Weight for temporal-efficiency reward.
    """

    def __init__(self, w_adv: float = 1.0, w_info: float = 0.5,
                 w_stealth: float = 0.8, w_temporal: float = 0.3):
        self.w_adv = w_adv
        self.w_info = w_info
        self.w_stealth = w_stealth
        self.w_temporal = w_temporal

    # ------------------------------------------------------------------
    # Individual reward components
    # ------------------------------------------------------------------

    def calculate_adversarial_reward(self, sentinel_score: torch.Tensor) -> torch.Tensor:
        """
        R_adv = P(Allowed).  Sentinel outputs logits [Blocked, Allowed];
        the caller should pass the softmaxed P(Allowed) column.
        """
        return sentinel_score

    def calculate_info_gain_reward(self, prev_entropy: float, current_entropy: float) -> float:
        """
        R_info: positive reward when Knowledge-Graph entropy decreases.
        """
        gain = prev_entropy - current_entropy
        return max(0.0, gain)

    def calculate_stealth_reward(self, traffic_dist: torch.Tensor,
                                 baseline_dist: torch.Tensor) -> torch.Tensor:
        """
        R_stealth = −KL(traffic ‖ baseline).

        Both inputs are expected to be un-normalised probabilities (or
        probability vectors).  They are clamped to [1e-8, ∞) before the
        log to prevent NaN/Inf.
        """
        if traffic_dist.shape != baseline_dist.shape:
            return torch.tensor(0.0, device=traffic_dist.device,
                                dtype=traffic_dist.dtype)

        traffic_log = torch.log(torch.clamp(traffic_dist, min=1e-8))
        baseline_clamped = torch.clamp(baseline_dist, min=1e-8)
        kl = F.kl_div(traffic_log, baseline_clamped, reduction='batchmean')
        return -kl

    def calculate_temporal_reward(self, elapsed_steps: int,
                                  budget: int = 100) -> float:
        """
        R_temporal: rewards faster completion.

        Returns a value in (0, 1] that decays linearly as elapsed_steps
        approaches the budget.
        """
        return max(0.0, 1.0 - elapsed_steps / max(budget, 1))

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    def compute(self, adv_score: torch.Tensor, info_gain: float,
                traffic_dist: torch.Tensor, baseline_dist: torch.Tensor,
                elapsed_steps: int = 0, step_budget: int = 100) -> torch.Tensor:
        """Weighted sum of all reward components."""
        r_adv = self.calculate_adversarial_reward(adv_score)
        r_stealth = self.calculate_stealth_reward(traffic_dist, baseline_dist)
        r_temporal = self.calculate_temporal_reward(elapsed_steps, step_budget)

        total = (
            self.w_adv * r_adv
            + self.w_info * info_gain
            + self.w_stealth * r_stealth
            + self.w_temporal * r_temporal
        )
        return total
