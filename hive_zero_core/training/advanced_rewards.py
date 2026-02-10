"""
Advanced Reward System with Extended Affect Types.

This module extends the basic reward system with additional affect types:
- Temporal affects (timing precision, speed)
- Resource affects (computational efficiency)
- Reliability affects (consistency, success rate)
- Novelty affects (discovering new patterns)
- Coordination affects (multi-agent synergy)
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


class AdvancedCompositeReward:
    """
    Enhanced reward system with multiple affect types for HIVE-ZERO.

    Affect Components:
        - Adversarial (Evasion): Ability to bypass detection
        - Information Gain: Knowledge graph entropy reduction
        - Stealth: Traffic mimicry quality
        - Temporal: Speed and timing precision
        - Resource: Computational efficiency
        - Reliability: Consistency and success rate
        - Novelty: Discovery of new techniques
        - Coordination: Multi-agent synergy bonus
    """

    def __init__(
        self,
        w_adv: float = 1.0,
        w_info: float = 0.5,
        w_stealth: float = 0.8,
        w_temporal: float = 0.4,
        w_resource: float = 0.3,
        w_reliability: float = 0.6,
        w_novelty: float = 0.5,
        w_coordination: float = 0.7,
    ):
        """
        Initialize with configurable weights for each affect type.

        Args:
            w_adv: Weight for adversarial evasion reward
            w_info: Weight for information gain reward
            w_stealth: Weight for stealth/mimicry reward
            w_temporal: Weight for temporal efficiency reward
            w_resource: Weight for resource efficiency reward
            w_reliability: Weight for reliability/consistency reward
            w_novelty: Weight for novelty/discovery reward
            w_coordination: Weight for multi-agent coordination reward
        """
        self.w_adv = w_adv
        self.w_info = w_info
        self.w_stealth = w_stealth
        self.w_temporal = w_temporal
        self.w_resource = w_resource
        self.w_reliability = w_reliability
        self.w_novelty = w_novelty
        self.w_coordination = w_coordination

    def _renormalize_distribution(self, dist: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Renormalize probability distribution with safety checks."""
        dist_clamped = torch.clamp(dist, min=epsilon)
        return dist_clamped / torch.clamp(dist_clamped.sum(dim=-1, keepdim=True), min=epsilon)

    def calculate_adversarial_reward(self, sentinel_score: torch.Tensor) -> torch.Tensor:
        """
        R_adv: Maximize probability of evasion.

        Args:
            sentinel_score: P(Allowed) from SentinelAgent [0, 1]

        Returns:
            Adversarial reward (higher = better evasion)
        """
        return torch.clamp(sentinel_score, 0.0, 1.0)

    def calculate_info_gain_reward(self, prev_entropy: float, current_entropy: float) -> float:
        """
        R_info: Reduction in entropy of Knowledge Graph.

        Args:
            prev_entropy: Entropy before reconnaissance
            current_entropy: Entropy after reconnaissance

        Returns:
            Information gain (higher = more knowledge acquired)
        """
        gain = prev_entropy - current_entropy
        return float(max(0.0, gain))

    def calculate_stealth_reward(
        self, traffic_dist: torch.Tensor, baseline_dist: torch.Tensor
    ) -> torch.Tensor:
        """
        R_stealth: Minimize KL Divergence between generated and baseline traffic.

        Args:
            traffic_dist: Distribution of generated traffic
            baseline_dist: Distribution of normal traffic

        Returns:
            Stealth reward (higher = better mimicry)
        """
        if traffic_dist.shape != baseline_dist.shape:
            return torch.tensor(0.0, device=traffic_dist.device)

        traffic_norm = self._renormalize_distribution(traffic_dist)
        baseline_norm = self._renormalize_distribution(baseline_dist)

        kl = F.kl_div(baseline_norm.log(), traffic_norm, reduction="batchmean")
        return -kl

    def calculate_temporal_reward(
        self,
        actual_duration: float,
        target_duration: float,
        timing_precision: float = 0.0,
    ) -> float:
        """
        R_temporal: Reward for speed and timing precision.

        Args:
            actual_duration: Time taken to execute action (seconds)
            target_duration: Target/optimal duration (seconds)
            timing_precision: Precision score for timing-sensitive ops [0, 1]

        Returns:
            Temporal reward (higher = faster and more precise)
        """
        # Speed component: reward faster execution, penalize slower
        if target_duration > 0:
            speed_score = max(0.0, 2.0 - (actual_duration / target_duration))
        else:
            speed_score = 1.0

        # Precision component: direct score
        precision_score = max(0.0, min(1.0, timing_precision))

        # Combine: 60% speed, 40% precision
        return 0.6 * speed_score + 0.4 * precision_score

    def calculate_resource_reward(
        self,
        compute_cost: float,
        memory_usage: float,
        max_compute: float = 1.0,
        max_memory: float = 1.0,
    ) -> float:
        """
        R_resource: Reward computational and memory efficiency.

        Args:
            compute_cost: Computational cost (normalized)
            memory_usage: Memory usage (normalized)
            max_compute: Maximum acceptable compute cost
            max_memory: Maximum acceptable memory usage

        Returns:
            Resource efficiency reward (higher = more efficient)
        """
        # Efficiency = 1 - (normalized usage)
        compute_efficiency = max(0.0, 1.0 - (compute_cost / max_compute))
        memory_efficiency = max(0.0, 1.0 - (memory_usage / max_memory))

        # Equal weighting for compute and memory
        return 0.5 * compute_efficiency + 0.5 * memory_efficiency

    def calculate_reliability_reward(
        self, success_count: int, total_attempts: int, consistency_score: float = 1.0
    ) -> float:
        """
        R_reliability: Reward consistency and success rate.

        Args:
            success_count: Number of successful executions
            total_attempts: Total number of attempts
            consistency_score: Variance/consistency metric [0, 1]

        Returns:
            Reliability reward (higher = more reliable)
        """
        if total_attempts == 0:
            return 0.5  # Neutral for untested

        success_rate = success_count / total_attempts
        consistency = max(0.0, min(1.0, consistency_score))

        # Combine: 70% success rate, 30% consistency
        return 0.7 * success_rate + 0.3 * consistency

    def calculate_novelty_reward(
        self, is_novel: bool, novelty_score: float = 0.0, exploration_bonus: float = 0.2
    ) -> float:
        """
        R_novelty: Reward discovering new patterns or techniques.

        Args:
            is_novel: Whether the action/pattern is novel
            novelty_score: Degree of novelty [0, 1]
            exploration_bonus: Bonus multiplier for exploration

        Returns:
            Novelty reward (higher = more novel/exploratory)
        """
        base_score = max(0.0, min(1.0, novelty_score))

        if is_novel:
            return base_score * (1.0 + exploration_bonus)
        return base_score * 0.5  # Reduced reward for non-novel

    def calculate_coordination_reward(
        self,
        active_agents: int,
        synergy_score: float = 0.0,
        optimal_team_size: int = 3,
    ) -> float:
        """
        R_coordination: Reward effective multi-agent coordination.

        Args:
            active_agents: Number of currently active agents
            synergy_score: Measured synergy between agents [0, 1]
            optimal_team_size: Optimal number of coordinated agents

        Returns:
            Coordination reward (higher = better teamwork)
        """
        if active_agents == 0:
            return 0.0

        # Team size component: reward optimal team size
        size_diff = abs(active_agents - optimal_team_size)
        size_score = max(0.0, 1.0 - (size_diff * 0.2))

        # Synergy component
        synergy = max(0.0, min(1.0, synergy_score))

        # Combine: 40% team size, 60% synergy quality
        return 0.4 * size_score + 0.6 * synergy

    def compute(
        self,
        # Core affects
        adv_score: torch.Tensor,
        info_gain: float = 0.0,
        traffic_dist: Optional[torch.Tensor] = None,
        baseline_dist: Optional[torch.Tensor] = None,
        # Temporal affects
        actual_duration: Optional[float] = None,
        target_duration: Optional[float] = None,
        timing_precision: float = 0.0,
        # Resource affects
        compute_cost: float = 0.0,
        memory_usage: float = 0.0,
        # Reliability affects
        success_count: int = 0,
        total_attempts: int = 0,
        consistency_score: float = 1.0,
        # Novelty affects
        is_novel: bool = False,
        novelty_score: float = 0.0,
        # Coordination affects
        active_agents: int = 1,
        synergy_score: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive composite reward with all affect types.

        Returns:
            Dictionary with individual rewards and total composite reward
        """
        device = adv_score.device

        # Core rewards
        r_adv = self.calculate_adversarial_reward(adv_score)
        r_info = torch.tensor(info_gain, device=device)

        rewards = {
            "adversarial": r_adv,
            "information": r_info,
        }

        total = (self.w_adv * r_adv) + (self.w_info * r_info)

        # Stealth reward
        if traffic_dist is not None and baseline_dist is not None:
            r_stealth = self.calculate_stealth_reward(traffic_dist, baseline_dist)
            rewards["stealth"] = r_stealth
            total = total + (self.w_stealth * r_stealth)

        # Temporal reward
        if actual_duration is not None and target_duration is not None:
            r_temporal = torch.tensor(
                self.calculate_temporal_reward(actual_duration, target_duration, timing_precision),
                device=device,
            )
            rewards["temporal"] = r_temporal
            total = total + (self.w_temporal * r_temporal)

        # Resource reward
        if compute_cost > 0 or memory_usage > 0:
            r_resource = torch.tensor(
                self.calculate_resource_reward(compute_cost, memory_usage), device=device
            )
            rewards["resource"] = r_resource
            total = total + (self.w_resource * r_resource)

        # Reliability reward
        if total_attempts > 0:
            r_reliability = torch.tensor(
                self.calculate_reliability_reward(success_count, total_attempts, consistency_score),
                device=device,
            )
            rewards["reliability"] = r_reliability
            total = total + (self.w_reliability * r_reliability)

        # Novelty reward
        if is_novel or novelty_score > 0:
            r_novelty = torch.tensor(
                self.calculate_novelty_reward(is_novel, novelty_score), device=device
            )
            rewards["novelty"] = r_novelty
            total = total + (self.w_novelty * r_novelty)

        # Coordination reward
        if active_agents > 0:
            r_coordination = torch.tensor(
                self.calculate_coordination_reward(active_agents, synergy_score),
                device=device,
            )
            rewards["coordination"] = r_coordination
            total = total + (self.w_coordination * r_coordination)

        rewards["total"] = total
        return rewards
