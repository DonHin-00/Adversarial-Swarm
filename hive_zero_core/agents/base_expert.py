from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from hive_zero_core.utils.logging_config import setup_logger


class SkillLevel(Enum):
    """Skill proficiency levels for agents"""

    NOVICE = 1
    INTERMEDIATE = 2
    EXPERT = 3
    MASTER = 4


class BaseExpert(nn.Module, ABC):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        name: str = "BaseExpert",
        hidden_dim: int = 64,
        primary_skills: Optional[List[str]] = None,
        secondary_skills: Optional[List[str]] = None,
        skill_level: SkillLevel = SkillLevel.INTERMEDIATE,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.name = name
        self.logger = setup_logger(f"Expert_{name}")

        # Gating Logic
        self.is_active = False

        # Skill System
        self.skill_level = skill_level
        self.primary_skills = primary_skills or []
        self.secondary_skills = secondary_skills or []

        # Skill proficiency multipliers based on level
        self.skill_multipliers = {
            SkillLevel.NOVICE: 0.7,
            SkillLevel.INTERMEDIATE: 1.0,
            SkillLevel.EXPERT: 1.3,
            SkillLevel.MASTER: 1.5,
        }

        # Track skill usage and effectiveness
        self.skill_stats = {
            "activations": 0,
            "successes": 0,
            "failures": 0,
            "avg_confidence": 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"skill_level={self.skill_level.name}, "
            f"obs_dim={self.observation_dim}, action_dim={self.action_dim})"
        )

    def __str__(self) -> str:
        return (
            f"[{self.name}] Agent (Active: {self.is_active}, "
            f"Level: {self.skill_level.name}, "
            f"Skills: {len(self.primary_skills)})"
        )

    def get_skill_multiplier(self) -> float:
        """Returns the proficiency multiplier for this agent's skill level."""
        return self.skill_multipliers[self.skill_level]

    def record_activation(self, success: bool = False, confidence: float = 0.0):
        """Track skill usage statistics."""
        self.skill_stats["activations"] += 1
        if success:
            self.skill_stats["successes"] += 1
        else:
            self.skill_stats["failures"] += 1

        # Update running average of confidence
        n = self.skill_stats["activations"]
        self.skill_stats["avg_confidence"] = (
            (n - 1) * self.skill_stats["avg_confidence"] + confidence
        ) / n

    def get_effectiveness_score(self) -> float:
        """Calculate overall effectiveness score (0.0 to 1.0)."""
        if self.skill_stats["activations"] == 0:
            return 0.5  # Neutral score for untested agents

        success_rate = self.skill_stats["successes"] / self.skill_stats["activations"]
        confidence = self.skill_stats["avg_confidence"]

        # Combine success rate and confidence with skill multiplier
        base_score = success_rate * 0.7 + confidence * 0.3
        return min(1.0, base_score * self.get_skill_multiplier())

    def forward(
        self,
        x: Union[torch.Tensor, HeteroData],
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standardized forward pass for all experts.
        Enforces Gating Logic (Sparse MoE).
        """
        # Hardening: Input validation (Relaxed for HeteroData)
        if not isinstance(x, (torch.Tensor, HeteroData)):
            self.logger.error(f"Input x must be Tensor or HeteroData, got {type(x)}")
            raise TypeError("Input x must be Tensor or HeteroData")

        if not self.is_active:
            # How to return zeros? Need batch size.
            if isinstance(x, torch.Tensor):
                batch_size = x.size(0)
                device = x.device
            elif isinstance(x, HeteroData):
                # Assume batch size based on 'ip' nodes or primary node type?
                # For safety, use 0 if ambiguous, or try to infer from typical node type 'ip'
                if "ip" in x:
                    batch_size = x["ip"].x.size(0)
                    device = x["ip"].x.device
                else:
                    batch_size = 0
                    device = torch.device("cpu")  # Fallback

            return torch.zeros((batch_size, self.action_dim), device=device)

        try:
            return self._forward_impl(x, context, mask)
        except Exception as e:
            self.logger.error(f"Error in {self.name} forward pass: {str(e)}")
            # Fail gracefully
            if isinstance(x, torch.Tensor):
                return torch.zeros((x.size(0), self.action_dim), device=x.device)
            return torch.zeros((0, self.action_dim))  # Fallback

    @abstractmethod
    def _forward_impl(
        self,
        x: Union[torch.Tensor, HeteroData],
        context: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass

    def log_step(self, metrics: Dict[str, Any]):
        self.logger.info(f"Step Metrics: {metrics}")
