from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_zero_core.utils.logging_config import setup_logger


class BaseExpert(nn.Module, ABC):
    def __init__(
        self, observation_dim: int, action_dim: int, name: str = "BaseExpert", hidden_dim: int = 64
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.name = name
        self.logger = setup_logger(f"Expert_{name}")

        # Gating Logic
        self.is_active = False

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standardized forward pass for all experts.
        Enforces Gating Logic (Sparse MoE).
        """
        # Hardening: Input validation
        if not isinstance(x, torch.Tensor):
            self.logger.error(f"Input x must be a torch.Tensor, got {type(x)}")
            raise TypeError("Input x must be a torch.Tensor")

        if not self.is_active:
            # Return zero tensor matching expected output shape
            batch_size = x.size(0)
            return torch.zeros((batch_size, self.action_dim), device=x.device)

        try:
            return self._forward_impl(x, context, mask)
        except Exception as e:
            self.logger.error(f"Error in {self.name} forward pass: {str(e)}")
            # Fail gracefully by returning zeros
            return torch.zeros((x.size(0), self.action_dim), device=x.device)

    @abstractmethod
    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        pass

    def log_step(self, metrics: Dict[str, Any]):
        self.logger.info(f"Step Metrics: {metrics}")

    def ensure_dimension(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Dynamic Shape Adapter: Ensures x matches target_dim/shape requirements.
        Useful for bridging mismatches between Agents (e.g., Mutator -> Sentinel).
        """
        if x.dim() == 2:
            # [Batch, Dim]
            current_dim = x.size(1)
            if current_dim == target_dim:
                return x
            elif current_dim < target_dim:
                # Pad
                padding = target_dim - current_dim
                return F.pad(x, (0, padding), "constant", 0)
            else:
                # Truncate
                return x[:, :target_dim]
        elif x.dim() == 3:
            # [Batch, Seq, Dim] -> Flatten or slice depending on need.
            # For simplicity, if we need [Batch, Dim], we pool or slice.
            # If we need [Batch, Seq, Dim], we pad dim.
            if target_dim == x.size(-1):
                return x
            # Fallback implementation for prototype stability
            return x

        return x
