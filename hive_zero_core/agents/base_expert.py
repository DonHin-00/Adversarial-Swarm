from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_zero_core.utils.logging_config import setup_logger


class BaseExpert(nn.Module, ABC):
    """
    Abstract base class for all HIVE-ZERO expert agents.

    Provides standardized forward-pass gating (Sparse MoE enforcement),
    input validation, graceful error handling, dynamic shape adaptation,
    and gradient checkpointing support for memory-efficient training.
    """

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

        # Step counter for diagnostics and scheduling
        self._step_count = 0

        # Gradient checkpointing flag (reduces VRAM at cost of compute)
        self._use_checkpoint = False

    def enable_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training."""
        self._use_checkpoint = True

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standardized forward pass for all experts.
        Enforces Gating Logic (Sparse MoE).

        Args:
            x: Input tensor, typically [batch, dim] or [batch, seq, dim].
            context: Optional context tensor (e.g. edge_index for graph experts).
            mask: Optional binary mask tensor for action masking.

        Returns:
            Output tensor of shape [batch, action_dim].
        """
        # Hardening: Input validation
        if not isinstance(x, torch.Tensor):
            self.logger.error(f"Input x must be a torch.Tensor, got {type(x)}")
            raise TypeError("Input x must be a torch.Tensor")

        if not self.is_active:
            # Return zero tensor matching expected output shape
            batch_size = x.size(0)
            return torch.zeros((batch_size, self.action_dim), device=x.device)

        self._step_count += 1

        try:
            if self._use_checkpoint and self.training:
                # Wrap non-Tensor args (context, mask may be None) in a closure
                # to avoid checkpoint errors with non-Tensor inputs.
                def _run_forward_impl(input_x: torch.Tensor) -> torch.Tensor:
                    return self._forward_impl(input_x, context, mask)

                return torch.utils.checkpoint.checkpoint(_run_forward_impl, x, use_reentrant=False)
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
        self.logger.info(f"[step={self._step_count}] Metrics: {metrics}")

    def ensure_dimension(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Dynamic Shape Adapter: Ensures the last dimension of x matches target_dim.
        Handles 2D [Batch, Dim] and 3D [Batch, Seq, Dim] inputs by padding or
        truncating the feature dimension.
        """
        if x.dim() == 2:
            current_dim = x.size(1)
            if current_dim == target_dim:
                return x
            elif current_dim < target_dim:
                padding = target_dim - current_dim
                return F.pad(x, (0, padding), "constant", 0)
            else:
                return x[:, :target_dim]
        elif x.dim() == 3:
            current_dim = x.size(-1)
            if current_dim == target_dim:
                return x
            elif current_dim < target_dim:
                padding = target_dim - current_dim
                return F.pad(x, (0, padding), "constant", 0)
            else:
                return x[:, :, :target_dim]
        return x
