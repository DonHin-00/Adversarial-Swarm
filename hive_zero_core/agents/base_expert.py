import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
from hive_zero_core.utils.logging_config import setup_logger

class BaseExpert(nn.Module, ABC):
    def __init__(self, observation_dim: int, action_dim: int, name: str = "BaseExpert", hidden_dim: int = 64):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.name = name
        self.logger = setup_logger(f"Expert_{name}")

        # Gating Logic
        self.is_active = False

        # Common device management - will be set by external logic or inherited
        # self.device is handled by PyTorch internals (self.to(device))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Standardized forward pass for all experts.
        Enforces Gating Logic (Sparse MoE).

        Args:
            x: Input tensor (observation)
            context: Global context (e.g., Graph embedding)
            mask: Optional mask

        Returns:
            Output tensor. Returns zero tensor if is_active is False.
        """
        # Hardening: Input validation
        if not isinstance(x, torch.Tensor):
            self.logger.error(f"Input x must be a torch.Tensor, got {type(x)}")
            raise TypeError("Input x must be a torch.Tensor")

        if not self.is_active:
            # Return zero tensor matching expected output shape
            # We assume output shape is [batch_size, action_dim] usually
            batch_size = x.size(0)
            return torch.zeros((batch_size, self.action_dim), device=x.device) # Use x.device

        try:
            return self._forward_impl(x, context, mask)
        except Exception as e:
            self.logger.error(f"Error in {self.name} forward pass: {str(e)}")
            # Fail gracefully by returning zeros
            return torch.zeros((x.size(0), self.action_dim), device=x.device)

    @abstractmethod
    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Specific implementation logic for the expert.
        """
        pass

    def log_step(self, metrics: Dict[str, Any]):
        """
        Log metrics to logger (and potentially TensorBoard later).
        """
        self.logger.info(f"Step Metrics: {metrics}")
