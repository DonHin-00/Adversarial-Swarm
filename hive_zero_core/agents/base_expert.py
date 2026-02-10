import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union
from abc import ABC, abstractmethod
from hive_zero_core.utils.logging_config import setup_logger
from torch_geometric.data import HeteroData

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', obs_dim={self.observation_dim}, action_dim={self.action_dim})"

    def __str__(self) -> str:
        return f"[{self.name}] Agent (Active: {self.is_active})"

    def forward(self, x: Union[torch.Tensor, HeteroData], context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
                if 'ip' in x:
                    batch_size = x['ip'].x.size(0)
                    device = x['ip'].x.device
                else:
                    batch_size = 0
                    device = torch.device('cpu') # Fallback

            return torch.zeros((batch_size, self.action_dim), device=device)

        try:
            return self._forward_impl(x, context, mask)
        except Exception as e:
            self.logger.error(f"Error in {self.name} forward pass: {str(e)}")
            # Fail gracefully
            if isinstance(x, torch.Tensor):
                 return torch.zeros((x.size(0), self.action_dim), device=x.device)
            return torch.zeros((0, self.action_dim)) # Fallback

    def forward_ungated(self, x: Union[torch.Tensor, HeteroData], context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Bypasses gating logic and directly calls the implementation.
        Useful for internal dependencies between experts during optimization.
        Includes error handling to prevent crashes in optimization loops.
        """
        try:
            return self._forward_impl(x, context, mask)
        except Exception as e:
            self.logger.error(f"Error in {self.name} forward_ungated pass: {str(e)}")
            # Fail gracefully - determine device from input
            device = torch.device('cpu')  # Default fallback
            if isinstance(x, torch.Tensor):
                device = x.device
                return torch.zeros((x.size(0), self.action_dim), device=device)
            elif isinstance(x, HeteroData):
                 if 'ip' in x:
                     device = x['ip'].x.device
                     return torch.zeros((x['ip'].x.size(0), self.action_dim), device=device)
            return torch.zeros((0, self.action_dim), device=device)

    @abstractmethod
    def _forward_impl(self, x: Union[torch.Tensor, HeteroData], context: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        pass

    def log_step(self, metrics: Dict[str, Any]):
        self.logger.info(f"Step Metrics: {metrics}")
