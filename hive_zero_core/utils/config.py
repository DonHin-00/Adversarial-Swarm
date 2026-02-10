from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HiveConfig:
    """
    Centralized configuration for the HIVE-ZERO system.
    """
    # System Dimensions
    observation_dim: int = 64
    hidden_dim: int = 64

    # MoE Routing
    top_k: int = 3
    noise_epsilon: float = 1e-2

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    buffer_capacity: int = 10000

    # Agent Specific
    max_seq_len: int = 1000
    sentinel_model: str = "prajjwal1/bert-tiny"
    payload_gen_model: str = "t5-small"

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HiveConfig':
        return cls(**data)
