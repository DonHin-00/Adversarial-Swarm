from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class HiveConfig:
    """
    Centralized configuration for the HIVE-ZERO system.
    
    Configuration values can be overridden via environment variables:
    - HIVE_SENTINEL_MODEL: Sentinel model name
    - HIVE_PAYLOAD_GEN_MODEL: Payload generator model name
    - HIVE_MUTATOR_LR: Mutator SGD learning rate
    - HIVE_T5_MAX_LENGTH: T5 generation max length
    - HIVE_STEGO_INJECTION_ALPHA: Steganography injection strength
    - HIVE_SENTINEL_HIDDEN_SIZE_FALLBACK: Fallback hidden size if model config unavailable
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
    sentinel_model: str = os.getenv('HIVE_SENTINEL_MODEL', 'prajjwal1/bert-tiny')
    payload_gen_model: str = os.getenv('HIVE_PAYLOAD_GEN_MODEL', 't5-small')
    
    # Mutator Configuration
    mutator_learning_rate: float = float(os.getenv('HIVE_MUTATOR_LR', '0.01'))
    mutator_iterations: int = 5
    mutator_noise_scale: float = 0.05
    
    # PayloadGen Configuration
    t5_max_length: int = int(os.getenv('HIVE_T5_MAX_LENGTH', '64'))
    rag_db_size: int = 10
    rag_embedding_dim: int = 64
    rag_template_length: int = 20
    
    # Steganography Configuration
    stego_injection_alpha: float = float(os.getenv('HIVE_STEGO_INJECTION_ALPHA', '0.05'))
    
    # HiveMind Configuration
    sentinel_hidden_size_fallback: int = int(os.getenv('HIVE_SENTINEL_HIDDEN_SIZE_FALLBACK', '128'))

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HiveConfig':
        return cls(**data)
