"""
Configuration management for training the HiveMind system.
Centralizes all hyperparameters and training settings.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the HiveMind model architecture."""

    observation_dim: int = 64
    action_dim: int = 128
    hidden_dim: int = 64
    num_experts: int = 14
    pretrained: bool = False
    
    # Expert-specific settings
    bert_model: str = "prajjwal1/bert-tiny"
    t5_model: str = "t5-small"
    
    # Gating network settings
    gating_hidden_dim: int = 128
    top_k_experts: int = 3


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    num_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # Loss weights
    adversarial_weight: float = 1.0
    info_gain_weight: float = 0.5
    stealth_weight: float = 0.3
    
    # Auxiliary loss weights
    l2_regularization: float = 0.01
    diversity_weight: float = 0.1
    
    # Optimization settings
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0001
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"  # "cosine", "step", "exponential"
    lr_warmup_epochs: int = 2
    
    # Checkpoint settings
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_frequency: int = 5  # Save every N epochs
    keep_last_n_checkpoints: int = 3


@dataclass
class DataConfig:
    """Configuration for data loading."""

    data_source: Optional[str] = None
    batch_size: int = 32
    synthetic: bool = True
    num_synthetic_samples: int = 1000
    shuffle: bool = True
    num_workers: int = 0  # For future DataLoader integration


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    experiment_name: str = "adversarial_swarm_experiment"
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Logging
    log_level: str = "INFO"
    log_frequency: int = 1  # Log every N batches
    tensorboard: bool = False
    wandb: bool = False

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate values
        assert self.model.observation_dim > 0, "observation_dim must be positive"
        assert self.model.action_dim > 0, "action_dim must be positive"
        assert self.training.num_epochs > 0, "num_epochs must be positive"
        assert self.training.learning_rate > 0, "learning_rate must be positive"
        assert 1 <= self.model.top_k_experts <= self.model.num_experts, \
            "top_k_experts must be between 1 and num_experts"

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            experiment_name=config_dict.get("experiment_name", "adversarial_swarm_experiment"),
            output_dir=Path(config_dict.get("output_dir", "outputs")),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "auto"),
        )


def get_default_config() -> ExperimentConfig:
    """Get default configuration for experiments."""
    return ExperimentConfig()


def get_quick_test_config() -> ExperimentConfig:
    """Get configuration optimized for quick testing."""
    config = ExperimentConfig()
    config.training.num_epochs = 2
    config.data.num_synthetic_samples = 100
    config.training.save_frequency = 1
    return config


def get_full_training_config() -> ExperimentConfig:
    """Get configuration for full training run."""
    config = ExperimentConfig()
    config.training.num_epochs = 100
    config.data.num_synthetic_samples = 10000
    config.training.save_frequency = 10
    config.training.use_lr_scheduler = True
    return config
