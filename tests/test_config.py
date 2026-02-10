"""
Tests for training configuration management.
"""

import pytest
from pathlib import Path

from hive_zero_core.training.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
    get_full_training_config,
    get_quick_test_config,
)


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig()
    assert config.observation_dim == 64
    assert config.action_dim == 128
    assert config.hidden_dim == 64
    assert config.num_experts == 14
    assert config.pretrained is False
    assert config.top_k_experts == 3


def test_training_config_defaults():
    """Test TrainingConfig default values."""
    config = TrainingConfig()
    assert config.num_epochs == 10
    assert config.learning_rate == 0.001
    assert config.batch_size == 32
    assert config.optimizer == "adam"
    assert config.gradient_clip_norm == 1.0


def test_data_config_defaults():
    """Test DataConfig default values."""
    config = DataConfig()
    assert config.batch_size == 32
    assert config.synthetic is True
    assert config.num_synthetic_samples == 1000
    assert config.shuffle is True


def test_experiment_config_initialization():
    """Test ExperimentConfig initialization and validation."""
    config = ExperimentConfig()
    
    # Check that sub-configs are initialized
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.data, DataConfig)
    
    # Check experiment settings
    assert config.experiment_name == "adversarial_swarm_experiment"
    assert config.seed == 42
    assert config.device == "auto"
    
    # Check that directories are created
    assert config.output_dir.exists()
    assert config.training.checkpoint_dir.exists()


def test_experiment_config_validation():
    """Test that ExperimentConfig validates parameters."""
    # Valid config should work
    config = ExperimentConfig()
    assert config.model.observation_dim > 0
    
    # Invalid observation_dim should raise error
    with pytest.raises(AssertionError):
        ExperimentConfig(model=ModelConfig(observation_dim=-1))
    
    # Invalid action_dim should raise error
    with pytest.raises(AssertionError):
        ExperimentConfig(model=ModelConfig(action_dim=0))
    
    # Invalid top_k should raise error
    with pytest.raises(AssertionError):
        ExperimentConfig(model=ModelConfig(top_k_experts=20))


def test_experiment_config_to_dict():
    """Test converting ExperimentConfig to dictionary."""
    config = ExperimentConfig()
    config_dict = config.to_dict()
    
    assert "model" in config_dict
    assert "training" in config_dict
    assert "data" in config_dict
    assert config_dict["experiment_name"] == "adversarial_swarm_experiment"
    assert config_dict["seed"] == 42


def test_experiment_config_from_dict():
    """Test creating ExperimentConfig from dictionary."""
    config_dict = {
        "model": {"observation_dim": 128, "action_dim": 256},
        "training": {"num_epochs": 20, "learning_rate": 0.0001},
        "data": {"batch_size": 64},
        "experiment_name": "test_experiment",
        "seed": 123,
    }
    
    config = ExperimentConfig.from_dict(config_dict)
    
    assert config.model.observation_dim == 128
    assert config.model.action_dim == 256
    assert config.training.num_epochs == 20
    assert config.training.learning_rate == 0.0001
    assert config.data.batch_size == 64
    assert config.experiment_name == "test_experiment"
    assert config.seed == 123


def test_get_default_config():
    """Test getting default configuration."""
    config = get_default_config()
    assert isinstance(config, ExperimentConfig)
    assert config.training.num_epochs == 10


def test_get_quick_test_config():
    """Test getting quick test configuration."""
    config = get_quick_test_config()
    assert isinstance(config, ExperimentConfig)
    assert config.training.num_epochs == 2
    assert config.data.num_synthetic_samples == 100


def test_get_full_training_config():
    """Test getting full training configuration."""
    config = get_full_training_config()
    assert isinstance(config, ExperimentConfig)
    assert config.training.num_epochs == 100
    assert config.data.num_synthetic_samples == 10000
    assert config.training.use_lr_scheduler is True


def test_custom_config():
    """Test creating custom configuration."""
    model_config = ModelConfig(observation_dim=256, hidden_dim=128)
    training_config = TrainingConfig(num_epochs=50, learning_rate=0.0005)
    data_config = DataConfig(batch_size=16, num_synthetic_samples=500)
    
    config = ExperimentConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        experiment_name="custom_experiment",
    )
    
    assert config.model.observation_dim == 256
    assert config.model.hidden_dim == 128
    assert config.training.num_epochs == 50
    assert config.training.learning_rate == 0.0005
    assert config.data.batch_size == 16
    assert config.experiment_name == "custom_experiment"
