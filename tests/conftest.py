import pytest
import logging
from hive_zero_core.utils.logging_config import setup_logger

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Ensure logging is configured for tests."""
    setup_logger()
    logging.getLogger().setLevel(logging.DEBUG)

@pytest.fixture
def mock_env_config():
    """Fixture to provide a mock environment configuration."""
    return {
        "system": {
            "observation_dim": 64,
            "hidden_dim": 64,
            "device": "cpu"
        },
        "env": {
            "max_steps": 10,
            "num_nodes": 5
        }
    }
