import logging
import os

import pytest


@pytest.fixture
def mock_env():
    """
    Fixture to safely modify environment variables during a test.
    Restores original environment after the test completes.
    """
    original_env = os.environ.copy()
    yield os.environ
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def reset_logging():
    """
    Fixture to reset the logging configuration after each test.
    This prevents test pollution where one test's logging setup affects others.
    """
    logger = logging.getLogger()
    old_handlers = logger.handlers[:]
    yield
    logger.handlers = old_handlers
