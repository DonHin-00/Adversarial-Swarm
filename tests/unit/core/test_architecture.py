import importlib
import pkgutil
from pathlib import Path

import pytest

import hive_zero_core


def test_all_modules_importable():
    """
    Smoke test: Ensure all modules in hive_zero_core can be imported.
    This catches syntax errors and missing dependencies immediately.
    """
    package = hive_zero_core
    path = package.__path__
    prefix = package.__name__ + "."

    for _, name, _ in pkgutil.walk_packages(path, prefix):
        # Skip training modules if torch is not fully configured in CI/test environment
        # But we installed torch, so it should be fine.
        # If any import fails, the test fails.
        try:
            importlib.import_module(name)
        except ImportError as e:
            pytest.fail(f"Failed to import {name}: {e}")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {name}: {e}")


def test_readme_exists():
    """
    Ensure README.md exists and is not empty.
    """
    readme = Path("README.md")
    assert readme.exists(), "README.md is missing"
    assert readme.stat().st_size > 0, "README.md is empty"


def test_config_validity():
    """
    Ensure config.yaml exists and is valid YAML.
    """
    import yaml  # noqa: PLC0415

    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            try:
                config = yaml.safe_load(f)
                assert isinstance(config, (dict, list)), "Config root must be a dict or list"
            except yaml.YAMLError as e:
                pytest.fail(f"config.yaml is invalid: {e}")
    else:
        pytest.skip("configs/config.yaml not found")
