# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pull request cleanup tools and documentation
  - `scripts/close_duplicate_prs.sh` - Automated script to close duplicate PRs
  - `scripts/cleanup_duplicate_prs.py` - Python tool to detect and analyze duplicate PRs
  - `docs/PR_CLEANUP_REPORT.md` - Detailed analysis of duplicate PRs with recommendations
  - `docs/PR_CLEANUP_SUMMARY.md` - Executive summary and execution guide
  - `scripts/README.md` - Comprehensive documentation for all maintenance scripts
- Comprehensive training infrastructure with configuration management system
  - `config.py` with ModelConfig, TrainingConfig, DataConfig, and ExperimentConfig
  - Hierarchical configuration with validation and presets
  - Support for quick test, default, and full training configurations
- Data loading utilities with synthetic data generation
  - `data_loader.py` with NetworkLogDataset class
  - Synthetic network log generation for training
  - Iterator-based batch processing
- Checkpoint save/load functionality in training loop
  - Automatic checkpoint saving at configurable intervals
  - Resume training from saved checkpoints
- Learning rate scheduling support (cosine, step, exponential)
- Configurable loss component weights
- Comprehensive test suite with 55+ tests
  - `test_config.py` - 13 tests for configuration management
  - `test_data_loader.py` - 10 tests for data loading
  - `test_rewards.py` - 12 tests for reward calculations
  - `test_experts.py` - 20+ tests for expert implementations
- Detailed training documentation (`docs/TRAINING_GUIDE.md`)
- Comprehensive CI/CD pipeline with Python CI workflow
- Security scanning with CodeQL and Semgrep
- Docker support with Dockerfile and docker-compose.yml
- Pre-commit hooks for code quality
- Makefile for common development tasks
- Full test infrastructure with pytest
- Code formatting with Black and isort
- Linting with flake8, pylint, and mypy
- Project setup files (setup.py, pyproject.toml)
- Comprehensive README with build instructions
- Contributing guidelines
- Development documentation

### Changed
- Refactored `adversarial_loop.py` to use new configuration system
  - Better structured training loop with proper logging
  - Integration with NetworkLogDataset for data loading
  - Support for batch training instead of single-sample
  - Configurable optimizer and learning rate scheduling
  - More trainable experts (added Chronos, Stego, Cleaner)
- Improved DeepScope masking logic with proper broadcasting
  - Added shape checking and error handling
  - Support for various mask shapes
  - Fallback behavior for incompatible shapes
- Updated existing workflows to use modern action versions
- Fixed deprecated ubuntu-18.04 in neuralegion workflow
- Improved security-scan workflow configuration

### Fixed
- Fixed incomplete Mutator parameter optimization in training loop
  - Replaced `pass` statement with explicit `continue` and explanation
  - Clarified that Mutator uses inference-time optimization
- Implemented auxiliary loss computation for optimized payloads
  - Added L2 regularization to prevent explosion
  - Added diversity loss to prevent mode collapse
- Completed DeepScope masking broadcasting logic
  - Fixed `pass` statement with proper implementation
  - Added robust error handling for shape mismatches
- Removed broken steps from ethicalcheck workflow
- Fixed incorrect language configuration in CodeQL (was Java, now Python)
- Updated workflow triggers to include feature branches

## [0.1.0] - Initial Release

### Added
- Multi-expert architecture with 14 specialized agents
- HiveMind coordination system
- Graph-based memory storage
- Gating network for expert selection
- Five operational clusters: Recon, Attack, Post-Exploit, Defense, Kill Chain
- Basic project structure and dependencies
