# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Updated existing workflows to use modern action versions
- Fixed deprecated ubuntu-18.04 in neuralegion workflow
- Improved security-scan workflow configuration

### Fixed
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
