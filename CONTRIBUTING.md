# Contributing to Adversarial Swarm

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Adversarial-Swarm.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Submit a pull request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Code Quality Standards

### Style Guide

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Maximum line length: 100 characters

### Testing

- Write tests for all new features
- Maintain >90% code coverage
- Include unit and integration tests
- Test security-critical code thoroughly

### Security

- Never commit secrets or credentials
- Use secure coding practices
- Run security scanners before submitting
- Report security issues privately

## Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Run code quality checks
5. Update CHANGELOG.md
6. Request review from maintainers

## Questions?

Open an issue or reach out to the maintainers.
