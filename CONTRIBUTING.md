# Contributing to Adversarial-Swarm

Thank you for your interest in contributing to Adversarial-Swarm! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Adversarial-Swarm.git
   cd Adversarial-Swarm
   ```
3. **Set up the development environment**:
   ```bash
   make install-dev
   make pre-commit
   ```

## Development Workflow

### Creating a Branch

Create a feature branch for your work:
```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### Making Changes

1. **Write clean, readable code** following the project's style
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Run tests locally** before committing:
   ```bash
   make test
   make lint
   ```

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Style guide enforcement
- **pylint**: Code analysis
- **mypy**: Type checking

Format your code before committing:
```bash
make format
```

Check code quality:
```bash
make lint
```

### Commit Messages

Write clear, descriptive commit messages:

```
Add feature: Brief description of the change

More detailed explanation of what changed and why.
Reference any related issues (#123).
```

### Pre-commit Hooks

Pre-commit hooks will automatically run when you commit. They will:
- Format code with Black and isort
- Check for common issues
- Validate YAML and JSON files
- Check for large files

If hooks fail, fix the issues and re-commit.

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basic.py

# Run with coverage
make test
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `conftest.py`
- Aim for high test coverage

Example test:
```python
def test_new_feature(observation_dim):
    """Test description."""
    # Arrange
    hive = HiveMind(observation_dim=observation_dim)
    
    # Act
    result = hive.some_method()
    
    # Assert
    assert result is not None
```

## Pull Request Process

1. **Update your branch** with the latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Describe what changed and why
   - Include screenshots for UI changes

4. **Address review feedback**:
   - Respond to comments
   - Make requested changes
   - Push updates to your branch

5. **Wait for CI checks** to pass:
   - All tests must pass
   - Code must pass linting
   - Security scans must be clean

## Code Review

All contributions require code review before merging. Reviewers will check:

- Code quality and style
- Test coverage
- Documentation
- Security implications
- Performance considerations

## Questions?

If you have questions:
- Open an issue for discussion
- Reach out to maintainers
- Check existing issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
