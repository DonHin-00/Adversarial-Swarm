# CI/CD Documentation

## Overview

This repository uses GitHub Actions for continuous integration and deployment.

## Workflows

### Python CI
- Automated testing and linting
- Multi-version Python testing (3.10, 3.11, 3.12)
- Code quality checks (flake8, black, isort, mypy)

### Security Scan
- CodeQL Analysis for Python
- Semgrep security scanning
- Weekly scheduled scans

## Local Development

### Running Checks

```bash
make install-dev  # Install development dependencies
make lint         # Run all linters
make format       # Format code
make test         # Run tests
```

### Validation Script

```bash
./scripts/validate-build.sh
```

## Best Practices

1. Run checks locally before pushing
2. Keep dependencies updated
3. Review security alerts promptly
4. Format code before committing
5. Write tests for new features
