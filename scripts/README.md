# Scripts Directory

This directory contains utility scripts for repository maintenance and operations.

## Available Scripts

### PR Management

#### `close_duplicate_prs.sh`
**Purpose**: Close duplicate pull requests that have merged twins.

**Usage**:
```bash
./scripts/close_duplicate_prs.sh
```

**What it does**:
- Closes PR #17 (duplicate of merged PR #2)
- Closes PR #28 (duplicate of merged PR #31)
- Provides guidance for manual review of PRs #12 and #32

**Prerequisites**:
- GitHub CLI (`gh`) must be installed
- Must be authenticated with GitHub CLI (`gh auth login`)
- Requires write access to the repository

#### `cleanup_duplicate_prs.py`
**Purpose**: Analyze and generate reports on duplicate pull requests.

**Usage**:
```bash
python scripts/cleanup_duplicate_prs.py [--output OUTPUT_FILE]
```

**Options**:
- `--output`, `-o`: Output file for the report (default: `docs/PR_CLEANUP_REPORT.md`)

**What it does**:
- Fetches all pull requests from the repository
- Identifies duplicates based on normalized titles
- Finds related PRs (retries, sub-PRs)
- Generates a detailed markdown report with recommended actions

**Prerequisites**:
- Python 3.7+
- GitHub CLI (`gh`) installed and authenticated

**Example**:
```bash
# Generate report to default location
python scripts/cleanup_duplicate_prs.py

# Generate report to custom location
python scripts/cleanup_duplicate_prs.py --output /tmp/pr-report.md
```

### Build and Validation

#### `validate-build.sh`
**Purpose**: Validate that the project builds correctly.

**Usage**:
```bash
./scripts/validate-build.sh
```

## Adding New Scripts

When adding new scripts to this directory:

1. **Make it executable**: `chmod +x scripts/your_script.sh`
2. **Add documentation**: Update this README with script purpose and usage
3. **Add error handling**: Use `set -e` in bash scripts, try/except in Python
4. **Add help text**: Support `--help` flag to show usage
5. **Test it**: Verify the script works in a clean environment

## Script Conventions

- **Bash scripts**: Use `.sh` extension, include shebang `#!/bin/bash`
- **Python scripts**: Use `.py` extension, include shebang `#!/usr/bin/env python3`
- **Documentation**: Add docstrings/comments explaining purpose and usage
- **Error messages**: Print errors to stderr, use exit codes appropriately
- **Dependencies**: Document all required tools/packages

## Troubleshooting

### GitHub CLI Not Found
```bash
# Install GitHub CLI
# On Ubuntu/Debian:
sudo apt install gh

# On macOS:
brew install gh

# On Windows:
winget install --id GitHub.cli
```

### Not Authenticated
```bash
# Authenticate with GitHub
gh auth login
```

### Permission Denied
```bash
# Make script executable
chmod +x scripts/your_script.sh
```

## Related Documentation

- [PR Cleanup Report](../docs/PR_CLEANUP_REPORT.md) - Detailed analysis of duplicate PRs
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute to the project
- [Development Guide](../docs/DEVELOPMENT.md) - Development setup and practices
