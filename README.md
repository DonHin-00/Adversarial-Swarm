# Adversarial-Swarm

[![Python CI](https://github.com/DonHin-00/Adversarial-Swarm/actions/workflows/python-ci.yml/badge.svg)](https://github.com/DonHin-00/Adversarial-Swarm/actions/workflows/python-ci.yml)
[![Security Scan](https://github.com/DonHin-00/Adversarial-Swarm/actions/workflows/security-scan.yml/badge.svg)](https://github.com/DonHin-00/Adversarial-Swarm/actions/workflows/security-scan.yml)

My first adversarial arrangement I plan to keep and progressively upgrade :)

## Overview

Adversarial-Swarm is a sophisticated multi-expert architecture system featuring 14 specialized agents organized into distinct operational clusters for reconnaissance, attack, post-exploitation, and active defense capabilities.

## Features

- **Multi-Expert Architecture**: 14 specialized agents working in coordination
- **Modular Design**: Organized into functional clusters (Recon, Attack, Post-Exploit, Defense, Kill Chain)
- **Graph-Based Memory**: Sophisticated knowledge representation using PyTorch Geometric
- **Dynamic Gating**: Intelligent expert selection based on operational context
- **Active Defense**: Integrated counter-strike and defensive capabilities

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- See `requirements.txt` for full dependencies

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/DonHin-00/Adversarial-Swarm.git
cd Adversarial-Swarm

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
make pre-commit

# Or manually:
pip install pre-commit
pre-commit install
```

### Using Make

```bash
# See all available commands
make help

# Common commands:
make install        # Install production dependencies
make install-dev    # Install development dependencies
make test          # Run tests
make lint          # Run linters
make format        # Format code
make clean         # Clean build artifacts
```

### Docker Setup

```bash
# Build the Docker image
make docker-build

# Or manually:
docker-compose build

# Run the container
docker-compose up -d

# For development:
docker-compose run dev bash
```

## Usage

```python
from hive_zero_core.hive_mind import HiveMind

# Initialize the Hive Mind
hive = HiveMind(observation_dim=64, pretrained=False)

# Process logs and get expert recommendations
raw_logs = [
    {"timestamp": "2024-01-01", "event": "connection", "source_ip": "192.168.1.1"},
    {"timestamp": "2024-01-01", "event": "request", "source_ip": "192.168.1.2"},
]

results = hive.forward(raw_logs, top_k=3)
```

## Architecture

### Expert Clusters

1. **Cluster A: Reconnaissance**
   - Agent_Cartographer: Network topology mapping
   - Agent_DeepScope: Deep constraint analysis
   - Agent_Chronos: Temporal analysis

2. **Cluster B: Attack**
   - Agent_Sentinel: Defense detection
   - Agent_PayloadGen: Payload generation
   - Agent_Mutator: Adaptive payload mutation

3. **Cluster C: Post-Exploitation**
   - Agent_Mimic: Traffic shaping and mimicry
   - Agent_Ghost: Stealth and hiding
   - Agent_Stego: Covert channel creation
   - Agent_Cleaner: Evidence removal

4. **Cluster D: Active Defense**
   - Agent_Tarpit: Attacker engagement and delay

5. **Cluster E: Kill Chain**
   - Agent_FeedbackLoop: Counter-strike coordination
   - Agent_Flashbang: Overwhelming response
   - Agent_GlassHouse: Total exposure tactics

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hive_zero_core --cov-report=html

# Or using make
make test
```

### Code Quality

```bash
# Run all linters
make lint

# Format code
make format

# Individual tools
flake8 hive_zero_core
pylint hive_zero_core
black hive_zero_core
isort hive_zero_core
mypy hive_zero_core
```

## CI/CD

The project includes comprehensive CI/CD pipelines:

- **Python CI**: Automated testing, linting, and code quality checks
- **Security Scan**: CodeQL and Semgrep security analysis
- **Dependency Management**: Automated dependency updates via Dependabot

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with PyTorch, PyTorch Geometric, and other excellent open-source tools.
 
