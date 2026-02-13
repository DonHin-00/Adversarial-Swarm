"""
HIVE-ZERO: Hierarchical Multi-Agent Reinforcement Learning system with adversarial swarm intelligence for network security research.

## Architecture Overview

HIVE-ZERO implements a Sparse Mixture-of-Experts (MoE) architecture with 19 specialized
expert agents organized into 7 functional clusters:

- **agents/**: 19 expert agents implementing reconnaissance, attack, post-exploitation, 
  defense, and offensive capabilities
- **memory/**: Graph-based network topology storage, threat intelligence database, 
  and knowledge foundation
- **security/**: Cryptographic utilities, input validation, audit logging, and access control
- **training/**: Adversarial co-evolution training loop, configuration management, 
  and data loading
- **mitre/**: MITRE ATT&CK and ATLAS technique mapping and integration
- **data/**: Advanced parsers for CSV, JSON, PCAP, and streaming log formats
- **utils/**: Logging configuration and common utilities

## Key Features

- Red/blue adversarial co-evolution with persistent threat intelligence
- Gating network for dynamic expert activation (top-k routing)
- 14 defensive and offensive expert clusters
- MITRE ATT&CK and ATLAS integration (145 techniques)
- Comprehensive security infrastructure with audit logging
- Genetic evolution and polymorphic payload generation
"""

__version__ = "0.1.0"
