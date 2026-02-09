# HIVE-ZERO: Deep H-MARL Architecture

## Overview
HIVE-ZERO is a Hierarchical Multi-Agent Reinforcement Learning system for autonomous cyber-operations. It utilizes a Sparse Mixture-of-Experts (MoE) architecture orchestrated by a Gating Network and a Strategic Planner.

## Core Modules

### Orchestration
- **Strategic Planner**: Sets latent high-level goals (Recon, Infiltrate, Exfiltrate).
- **HiveMind**: The central Gating Network (Noisy Top-K) routing tasks to 10 experts.
- **Safety Monitor**: Real-time "Kill Switch" enforcing RoE and alert thresholds.

### Data Layer
- **HeteroLogEncoder**: Converts raw network logs into dynamic Heterogeneous Graphs (IP, Port, Protocol nodes).
- **Flow Tracking**: Uses GRU cells to maintain stateful connection history.

### Experts (The Swarm)
#### Reconnaissance (Cluster A)
- **Cartographer**: HGT (Heterogeneous Graph Transformer) for topology inference.
- **DeepScope**: Priority-based target selection with hard RoE masking.
- **Chronos**: Fourier-enhanced Transformer for packet timing analysis.

#### Adversarial (Cluster B)
- **PayloadGen**: RAG (Retrieval-Augmented Generation) using T5 and VectorDB.
- **Mutator**: MCTS (Monte Carlo Tree Search) for optimal payload obfuscation.
- **Sentinel**: 3-Head Ensemble BERT for robust intrusion detection simulation.

#### Post-Exploitation (Cluster C)
- **Mimic**: Conditional VAE-GAN for protocol-specific traffic shaping.
- **Ghost**: Kernel metadata entropy analyzer for persistence.
- **Stego**: DCT-based steganography for covert channels.
- **Cleaner**: Formal logic verification for trace removal.

## Infrastructure
- **Gymnasium Env**: Standard RL interface for the Cyber Range.
- **Network Digital Twin**: NetworkX-based simulation of 3-tier corporate networks.
- **FastAPI Interface**: C2 Server for remote command execution.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest tests/`
3. Start C2 Server: `python hive_zero_core/api/server.py`
