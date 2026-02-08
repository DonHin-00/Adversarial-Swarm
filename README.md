# Adversarial Swarm (HIVE-ZERO)

A Hierarchical Multi-Agent Reinforcement Learning (H-MARL) system for advanced adversarial network simulation and traffic generation.

## 🏗️ Architecture: HIVE-ZERO

HIVE-ZERO utilizes a **Sparse Mixture-of-Experts (MoE)** architecture where a central **HiveMind** Gating Network dynamically routes observations to specialized agents based on the current environment state.

### 🧠 HiveMind Gating Network
- **Noisy Gating:** Adds learnable noise to gating logits to encourage exploration during training.
- **Top-K Selection:** Activates only the most relevant experts (default $k=3$) to ensure computational efficiency and behavioral sparsity.

### 🛡️ Specialized Experts (The 10 Specialists)

1.  **Agent_Cartographer (Recon):** Temporal Graph Attention Network (T-GAT). Uses `GATv2Conv` and `GRU` to reason about evolving network topologies.
2.  **Agent_DeepScope (Recon):** Prioritization engine for identifying high-value targets within the network graph.
3.  **Agent_Chronos (Recon):** Transformer-based Forecaster. Uses causal masking to predict packet inter-arrival times and optimize timing.
4.  **Agent_PayloadGen (Attack):** RAG-enhanced Generator. Retrieves exploit templates from a VectorDB to condition T5-based payload generation.
5.  **Agent_Mutator (Attack):** Hybrid Optimizer. Implements Inference-Time Search (ITS) using gradient descent and discrete noise to evade detection.
6.  **Agent_Sentinel (Defense):** Stateful Ensemble Discriminator. Uses a history-aware GRU and multiple heads to model IDS alert thresholds.
7.  **Agent_Mimic (Post):** Conditional VAE-GAN. Generates realistic network traffic shapes conditioned on protocol metadata.
8.  **Agent_Ghost (Post):** Kernel Metadata Analyzer. Identifies optimal hiding spots by analyzing simulated syscall and inode structures.
9.  **Agent_Stego (Post):** Frequency-Domain Steganography. Embeds payloads into DCT coefficients using 2D Fourier transforms.
10. **Agent_Cleaner (Post):** Formal Logic Verifier. Generates cleanup scripts and verifies the restoration of the environment state.

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from hive_zero_core.hive_mind import HiveMind

# Initialize the system
hive = HiveMind(observation_dim=64)

# Process a stream of raw network logs
raw_logs = [{'src_ip': '192.168.1.5', 'dst_ip': '10.0.0.1', 'port': 443, 'proto': 6}]
results = hive.forward(raw_logs, top_k=3)

# Access specialized outputs
if "optimized_payload" in results:
    payload = results["optimized_payload"]
```

## 🧪 Testing
Run the comprehensive test suite to verify agent logic and architectures:
```bash
pytest tests/test_agents.py
```
