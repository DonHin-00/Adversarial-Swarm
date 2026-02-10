# Adversarial Swarm (HIVE-ZERO)

A Hierarchical Multi-Agent Reinforcement Learning (H-MARL) system for advanced adversarial network simulation and traffic generation.

## 🏗️ Architecture: HIVE-ZERO

HIVE-ZERO utilizes a **Sparse Mixture-of-Experts (MoE)** architecture where a central **HiveMind** Gating Network dynamically routes observations to specialized agents based on the current environment state.

### 🧠 HiveMind Gating Network
- **Noisy Gating:** Adds learnable noise to gating logits to encourage exploration during training.
- **Top-K Selection:** Activates only the most relevant experts (default $k=3$) to ensure computational efficiency and behavioral sparsity.

### 🛡️ Specialized Experts (The 10 Specialists)

1.  **CartographerAgent (Recon):** Temporal Graph Attention Network (T-GAT). Uses `GATv2Conv` and `GRU` to reason about evolving network topologies.
2.  **DeepScopeAgent (Recon):** Prioritization engine for identifying high-value targets within the network graph.
3.  **ChronosAgent (Recon):** Transformer-based Forecaster. Uses causal masking to predict packet inter-arrival times and optimize timing.
4.  **PayloadGenAgent (Attack):** RAG-enhanced Generator. Retrieves exploit templates from a VectorDB to condition T5-based payload generation.
5.  **MutatorAgent (Attack):** Hybrid Optimizer. Implements Inference-Time Search (ITS) using gradient descent and discrete noise to evade detection.
6.  **SentinelAgent (Defense):** Stateful Ensemble Discriminator. Uses a history-aware GRU and multiple heads to model IDS alert thresholds.
7.  **MimicAgent (Post):** Conditional VAE-GAN. Generates realistic network traffic shapes conditioned on protocol metadata.
8.  **GhostAgent (Post):** Kernel Metadata Analyzer. Identifies optimal hiding spots by analyzing simulated syscall and inode structures.
9.  **StegoAgent (Post):** Frequency-Domain Steganography. Embeds payloads into DCT coefficients using 2D Fourier transforms.
10. **CleanerAgent (Post):** Formal Logic Verifier. Generates cleanup scripts and verifies the restoration of the environment state.

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

## 🎓 Advanced Features

### Skill Knowledge System
Agents now have proficiency levels (Novice, Intermediate, Expert, Master) and track skill usage:
```python
print(f"Skill Level: {agent.skill_level.name}")
print(f"Effectiveness: {agent.get_effectiveness_score()}")
```

### Threat Intelligence Store
Centralized knowledge management with skill taxonomy and knowledge sharing:
```python
from hive_zero_core.memory.threat_intelligence import ThreatIntelligenceStore
kb = ThreatIntelligenceStore()
```

### Advanced Reward System
8 affect types for comprehensive reward shaping:
- Adversarial, Information, Stealth, Temporal
- Resource, Reliability, Novelty, Coordination

```python
from hive_zero_core.training.advanced_rewards import AdvancedCompositeReward
rewards = AdvancedCompositeReward().compute(...)
```

See [Skill Knowledge & Affects Documentation](docs/SKILL_KNOWLEDGE_AFFECTS.md) for details.
