# Adversarial-Swarm (HIVE-ZERO)

A Hierarchical Multi-Agent Reinforcement Learning (H-MARL) system implementing adversarial swarm intelligence for network security research.

## Architecture

HIVE-ZERO uses a **Sparse Mixture-of-Experts (MoE)** architecture with a learned gating network that dynamically activates the most relevant experts for each observation.

### Expert Clusters

| Cluster | Experts | Purpose |
|---------|---------|---------|
| **A — Recon** | Cartographer (GAT), DeepScope (Mask), Chronos (LSTM) | Network mapping, constraint enforcement, timing |
| **B — Attack** | Sentinel (BERT), PayloadGen (T5), Mutator (PPO) | Payload classification, generation, and adversarial optimization |
| **C — Post-Exploit** | Mimic (GAN), Ghost, Stego (AE), Cleaner | Traffic shaping, hiding, steganography, cleanup |
| **D — Active Defense** | Tarpit | 20+ trap signatures with attention-weighted fusion |
| **E — Kill Chain** | FeedbackLoop, Flashbang, GlassHouse | Counter-strike synergizers auto-activated with Tarpit |

### Synergy Logic

When the **Tarpit** expert is activated by the gating network, the **Kill Chain** cluster (FeedbackLoop, Flashbang, GlassHouse) is force-enabled to execute a coordinated quad-strike response.

## Project Structure

```
hive_zero_core/
├── __init__.py
├── hive_mind.py              # Master orchestrator with gating network
├── agents/
│   ├── base_expert.py        # Abstract expert base class with gating logic
│   ├── recon_experts.py      # Cartographer, DeepScope, Chronos
│   ├── attack_experts.py     # Sentinel, PayloadGen, Mutator
│   ├── defense_experts.py    # Tarpit with TrapArsenal
│   ├── post_experts.py       # Mimic, Ghost, Stego, Cleaner
│   └── offensive_defense.py  # FeedbackLoop, Flashbang, GlassHouse
├── memory/
│   ├── foundation.py         # Synthetic experience generator and weight init
│   └── graph_store.py        # Network log → PyG graph encoder
├── training/
│   ├── adversarial_loop.py   # Main training loop
│   └── rewards.py            # Composite reward function (R_adv + R_info + R_stealth)
└── utils/
    └── logging_config.py     # Logging configuration
```

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Usage

```python
from hive_zero_core.hive_mind import HiveMind

# Initialize with pretrained knowledge bootstrap
hive = HiveMind(observation_dim=64, pretrained=True)

# Process network logs
logs = [
    {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5", "port": 80, "proto": 6},
    {"src_ip": "10.0.0.5", "dst_ip": "8.8.8.8", "port": 53, "proto": 17},
]
results = hive.forward(logs, top_k=3)
```

## Training

```python
from hive_zero_core.training.adversarial_loop import train_hive_mind_adversarial

train_hive_mind_adversarial(num_epochs=10)
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.3
- Transformers ≥ 4.30
- See `requirements.txt` for full list
