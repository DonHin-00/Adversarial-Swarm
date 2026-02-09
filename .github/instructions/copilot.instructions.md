# GitHub Copilot Instructions for Adversarial-Swarm

## Project Overview

### Identity
- **Name**: Adversarial-Swarm (Hive Zero Core)
- **Purpose**: Advanced AI-driven adversarial security system using Mixture of Experts (MoE) architecture
- **Architecture**: 14 specialized expert agents coordinated by a HiveMind controller with a Gating Network
- **Domain**: Cybersecurity, Adversarial AI, Network Security, Penetration Testing Automation

### Core Concept
This system implements a sophisticated Mixture of Experts architecture where specialized AI agents collaborate to perform security operations across the full attack lifecycle - from reconnaissance through post-exploitation and active defense.

---

## Expert Agent Clusters

The system contains **14 specialized expert agents** organized into **5 functional clusters**:

### Cluster A: Reconnaissance
1. **Agent_Cartographer**
   - Role: Network topology mapper
   - Technology: Graph Neural Network (GNN)
   - Function: Builds and maintains graph representations of network infrastructure
   - Output: Network topology graphs with nodes (hosts) and edges (connections)

2. **Agent_DeepScope**
   - Role: Deep packet inspector
   - Technology: Neural network with 10 discrete actions
   - Function: Analyzes network traffic at packet level
   - Actions: Packet classification, protocol analysis, anomaly detection

3. **Agent_Chronos**
   - Role: Temporal analyzer
   - Technology: Time-series analysis with neural networks
   - Function: Detects time-based patterns in network behavior
   - Use Case: Identifying periodic beaconing, scheduled tasks, temporal anomalies

### Cluster B: Attack
4. **Agent_Sentinel**
   - Role: Payload discriminator
   - Technology: BERT-based transformer (BERT-tiny)
   - Function: Classifies payloads as Blocked/Allowed
   - Purpose: Acts as adversarial judge for payload effectiveness
   - Training: Fine-tuned on security filter datasets

5. **Agent_PayloadGen**
   - Role: Exploit string generator
   - Technology: T5-based sequence-to-sequence model (T5-small)
   - Function: Generates exploit payloads from attack specifications
   - Approach: Text-to-text generation conditioned on target vulnerabilities
   - Output: Obfuscated, mutation-resistant attack strings

6. **Agent_Mutator**
   - Role: Evolutionary payload optimizer
   - Technology: Genetic algorithm + reinforcement learning
   - Function: Combines Sentinel and PayloadGen in adversarial loop
   - Process: Generate → Test → Mutate → Repeat until evasion achieved
   - Fitness: Maximize evasion probability (1 - Sentinel score)

### Cluster C: Post-Exploitation
7. **Agent_Mimic**
   - Role: Traffic shape mimicry
   - Technology: Flow-GAN (Generative Adversarial Network)
   - Function: Generates network traffic that mimics legitimate patterns
   - Goal: Minimize detection by behavioral analysis systems
   - Training: Adversarial against traffic classifiers

8. **Agent_Ghost**
   - Role: System entropy minimizer
   - Technology: Reinforcement learning with system state modeling
   - Function: Hides files and processes by minimizing system footprint
   - Metrics: File system entropy, process tree anomalies, registry changes

9. **Agent_Stego**
   - Role: Steganographic data embedding
   - Technology: Neural autoencoder
   - Function: Embeds data into cover media (images, network packets)
   - Architecture: Encoder-Decoder with adversarial loss
   - Use Case: Covert data exfiltration

10. **Agent_Cleaner**
    - Role: State restoration and log sanitization
    - Technology: Causal inference module
    - Function: Reverses system changes to pre-compromise state
    - Approach: Tracks causal relationships between actions
    - Output: Minimal set of operations to restore clean state

### Cluster D: Active Defense
11. **Agent_Tarpit**
    - Role: Resource trap for attackers
    - Technology: Honeypot with adaptive response
    - Function: Wastes attacker resources through deliberate slowdowns
    - Techniques: Connection delays, infinite loops, resource exhaustion

### Cluster E: Kill Chain / Offensive Defense
12. **Agent_FeedbackLoop**
    - Role: Signal amplifier and attack reflector
    - Technology: Recursive amplification network
    - Function: Reflects attacks back to source with amplification
    - Strategy: Turn adversary's actions against them

13. **Agent_Flashbang**
    - Role: Data burst generator
    - Technology: High-throughput packet generator
    - Function: Overwhelms targets with traffic bursts
    - Use Case: Denial of service, defensive flooding

14. **Agent_GlassHouse**
    - Role: Holographic exposure system
    - Technology: Quantum-inspired port state superposition
    - Function: Creates decoy services that expose attacker techniques
    - Approach: Transparent honeypots that document attack methods

---

## Technical Stack

### Core Frameworks
- **PyTorch**: Primary deep learning framework
- **PyTorch Geometric**: Graph neural network operations
- **HuggingFace Transformers**: Pre-trained NLP models (BERT-tiny, T5-small)

### Model Architectures
- **Sparse MoE**: Mixture of Experts with top-k gating
- **Graph Neural Networks**: For network topology processing
- **GANs**: Generative Adversarial Networks for traffic/steganography
- **Autoencoders**: For dimensionality reduction and steganography
- **Transformers**: BERT and T5 for NLP-based attacks

### Training Paradigm
- **Adversarial Training Loop**: Experts compete and cooperate
- **Composite Rewards**: Multi-objective optimization
  - `R_adv`: Maximize evasion probability
  - `R_info`: Maximize information gain
  - `R_stealth`: Minimize detection probability
- **Sparse Activation**: Only top-k experts active per task

---

## Key Components

### Core Modules
1. **`hive_zero_core/hive_mind.py`**
   - Main coordinator class: `HiveMind`
   - Contains `GatingNetwork` for expert selection
   - Implements sparse MoE routing logic
   - Manages expert activation and result aggregation

2. **`hive_zero_core/memory/graph_store.py`**
   - `LogEncoder`: Converts network logs to PyTorch Geometric graphs
   - Graph construction from raw traffic data
   - Node/edge feature extraction

3. **`hive_zero_core/memory/foundation.py`**
   - `SyntheticExperienceGenerator`: Creates training data
   - `WeightInitializer`: Smart weight initialization strategies
   - Memory management utilities

4. **`hive_zero_core/training/adversarial_loop.py`**
   - Main training pipeline
   - Implements adversarial game between agents
   - Handles episode management and checkpointing

5. **`hive_zero_core/training/rewards.py`**
   - `CompositeReward` class
   - Implements multi-objective reward functions
   - Balances competing objectives (stealth vs. impact)

6. **`hive_zero_core/agents/base_expert.py`**
   - `BaseExpert` abstract class
   - All experts inherit from this
   - Defines standardized interface

### Agent Modules
- **`recon_experts.py`**: Cartographer, DeepScope, Chronos
- **`attack_experts.py`**: Sentinel, PayloadGen, Mutator
- **`post_experts.py`**: Mimic, Ghost, Stego, Cleaner
- **`defense_experts.py`**: Tarpit
- **`offensive_defense.py`**: FeedbackLoop, Flashbang, GlassHouse

---

## Coding Guidelines

### Expert Implementation
All experts must follow these patterns:

```python
from hive_zero_core.agents.base_expert import BaseExpert

class MyExpert(BaseExpert):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.is_active = True  # For sparse MoE gating
        # Define layers here
        
    def _forward_impl(self, x):
        # Expert-specific logic
        # Must handle input validation
        # Must include error handling
        return output
        
    def forward(self, x):
        # Standardized forward pass from BaseExpert
        if not self.is_active:
            return None
        return self._forward_impl(x)
```

### Design Principles
1. **Inheritance**: All experts inherit from `BaseExpert`
2. **Sparse Activation**: Use `is_active` flag for MoE gating
3. **Standard Interface**: Implement `_forward_impl()` for custom logic
4. **Input Validation**: Always validate tensor shapes and types
5. **Error Handling**: Graceful degradation, never crash the swarm
6. **Dimension Adapters**: Use `ensure_dimension()` for shape mismatches
7. **Logging**: Use configured loggers from `utils/logging_config.py`

### Code Quality Standards
- **Type Hints**: Use Python type hints for all function signatures
- **Docstrings**: Include docstrings for all classes and public methods
- **Error Messages**: Provide informative error messages with context
- **Unit Tests**: Write tests for new expert implementations
- **Performance**: Profile critical paths, optimize hot loops

---

## Reward Functions

The system uses a composite reward to balance competing objectives:

### R_adv: Adversarial Success
- **Formula**: `R_adv = 1 - Sentinel_score`
- **Purpose**: Maximize evasion probability
- **Range**: [0, 1], higher is better
- **Used by**: PayloadGen, Mutator

### R_info: Information Gain
- **Formula**: `R_info = -Entropy(knowledge_graph)`
- **Purpose**: Maximize knowledge extraction
- **Metric**: Reduction in uncertainty over network state
- **Used by**: Cartographer, DeepScope, Chronos

### R_stealth: Stealth Maintenance
- **Formula**: `R_stealth = -KL(generated || baseline)`
- **Purpose**: Minimize divergence from normal traffic
- **Metric**: KL divergence between generated and baseline distributions
- **Used by**: Mimic, Ghost, Stego

### Composite Reward
```python
R_total = α * R_adv + β * R_info + γ * R_stealth
```
Where α, β, γ are hyperparameters balancing objectives.

---

## Data Flow

### Pipeline Overview
1. **Input**: Raw network logs, PCAP files, system events
2. **Encoding**: `LogEncoder` → PyTorch Geometric `Data` objects
3. **Routing**: `HiveMind.forward()` → `GatingNetwork` selects top-k experts
4. **Execution**: Selected experts process in parallel
5. **Aggregation**: Results combined via weighted average
6. **Reward**: Composite reward calculated
7. **Learning**: Backpropagation through trainable experts

### Detailed Flow
```
Raw Logs 
  → LogEncoder.encode() 
  → PyG Graph (nodes, edges, features)
  → HiveMind.forward(graph)
  → GatingNetwork.select_experts(graph.global_features)
  → [Expert_1, Expert_2, ..., Expert_k].forward(graph)
  → weighted_average(expert_outputs, gating_weights)
  → CompositeReward.compute(action, environment)
  → optimizer.step()
```

### Graph Structure
- **Nodes**: Network hosts, processes, files
- **Edges**: Connections, dependencies, causal links
- **Node Features**: IP, ports, services, timestamps
- **Edge Features**: Protocol, byte counts, direction
- **Global Features**: Aggregate statistics for gating

---

## Windows Compatibility Notes

### File Path Restrictions
Windows file systems prohibit certain characters in file names:
- **Forbidden in filenames**: `<`, `>`, `:`, `"`, `/`, `\`, `|`, `?`, `*`
- **Note**: `/` and `\` are reserved as system path separators and cannot appear within filenames
- **Important**: Never manually construct paths with separators - use `pathlib.Path` instead
- **Recommendation**: Use alphanumeric characters, hyphens, underscores, and periods only in file names

### Best Practices
1. **Avoid Wildcards**: Never use `*` or `?` in committed file names
2. **Test on Windows**: Run CI/CD on Windows runners
3. **Cross-Platform Paths**: Always use `pathlib.Path` for all path operations - it handles platform-specific separators automatically
4. **No Manual Path Separators**: Avoid manually constructing paths with `/` or `\` - use `pathlib.Path` methods like `.joinpath()` or the `/` operator
5. **Git Line Endings**: Configure `.gitattributes` for consistent line endings

### GitHub Actions
- Test workflows on multiple OS: `runs-on: [ubuntu-latest, windows-latest, macos-latest]`
- Ensure scripts work with both Unix and Windows shells
- Use `${{ runner.os }}` for OS-specific logic

---

## Development Workflow

### Adding a New Expert
1. Create class in appropriate module (`*_experts.py`)
2. Inherit from `BaseExpert`
3. Implement `__init__()` and `_forward_impl()`
4. Register in `hive_mind.py` expert registry
5. Add to appropriate cluster
6. Write unit tests
7. Update this documentation

### Training a New Model
1. Prepare dataset in graph format
2. Configure hyperparameters in training script
3. Run `adversarial_loop.py` with appropriate args
4. Monitor composite reward convergence
5. Evaluate on held-out test set
6. Save checkpoint for deployment

### Debugging Tips
- Use `logging_config.py` for structured logging
- Enable debug mode: `export DEBUG=1`
- Visualize graphs: Use NetworkX plotting
- Profile slow experts: Use `torch.profiler`
- Check tensor shapes: Add assertions in forward pass

---

## Security Considerations

This is a security research project. Key considerations:

1. **Ethical Use**: Only use in authorized environments
2. **Responsible Disclosure**: Report vulnerabilities found
3. **Isolation**: Run in sandboxed/virtualized environments
4. **No Production**: Not for production security infrastructure
5. **Educational**: Designed for learning and research

---

## Quick Reference

### File Structure
```
hive_zero_core/
├── __init__.py
├── hive_mind.py           # Main coordinator
├── agents/
│   ├── base_expert.py     # Base class
│   ├── recon_experts.py   # Cluster A
│   ├── attack_experts.py  # Cluster B
│   ├── post_experts.py    # Cluster C
│   ├── defense_experts.py # Cluster D
│   └── offensive_defense.py # Cluster E
├── memory/
│   ├── graph_store.py     # Graph encoding
│   └── foundation.py      # Utilities
├── training/
│   ├── adversarial_loop.py # Training
│   └── rewards.py          # Reward functions
└── utils/
    └── logging_config.py   # Logging setup
```

### Common Commands
```bash
# Train the system
python -m hive_zero_core.training.adversarial_loop --config config.yaml

# Evaluate an expert
python -m hive_zero_core.agents.test_expert --expert Agent_PayloadGen

# Visualize network graph
python -m hive_zero_core.memory.visualize_graph --input logs.pcap
```

---

## Additional Resources

- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **MITRE ATT&CK**: https://attack.mitre.org/
- **Adversarial ML**: https://adversarial-ml-tutorial.org/

---

*This document provides comprehensive context for GitHub Copilot to assist with development of the Adversarial-Swarm project. Keep it updated as the system evolves.*
