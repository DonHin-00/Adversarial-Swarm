# Known Limitations and Future Work

This document tracks known limitations, incomplete features, and planned improvements for Adversarial-Swarm.

## Current Limitations

### 1. Data Loading

**Status**: Synthetic data only

The current implementation generates synthetic network logs for training. Real log file parsing is not yet implemented.

**Workaround**: Provide logs as a list of dictionaries to `NetworkLogDataset(data_source=your_logs)`

**Future Work**:
- [ ] Implement CSV log file parser
- [ ] Implement JSON log file parser
- [ ] Add PCAP file support
- [ ] Add streaming data support

### 2. Transformer Models (Sentinel & PayloadGen)

**Status**: Require network access for initial download

Tests that use Sentinel and PayloadGen will fail in sandboxed environments without network access because they need to download pre-trained models from HuggingFace.

**Workaround**: 
- Set `pretrained=False` when initializing HiveMind
- Pre-download models before running in sandboxed environments
- Use mocked tests that skip these experts

**Affected Files**:
- `hive_zero_core/agents/attack_experts.py` (Agent_Sentinel, Agent_PayloadGen)
- `tests/test_basic.py` (test_hive_mind_initialization, test_hive_mind_forward_pass)

**Future Work**:
- [ ] Add offline model support
- [ ] Include pre-trained models in repository (if licensing permits)
- [ ] Add model caching system
- [ ] Create lightweight mock versions for testing

### 3. Entropy Calculation

**Status**: Uses placeholder values

The information gain reward currently uses hardcoded placeholder values for entropy instead of computing actual entropy from network topology.

**Location**: `hive_zero_core/training/adversarial_loop.py:181-185`

```python
# NOTE: In production, track actual entropy changes
# For now, use placeholder values
current_entropy = 0.5  # Would be computed from actual topology
prev_entropy = 0.8  # Would be tracked from previous state
```

**Future Work**:
- [ ] Implement graph entropy calculation from network topology
- [ ] Track entropy changes across training steps
- [ ] Add entropy-based metrics to training logs

### 4. Real-Time Training Metrics

**Status**: Basic logging only

Current training only logs loss values. No support for TensorBoard, Weights & Biases, or other experiment tracking tools.

**Future Work**:
- [ ] Add TensorBoard integration
- [ ] Add Weights & Biases support
- [ ] Track additional metrics (expert usage, gradient norms, etc.)
- [ ] Add visualization utilities

### 5. Distributed Training

**Status**: Single-GPU only

The training system currently supports single-GPU or CPU training. Multi-GPU and distributed training are not implemented.

**Future Work**:
- [ ] Add DataParallel support
- [ ] Add DistributedDataParallel support
- [ ] Implement gradient accumulation
- [ ] Add mixed-precision training

### 6. Expert Optimization

**Status**: Partial implementation

Some experts have simplified implementations that could be enhanced:

**Agent_Sentinel** (Line 72-78 in attack_experts.py):
- Uses fallback padding/slicing for dimension mismatches
- Could use learned projection layers instead

**Agent_Tarpit** (Line 125-129 in defense_experts.py):
- Attention weights artificially clamped with boost
- Could use more sophisticated trap selection strategy

**Agent_Mutator** (Line 195 in attack_experts.py):
- Only 2 optimization steps for stability
- Could be increased for better payload optimization

**Future Work**:
- [ ] Add learned dimension projectors to Sentinel
- [ ] Implement advanced trap selection for Tarpit
- [ ] Increase Mutator optimization steps with proper convergence checks
- [ ] Add PPO policy network to Mutator (as mentioned in comments)

### 7. Graph Processing

**Status**: Simplified

The current graph encoding always averages to a global state vector, not fully utilizing the graph structure.

**Location**: `hive_zero_core/hive_mind.py:121-123`

**Future Work**:
- [ ] Implement graph-level readout with attention
- [ ] Add hierarchical graph pooling
- [ ] Support multiple graph structures simultaneously
- [ ] Add graph coarsening for efficiency

### 8. Model Deployment

**Status**: Not implemented

No inference-only deployment mode or model export functionality.

**Future Work**:
- [ ] Add inference-only mode
- [ ] Implement model export (ONNX, TorchScript)
- [ ] Create deployment container
- [ ] Add model serving API
- [ ] Implement batch inference

### 9. Hyperparameter Optimization

**Status**: Manual tuning only

No automatic hyperparameter search or optimization.

**Future Work**:
- [ ] Add Optuna integration for hyperparameter search
- [ ] Implement learning rate finder
- [ ] Add automated batch size selection
- [ ] Create hyperparameter search presets

### 10. Testing Coverage

**Status**: Good but incomplete

While 55+ tests cover major components, some areas need more coverage:

- Integration tests for full training pipeline
- Tests for checkpoint save/load with actual models
- Tests for learning rate schedulers
- Tests for edge cases in expert implementations
- Performance/benchmark tests

**Future Work**:
- [ ] Add end-to-end integration tests
- [ ] Add performance benchmarks
- [ ] Add stress tests for large-scale data
- [ ] Add tests for all configuration combinations

## Design Decisions & Rationale

### Why Sparse MoE with Top-K Selection?

**Rationale**: Computational efficiency while maintaining expert diversity. Activating all 14 experts every forward pass would be expensive. Top-K (default 3) provides good balance.

**Trade-off**: Some experts may be underutilized during training.

### Why Freeze Sentinel and PayloadGen?

**Rationale**: Pre-trained transformer models are already effective at their tasks (classification/generation). Fine-tuning them requires significant resources and may not improve performance significantly.

**Trade-off**: These experts can't adapt to domain-specific patterns during training.

### Why Synthetic Data Generation?

**Rationale**: Allows immediate testing without requiring real network logs. Helps development and testing in sandboxed environments.

**Trade-off**: Synthetic patterns may not represent real attack scenarios accurately.

### Why Separate Configuration Classes?

**Rationale**: Modularity and clarity. ModelConfig, TrainingConfig, and DataConfig can be developed and tested independently. Also allows easy preset creation.

**Trade-off**: Slightly more verbose than a single config dictionary.

## Performance Considerations

### Memory Usage

**Current**: ~2-4GB GPU memory for default configuration (observation_dim=64, batch_size=32, top_k=3)

**Factors**:
- Transformer models (Sentinel, PayloadGen) are largest contributors
- Graph neural networks scale with number of nodes/edges
- Batch size has linear impact

**Optimization tips**:
- Reduce batch size if OOM
- Reduce observation_dim and action_dim
- Use CPU for development/testing
- Reduce top_k to activate fewer experts

### Training Speed

**Current**: ~1-2 seconds per epoch with synthetic data (1000 samples, batch_size=32) on CPU

**Factors**:
- Mutator inference-time optimization is slowest (gradient-based search)
- Graph encoding overhead
- Transformer forward passes

**Optimization tips**:
- Use GPU for training
- Increase batch size (better GPU utilization)
- Reduce Mutator optimization steps
- Use simpler experts during development

## Contributing to Fix Limitations

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute fixes for these limitations.

When working on any of these items:
1. Check if someone else is already working on it (check issues/PRs)
2. Create an issue to discuss the approach
3. Reference this document in your PR
4. Update this document when the limitation is fixed
