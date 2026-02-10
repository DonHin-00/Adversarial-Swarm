# Training Guide for Adversarial-Swarm

This guide explains how to train the HiveMind system using the adversarial training loop.

## Quick Start

The simplest way to start training is using the default configuration:

```python
from hive_zero_core.training import train_hive_mind_adversarial, get_quick_test_config

# For a quick test (2 epochs, 100 samples)
config = get_quick_test_config()
trained_hive = train_hive_mind_adversarial(config=config)
```

## Configuration System

The training system uses a hierarchical configuration approach with four main components:

### 1. ModelConfig

Controls the HiveMind model architecture:

```python
from hive_zero_core.training import ModelConfig

model_config = ModelConfig(
    observation_dim=64,      # Dimension of input observations
    action_dim=128,          # Dimension of expert outputs
    hidden_dim=64,           # Hidden layer dimensions
    num_experts=14,          # Number of expert agents
    pretrained=False,        # Whether to use pretrained models
    bert_model="prajjwal1/bert-tiny",  # BERT model for Sentinel
    t5_model="t5-small",     # T5 model for PayloadGen
    gating_hidden_dim=128,   # Gating network hidden size
    top_k_experts=3,         # Number of experts to activate per forward pass
)
```

### 2. TrainingConfig

Controls training hyperparameters:

```python
from hive_zero_core.training import TrainingConfig

training_config = TrainingConfig(
    num_epochs=10,
    learning_rate=0.001,
    batch_size=32,
    
    # Loss component weights
    adversarial_weight=1.0,
    info_gain_weight=0.5,
    stealth_weight=0.3,
    
    # Auxiliary loss weights
    l2_regularization=0.01,
    diversity_weight=0.1,
    
    # Optimization
    optimizer="adam",              # "adam", "sgd", or "adamw"
    gradient_clip_norm=1.0,
    weight_decay=0.0001,
    
    # Learning rate scheduling
    use_lr_scheduler=True,
    lr_scheduler_type="cosine",    # "cosine", "step", or "exponential"
    lr_warmup_epochs=2,
    
    # Checkpointing
    save_frequency=5,              # Save every N epochs
    keep_last_n_checkpoints=3,
)
```

### 3. DataConfig

Controls data loading:

```python
from hive_zero_core.training import DataConfig

data_config = DataConfig(
    data_source=None,              # Path to log file or None for synthetic
    batch_size=32,
    synthetic=True,                # Generate synthetic data
    num_synthetic_samples=1000,    # Number of synthetic samples
    shuffle=True,
    num_workers=0,
)
```

### 4. ExperimentConfig

Combines all configurations:

```python
from hive_zero_core.training import ExperimentConfig

config = ExperimentConfig(
    model=model_config,
    training=training_config,
    data=data_config,
    experiment_name="my_experiment",
    output_dir="outputs",
    seed=42,
    device="auto",  # "auto", "cuda", or "cpu"
    log_level="INFO",
)
```

## Training Examples

### Example 1: Quick Test Run

For quick testing and debugging:

```python
from hive_zero_core.training import train_hive_mind_adversarial, get_quick_test_config

config = get_quick_test_config()
# 2 epochs, 100 synthetic samples, saves checkpoint after 1 epoch
trained_hive = train_hive_mind_adversarial(config=config)
```

### Example 2: Full Training Run

For a complete training session:

```python
from hive_zero_core.training import train_hive_mind_adversarial, get_full_training_config

config = get_full_training_config()
# 100 epochs, 10,000 synthetic samples, LR scheduling enabled
trained_hive = train_hive_mind_adversarial(config=config)
```

### Example 3: Custom Configuration

For complete control:

```python
from hive_zero_core.training import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    train_hive_mind_adversarial,
)

# Customize model
model_config = ModelConfig(
    observation_dim=128,
    action_dim=256,
    top_k_experts=5,
)

# Customize training
training_config = TrainingConfig(
    num_epochs=50,
    learning_rate=0.0005,
    batch_size=64,
    adversarial_weight=1.5,
    use_lr_scheduler=True,
)

# Customize data
data_config = DataConfig(
    batch_size=64,
    num_synthetic_samples=5000,
)

# Combine
config = ExperimentConfig(
    model=model_config,
    training=training_config,
    data=data_config,
    experiment_name="custom_training",
)

trained_hive = train_hive_mind_adversarial(config=config)
```

### Example 4: Resume from Checkpoint

```python
import torch
from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.training import load_checkpoint, train_hive_mind_adversarial
from pathlib import Path

# Load checkpoint
hive = HiveMind(observation_dim=64)
optimizer = torch.optim.Adam(hive.parameters())
checkpoint_path = Path("checkpoints/checkpoint_epoch_10.pt")

start_epoch = load_checkpoint(hive, optimizer, checkpoint_path)
print(f"Resumed from epoch {start_epoch}")

# Continue training
config = get_quick_test_config()
config.training.num_epochs = 20  # Train for 10 more epochs
trained_hive = train_hive_mind_adversarial(config=config)
```

## Data Loading

### Synthetic Data (Default)

By default, the system generates synthetic network logs:

```python
from hive_zero_core.training import NetworkLogDataset

dataset = NetworkLogDataset(
    synthetic=True,
    num_synthetic_samples=1000,
    batch_size=32,
)

# Iterate through batches
for batch in dataset:
    print(f"Batch size: {len(batch)}")
    # Each batch is a list of log dictionaries
```

### Custom Data

To use your own log data:

```python
# From a list
my_logs = [
    {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5", "port": 80, "proto": 6},
    {"src_ip": "192.168.1.2", "dst_ip": "10.0.0.6", "port": 443, "proto": 6},
    # ... more logs
]

dataset = NetworkLogDataset(
    data_source=my_logs,
    batch_size=32,
)

# Or from a file (future implementation)
dataset = NetworkLogDataset(
    data_source="path/to/logs.csv",
    batch_size=32,
)
```

## Loss Functions

The training loop uses a composite loss with three components:

### 1. Adversarial Loss

Maximizes the probability that generated payloads evade detection:

```
L_adv = -mean(P(Allowed))
```

Weight controlled by `training_config.adversarial_weight`

### 2. Auxiliary Loss (Payload Quality)

Ensures payloads have desirable properties:

```
L_aux = L2_reg + Diversity_loss
L2_reg = mean(payload²) * l2_regularization
Diversity_loss = -log(var(payload) + ε) * diversity_weight
```

### 3. Information Gain Loss

Rewards reduction in network state entropy:

```
L_info = -(prev_entropy - current_entropy) * info_gain_weight
```

Total loss: `L_total = L_adv + L_aux + L_info`

## Trainable Components

The training loop optimizes:

1. **Gating Network**: Learns which experts to activate for each input
2. **Recon Experts**: Cartographer, DeepScope, Chronos
3. **Post-Exploit Experts**: Mimic, Ghost, Stego, Cleaner

**Not trained** (frozen or inference-time optimization):
- Sentinel and PayloadGen (pre-trained transformers)
- Mutator (uses gradient-based search at inference time)
- Defense experts (Tarpit, FeedbackLoop, Flashbang, GlassHouse)

## Monitoring Training

The training loop logs progress at configurable intervals:

```python
config = get_default_config()
config.log_level = "INFO"      # "DEBUG", "INFO", "WARNING", "ERROR"
config.log_frequency = 10      # Log every N batches

trained_hive = train_hive_mind_adversarial(config=config)
```

Example output:
```
INFO: Starting HiveMind Adversarial Training
INFO: Experiment: adversarial_swarm_experiment
INFO: Epochs: 10
INFO: Using device: cuda
INFO: Dataset size: 1000 samples, 32 batches
INFO: Epoch 0/10 complete. Average Loss: 0.8234
INFO: Checkpoint saved to checkpoints/checkpoint_epoch_5.pt
INFO: Training complete!
```

## Best Practices

1. **Start Small**: Use `get_quick_test_config()` to verify your setup
2. **Monitor Loss**: Watch for divergence or NaN values
3. **Adjust Weights**: Balance adversarial, info gain, and stealth components
4. **Use Checkpoints**: Save regularly to avoid losing progress
5. **Experiment**: Try different learning rates, batch sizes, and expert selections

## Troubleshooting

### Loss is NaN
- Reduce learning rate
- Increase gradient clipping norm
- Check input data for extreme values

### Training is too slow
- Reduce `num_synthetic_samples`
- Increase `batch_size`
- Use GPU (set `device="cuda"`)
- Reduce `top_k_experts`

### Out of memory
- Reduce `batch_size`
- Reduce `observation_dim` or `action_dim`
- Use CPU (set `device="cpu"`)

### Experts not learning
- Increase learning rate
- Reduce regularization weights
- Ensure experts are in trainable list
- Check that loss has gradients

## Advanced Topics

### Custom Reward Functions

Modify `hive_zero_core/training/rewards.py` to implement custom reward logic.

### Custom Experts

Add new experts by:
1. Inheriting from `BaseExpert`
2. Implementing `_forward_impl()`
3. Adding to expert list in training loop

### Distributed Training

For multi-GPU training, wrap the model:

```python
import torch.nn as nn

hive = HiveMind(observation_dim=64)
if torch.cuda.device_count() > 1:
    hive = nn.DataParallel(hive)
```

## Reference

For more details, see:
- `hive_zero_core/training/config.py` - Configuration classes
- `hive_zero_core/training/adversarial_loop.py` - Main training loop
- `hive_zero_core/training/data_loader.py` - Data loading utilities
- `hive_zero_core/training/rewards.py` - Reward functions
