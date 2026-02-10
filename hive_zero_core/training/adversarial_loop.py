import logging
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.training.config import ExperimentConfig, get_default_config
from hive_zero_core.training.data_loader import NetworkLogDataset
from hive_zero_core.training.rewards import CompositeReward


def train_hive_mind_adversarial(
    config: Optional[ExperimentConfig] = None,
    num_epochs: Optional[int] = None,
):
    """
    Train the HiveMind system using adversarial learning.

    Args:
        config: Experiment configuration. If None, uses default config.
        num_epochs: Number of training epochs. Overrides config if provided.
    """
    # Setup configuration
    if config is None:
        config = get_default_config()
    
    if num_epochs is not None:
        config.training.num_epochs = num_epochs

    # Setup logging
    logger = logging.getLogger("HiveTraining")
    logger.setLevel(getattr(logging, config.log_level))
    
    logger.info("=" * 80)
    logger.info("Starting HiveMind Adversarial Training")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Learning Rate: {config.training.learning_rate}")
    logger.info(f"Batch Size: {config.data.batch_size}")

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    # 1. Initialize HiveMind
    logger.info("Initializing HiveMind...")
    hive = HiveMind(
        observation_dim=config.model.observation_dim,
        pretrained=config.model.pretrained
    )
    
    # Move to device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    logger.info(f"Using device: {device}")
    hive.to(device)

    # 2. Setup data loading
    logger.info("Loading dataset...")
    dataset = NetworkLogDataset(
        data_source=config.data.data_source,
        batch_size=config.data.batch_size,
        synthetic=config.data.synthetic,
        num_synthetic_samples=config.data.num_synthetic_samples,
    )
    logger.info(f"Dataset size: {len(dataset)} samples, {dataset.num_batches} batches")

    # 3. Optimizer
    # Only optimize Gating Network + Learnable Experts (excluding Sentinel/PayloadGen)
    # Sentinel and PayloadGen are pre-trained/frozen
    logger.info("Setting up optimizer...")
    
    params = list(hive.gating_network.parameters())
    # Add expert parameters for trainable experts
    for expert in hive.experts:
        if expert.name == "Mutator":
            # Mutator optimizes at inference time (gradient-based search in embedding space)
            # No additional trainable parameters needed in the outer loop
            # The Sentinel and PayloadGen within Mutator are already pre-trained/frozen
            continue
        elif expert.name in ["Cartographer", "DeepScope", "Mimic", "Ghost", 
                             "Stego", "Cleaner", "Chronos"]:
            params.extend(list(expert.parameters()))

    if config.training.optimizer == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        optimizer = optim.SGD(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

    # 4. Learning rate scheduler (optional)
    scheduler = None
    if config.training.use_lr_scheduler:
        if config.training.lr_scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs
            )
        elif config.training.lr_scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )

    # 5. Reward Function
    reward_calc = CompositeReward()

    # 6. Training Loop
    logger.info("Starting training loop...")
    for epoch in range(config.training.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Iterate through batches
        for batch_idx, batch_logs in enumerate(dataset):
            optimizer.zero_grad()

            # --- Forward Pass ---
            results = hive.forward(batch_logs, top_k=config.model.top_k_experts)

            # --- Loss Calculation ---
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Adversarial Component
            if "defense_score" in results:
                # Score is P(Allowed)
                score = results["defense_score"]
                # We want to MAXIMIZE score. Loss = -Score
                loss_adv = -torch.mean(score) * config.training.adversarial_weight
                total_loss = total_loss + loss_adv

            # Payload Auxiliary Loss
            if "optimized_payload" in results:
                payload = results["optimized_payload"]
                
                # L2 regularization to prevent explosion
                l2_reg = torch.mean(torch.pow(payload, 2)) * config.training.l2_regularization
                
                # Diversity loss: Maximize variance to prevent mode collapse
                payload_var = torch.var(payload, dim=-1).mean()
                diversity_loss = -torch.log(payload_var + 1e-8) * config.training.diversity_weight
                
                total_loss = total_loss + l2_reg + diversity_loss

            # Information Gain Component
            if "topology" in results:
                # NOTE: In production, track actual entropy changes
                # For now, use placeholder values
                current_entropy = 0.5  # Would be computed from actual topology
                prev_entropy = 0.8  # Would be tracked from previous state
                r_info = reward_calc.calculate_info_gain_reward(prev_entropy, current_entropy)
                # Maximize reward -> Minimize -Reward
                total_loss = total_loss - r_info * config.training.info_gain_weight

            # --- Backward Pass ---
            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(params, max_norm=config.training.gradient_clip_norm)

            optimizer.step()

            # Track loss
            epoch_loss += total_loss.item()
            num_batches += 1

            # Log batch progress
            if batch_idx % config.log_frequency == 0:
                logger.debug(
                    f"Epoch {epoch}/{config.training.num_epochs}, "
                    f"Batch {batch_idx}/{dataset.num_batches}, "
                    f"Loss: {total_loss.item():.4f}"
                )

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(
            f"Epoch {epoch}/{config.training.num_epochs} complete. "
            f"Average Loss: {avg_loss:.4f}"
        )

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
            logger.debug(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if (epoch + 1) % config.training.save_frequency == 0:
            checkpoint_path = (
                config.training.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            )
            save_checkpoint(hive, optimizer, epoch, avg_loss, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)

    return hive


def save_checkpoint(
    model: HiveMind,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
):
    """Save a training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(
    model: HiveMind,
    optimizer: optim.Optimizer,
    path: Path,
) -> int:
    """
    Load a training checkpoint.
    
    Returns:
        The epoch number from the checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Quick test with 2 epochs
    from hive_zero_core.training.config import get_quick_test_config
    config = get_quick_test_config()
    train_hive_mind_adversarial(config=config)

