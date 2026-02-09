import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Optional
from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.training.rewards import CompositeReward


def train_hive_mind_adversarial(
    num_epochs: int = 10,
    lr: float = 1e-3,
    top_k: int = 3,
    observation_dim: int = 64,
    balance_weight: float = 0.1,
    device: Optional[torch.device] = None,
):
    """
    Main adversarial training loop for the HIVE-ZERO swarm.

    Trains the gating network and selected expert adapters using a
    composite loss comprising adversarial evasion, payload diversity,
    information-gain, and gating load-balance terms.  Uses a cosine-
    annealing LR scheduler for stable convergence.

    Args:
        num_epochs:      Number of training epochs.
        lr:              Peak learning rate for Adam.
        top_k:           Number of experts activated per forward pass.
        observation_dim: Observation embedding dimensionality.
        balance_weight:  Coefficient for the gating load-balance loss.
        device:          Torch device; defaults to CUDA if available.
    """
    logger = logging.getLogger("HiveTraining")
    logger.setLevel(logging.INFO)

    # 1. Initialise HiveMind
    hive = HiveMind(observation_dim=observation_dim)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hive.to(device)

    # 2. Collect trainable parameters
    params = list(hive.gating_network.parameters())
    for expert in hive.experts:
        if expert.name in ("Cartographer", "DeepScope", "Chronos",
                           "Mimic", "Ghost", "Stego", "Cleaner",
                           "Tarpit", "FeedbackLoop", "Flashbang", "GlassHouse"):
            params.extend(list(expert.parameters()))

    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 3. Reward function
    reward_calc = CompositeReward()

    # 4. Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # --- Environment step (mocked) ---
        mock_logs = [
            {'src_ip': '192.168.1.1', 'dst_ip': '10.0.0.5', 'port': 80, 'proto': 6},
            {'src_ip': '10.0.0.5', 'dst_ip': '8.8.8.8', 'port': 53, 'proto': 17},
        ]

        # --- Forward pass ---
        results = hive.forward(mock_logs, top_k=top_k)

        # --- Loss accumulation ---
        total_loss = torch.tensor(0.0, device=device)

        # Adversarial evasion
        if "defense_score" in results:
            score = results["defense_score"].to(device)
            total_loss = total_loss + (-torch.mean(score))

        # Payload diversity
        if "optimized_payload" in results:
            payload = results["optimized_payload"].to(device)
            total_loss = total_loss + 0.01 * (-torch.std(payload, dim=-1).mean())

        # Information gain
        if "topology" in results:
            topology = results["topology"].to(device)
            current_entropy = float(torch.mean(torch.abs(topology)).item())
            prev_entropy = 0.8
            r_info = reward_calc.calculate_info_gain_reward(prev_entropy, current_entropy)
            total_loss = total_loss - torch.tensor(r_info, device=device, dtype=torch.float32)

        # Stealth reward (when Mimic is active)
        if "traffic_shape" in results:
            traffic = results["traffic_shape"].to(device)
            baseline = torch.softmax(torch.randn_like(traffic), dim=-1)
            traffic_norm = torch.softmax(traffic, dim=-1)
            r_stealth = reward_calc.calculate_stealth_reward(traffic_norm, baseline)
            total_loss = total_loss + 0.1 * (-r_stealth)

        # Gating load-balance auxiliary loss
        with torch.no_grad():
            data = hive.log_encoder.update(mock_logs)
            if data.x.size(0) > 0:
                gs = torch.mean(data.x, dim=0, keepdim=True)
            else:
                gs = torch.zeros(1, observation_dim, device=data.x.device)
        weights = hive.gating_network(gs)
        balance_loss = hive.gating_network.load_balance_loss(weights)
        total_loss = total_loss + balance_weight * balance_loss

        # --- Backward pass ---
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

        scheduler.step()

        if epoch % max(1, num_epochs // 10) == 0:
            logger.info(
                f"Epoch {epoch}/{num_epochs}  loss={total_loss.item():.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

    logger.info("Training loop complete.")


if __name__ == "__main__":
    logging.basicConfig()
    train_hive_mind_adversarial(num_epochs=2)
