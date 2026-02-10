import logging

import torch
import torch.optim as optim

from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.training.rewards import CompositeReward


def train_hive_mind_adversarial(num_epochs: int = 10):
    logger = logging.getLogger("HiveTraining")
    logger.setLevel(logging.INFO)

    # 1. Initialize HiveMind
    hive = HiveMind(observation_dim=64)
    # Move to GPU if available (internal experts handle device, but parent should coordinate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hive.to(device)

    # 2. Optimizer
    # Only optimize Gating Network + Learnable Experts (excluding Sentinel/PayloadGen maybe?)
    # For Adversarial Loop, we mainly train Gating + Mutator + Recon/Post?
    # Sentinel and PayloadGen might be frozen or pre-trained.

    # Trainable parameters
    params = list(hive.gating_network.parameters())
    # Add expert parameters (e.g. adapters)
    for expert in hive.experts:
        if expert.name == "Mutator":
            # Mutator optimizes at inference time usually,
            # but might have learnable policy/value net (PPO) if implemented fully.
            pass
        elif expert.name in ["Cartographer", "DeepScope", "Mimic", "Ghost"]:
            params.extend(list(expert.parameters()))

    optimizer = optim.Adam(params, lr=0.001)

    # 3. Reward Function
    reward_calc = CompositeReward()

    # 4. Training Loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # --- Environment Step (Mocked) ---
        # Generate random logs to simulate observation
        mock_logs = [
            {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.5", "port": 80, "proto": 6},
            {"src_ip": "10.0.0.5", "dst_ip": "8.8.8.8", "port": 53, "proto": 17},
        ]

        # --- Forward Pass ---
        # Run HiveMind
        results = hive.forward(mock_logs, top_k=3)

        # --- Reward Calculation ---
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Extract results to compute rewards
        # This part depends on which experts were active.
        # Sparse MoE makes batch training tricky without masking.

        # Adversarial Component
        if "defense_score" in results:
            # Score is P(Allowed)
            score = results["defense_score"]
            # We want to MAXIMIZE score. Loss = -Score
            loss_adv = -torch.mean(score)
            total_loss = total_loss + loss_adv

        if "optimized_payload" in results:
            # Maybe auxiliary loss?
            pass

        if "topology" in results:
            # R_info: Entropy reduction?
            # Mock entropy calculation
            current_entropy = 0.5  # Placeholder
            prev_entropy = 0.8
            r_info = reward_calc.calculate_info_gain_reward(prev_entropy, current_entropy)
            # Maximize reward -> Minimize -Reward
            total_loss = total_loss - r_info

        # --- Backward Pass ---
        total_loss.backward()

        # Hardening: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()

        if epoch % 1 == 0:
            logger.info(f"Epoch {epoch}: Loss {total_loss.item()}")

    logger.info("Training loop complete.")


if __name__ == "__main__":
    logging.basicConfig()
    train_hive_mind_adversarial(num_epochs=2)
