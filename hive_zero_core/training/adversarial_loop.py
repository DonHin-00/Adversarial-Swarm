import torch
import torch.nn as nn
import torch.optim as optim
import logging
from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.training.rewards import CompositeReward
from hive_zero_core.memory.replay_buffer import PrioritizedReplayBuffer

def train_hive_mind_adversarial(num_epochs: int = 10, batch_size: int = 4):
    logger = logging.getLogger("HiveTraining")
    logger.setLevel(logging.INFO)

    # 1. Initialize HiveMind
    hive = HiveMind(observation_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hive.to(device)
    hive.train()

    # 2. Optimizer
    # Include gating noise parameters
    params = list(hive.gating_network.parameters())
    for expert in hive.experts:
        if expert.name == "Mutator":
            pass
        elif expert.name == "Sentinel":
            # Optional: Train sentinel too? For adversarial loop, usually fixed or alternating.
            # Let's freeze sentinel for now to treat it as environment.
            pass
        elif expert.name in ["Cartographer", "DeepScope", "Mimic", "Ghost"]:
            params.extend(list(expert.parameters()))

    optimizer = optim.Adam(params, lr=0.001)

    # 3. Reward & Replay
    reward_calc = CompositeReward()
    replay_buffer = PrioritizedReplayBuffer(capacity=1000)

    # 4. Training Loop
    for epoch in range(num_epochs):
        # Curriculum: Increase difficulty (thresholds)
        # Not implemented explicitly here, but logic placeholder

        optimizer.zero_grad()

        # --- Environment Step (Mocked) ---
        # Generate random logs to simulate observation
        mock_logs = [
            {'src_ip': '192.168.1.1', 'dst_ip': '10.0.0.5', 'port': 80, 'proto': 6},
            {'src_ip': '10.0.0.5', 'dst_ip': '8.8.8.8', 'port': 53, 'proto': 17}
        ]

        # --- Forward Pass ---
        results = hive.forward(mock_logs, top_k=3)

        # --- Loss Calculation ---
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 1. Task Rewards
        if "defense_score" in results:
            score = results["defense_score"] # P(Allowed)
            # Adv Loss: We want score -> 1. Loss = 1 - score? Or -log(score).
            # Using -log(score)
            loss_adv = -torch.log(score + 1e-8).mean()
            total_loss = total_loss + loss_adv

        # 2. Load Balancing Loss (Auxiliary)
        # Avoid expert collapse. Minimize CV of gating weights squared?
        if "gating_weights" in results:
            weights = results["gating_weights"] # [batch, num_experts]
            # Ideally variance should be low -> equal load? Or just high entropy?
            # Maximize entropy: -sum(p log p). Minimize negative entropy.
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
            loss_balance = -0.1 * entropy # Weight factor 0.1
            total_loss = total_loss + loss_balance

        # --- Backward Pass ---
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # --- Experience Replay Update ---
        # Store transition? State is graph embedding. Action is experts selected. Reward is -loss.
        # This is a bit abstract for direct RL update here without Q-network,
        # but we can store for offline analysis or separate PPO learner.

        if epoch % 1 == 0:
            logger.info(f"Epoch {epoch}: Loss {total_loss.item()}")

    logger.info("Training loop complete.")

if __name__ == "__main__":
    logging.basicConfig()
    train_hive_mind_adversarial(num_epochs=2)
