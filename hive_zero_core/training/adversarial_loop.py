import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.training.rewards import CompositeReward
from hive_zero_core.memory.replay_buffer import PrioritizedReplayBuffer
from hive_zero_core.environment.hive_env import HiveZeroEnv

def train_hive_mind_adversarial(num_epochs: int = 10, batch_size: int = 4):
    logger = logging.getLogger("HiveTraining")
    logger.setLevel(logging.INFO)

    # 1. Initialize HiveMind (The Agent)
    hive = HiveMind(observation_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hive.to(device)
    hive.train()

    # 2. Initialize Environment (The Target)
    env = HiveZeroEnv(observation_dim=64)

    # 3. Optimizer
    params = list(hive.gating_network.parameters())
    for expert in hive.experts:
        if expert.name == "Sentinel":
            # Sentinel is part of the 'Blue Team' logic, implicitly trained or pre-trained.
            # Here we might freeze it or train it adversarially.
            pass
        elif expert.name in ["Cartographer", "DeepScope", "Mimic", "Ghost", "Chronos", "PayloadGen", "Cleaner", "Stego"]:
            # Note: PayloadGen has fixed RAG DB but trainable generator parts if any
            params.extend(list(expert.parameters()))

    optimizer = optim.Adam(params, lr=0.001)

    # 4. Training Loop
    for epoch in range(num_epochs):
        # Reset Env
        obs_np, _ = env.reset()

        # Episode Loop
        total_episode_reward = 0
        done = False

        while not done:
            optimizer.zero_grad()

            # --- Agent Step ---
            # Convert Obs to Logs format for HiveMind
            # HiveMind expects List[Dict]. Env gives vector.
            # We mock the translation:
            mock_logs = [{'src_ip': '1.1.1.1', 'dst_ip': '2.2.2.2', 'port': 80, 'proto': 6}] # Using dummy for graph structure
            # But we inject the Env's observation features into the graph manually?
            # Or simpler: HiveMind forward handles the graph update.
            # We assume Env Obs *is* the feature vector for the global node?

            # Run HiveMind
            results = hive.forward(mock_logs, top_k=3)

            # Extract Action for Env
            # Env expects [128] vector.
            # Aggregate expert outputs?
            # Let's take 'raw_payload' (PayloadGen) or 'optimized_payload' (Mutator) if available.
            # Fallback to zero.

            action_tensor = torch.zeros(128, device=device)
            if "optimized_payload" in results:
                # payload might be [Batch, Seq, Dim] or similar. Flatten/Pool.
                p = results["optimized_payload"]
                if p.numel() > 0:
                    flat = p.view(-1)
                    if flat.size(0) >= 128:
                        action_tensor = flat[:128]
                    else:
                        action_tensor[:flat.size(0)] = flat

            # --- Environment Step ---
            action_np = action_tensor.detach().cpu().numpy()
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            total_episode_reward += reward

            # --- Loss Calculation ---
            # 1. RL Reward Loss (Policy Gradient? Or Differentiable Simulation?)
            # Since Env is non-differentiable (numpy), we usually need PPO/REINFORCE.
            # But HiveMind parts (Sentinel loop) are differentiable.
            # Mixed approach:
            # - Use Sentinel Score (internal) as differentiable proxy for 'Detectability'.
            # - Use Env Reward for external feedback.

            loss_total = torch.tensor(0.0, device=device, requires_grad=True)

            # Internal Adv Loss
            if "defense_score" in results:
                # Sentinel score (P(Allowed))
                score = results["defense_score"]
                loss_adv = -torch.log(score + 1e-8).mean()
                loss_total = loss_total + loss_adv

            # Load Balancing
            if "gating_weights" in results:
                w = results["gating_weights"]
                entropy = -torch.sum(w * torch.log(w + 1e-8), dim=-1).mean()
                loss_total = loss_total - 0.05 * entropy

            # Backward
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            obs_np = next_obs_np

        if epoch % 1 == 0:
            logger.info(f"Epoch {epoch}: Reward {total_episode_reward}")
