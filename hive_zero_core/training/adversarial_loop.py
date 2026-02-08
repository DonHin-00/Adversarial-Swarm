import logging

import torch
from torch import optim

from hive_zero_core.environment.hive_env import HiveZeroEnv
from hive_zero_core.hive_mind import HiveMind


def train_hive_mind_adversarial(num_epochs: int = 10, batch_size: int = 4):
    logger = logging.getLogger("HiveTraining")
    logger.setLevel(logging.INFO)

    # 1. HiveMind Agent
    hive = HiveMind(observation_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hive.to(device)
    hive.train()

    # 2. Real PenTest Environment
    env = HiveZeroEnv(observation_dim=64)

    # 3. Optimizer
    params = list(hive.gating_network.parameters())
    for expert in hive.experts:
        if hasattr(expert, 'parameters'):
             params.extend(list(expert.parameters()))
        if expert.name == "Sentinel":
            # Sentinel is part of the 'Blue Team' logic, implicitly trained or pre-trained.
            pass
        elif expert.name in ["Cartographer", "DeepScope", "Mimic", "Ghost", "Chronos", "PayloadGen", "Cleaner", "Stego"]:
            params.extend(list(expert.parameters()))

    optimizer = optim.Adam(params, lr=0.001)

    # 4. Training Loop
    for epoch in range(num_epochs):
        obs_np, _ = env.reset()

        # Episode Loop
        total_episode_reward = 0
        done = False

        while not done:
            optimizer.zero_grad()

            # --- Agent Decision ---
            # Forward pass to get Action Vector
            # We mock the input logs for now, assuming NmapAdapter would run here in real loop
            mock_logs = [{'src_ip': '127.0.0.1', 'dst_ip': '127.0.0.1', 'port': 80}]

            results = hive.forward(mock_logs, top_k=3)

            # Extract action for Env
            action_tensor = torch.zeros(128, device=device)

            if "optimized_payload" in results:
                # payload might be [Batch, Seq, Dim] or similar. Flatten/Pool.
                p = results["optimized_payload"]
                if p.numel() > 0:
                    flat = p.view(-1)
                    if flat.size(0) >= 128:  # noqa: PLR2004
                        action_tensor = flat[:128]
                    else:
                        action_tensor[:flat.size(0)] = flat

            # --- Environment Interaction ---
            action_np = action_tensor.detach().cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            total_episode_reward += reward
            obs_np = next_obs  # noqa: F841

            # --- Loss Calculation ---
            # Maximize Reward -> Minimize -Reward
            loss = -torch.tensor(reward, device=device, requires_grad=True)

            if "defense_score" in results:
                # Maximize P(Allowed)
                score = results["defense_score"]
                loss = loss - 0.1 * torch.log(score + 1e-8).mean()

            # Load Balancing (if available)
            if "gating_weights" in results:
                w = results["gating_weights"]
                entropy = -torch.sum(w * torch.log(w + 1e-8), dim=-1).mean()
                loss = loss - 0.05 * entropy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

        logger.info(f"Epoch {epoch}: Reward {total_episode_reward}")
