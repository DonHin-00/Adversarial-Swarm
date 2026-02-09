import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, TypedDict
from hive_zero_core.utils.logging_config import setup_logger
from hive_zero_core.memory.graph_store import HeteroLogEncoder
from hive_zero_core.agents.recon_experts import CartographerAgent, DeepScopeAgent, ChronosAgent
from hive_zero_core.agents.attack_experts import SentinelAgent, PayloadGenAgent, MutatorAgent
from hive_zero_core.agents.post_experts import MimicAgent, GhostAgent, StegoAgent, CleanerAgent

class NoisyGatingNetwork(nn.Module):
    """
    Implements a Noisy Gating Network for Sparse Mixture-of-Experts.

    Provides learnable routing with exploration noise to prevent
    expert collapse during training.
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64, noise_epsilon: float = 1e-2):
        """
        Args:
            input_dim: Dimension of the global state embedding.
            num_experts: Total number of specialists in the swarm.
            noise_epsilon: Base noise scale for exploration.
        """
        super().__init__()
        self.num_experts = num_experts
        self.noise_epsilon = noise_epsilon

        self.w_gating = nn.Linear(input_dim, num_experts)
        self.w_noise = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes gating weights for routing.
        """
        clean_logits = self.w_gating(x)

        if training:
            raw_noise_std = self.w_noise(x)
            noise_std = F.softplus(raw_noise_std)
            noise = torch.randn_like(clean_logits) * noise_std
            logits = clean_logits + noise
        else:
            logits = clean_logits

        weights = F.softmax(logits, dim=-1)
        return weights, logits

class HiveResults(TypedDict, total=False):
    topology: torch.Tensor
    constraints: torch.Tensor
    timing: torch.Tensor
    raw_payload: torch.Tensor
    optimized_payload: torch.Tensor
    defense_score: torch.Tensor
    traffic_shape: torch.Tensor
    hiding_spot: torch.Tensor
    covert_channel: torch.Tensor
    cleanup: torch.Tensor
    gating_weights: torch.Tensor

class HiveMind(nn.Module):
    """
    Central Controller for the HIVE-ZERO H-MARL System.

    Coordinates a swarm of 10 specialized agents using a sparse MoE architecture.
    Handles data encoding, routing via Noisy Gating, and result aggregation.
    """
    def __init__(self, observation_dim: int = 64):
        super().__init__()
        self.logger = setup_logger("HiveMind")
        self.observation_dim = observation_dim

        # 1. Data Layer (Updated to Hetero)
        self.log_encoder = HeteroLogEncoder(node_embed_dim=observation_dim)

        # 2. Experts
        self.expert_cartographer = CartographerAgent(observation_dim, action_dim=observation_dim)
        self.expert_deepscope = DeepScopeAgent(observation_dim, action_dim=10)
        self.expert_chronos = ChronosAgent(1, action_dim=1)

        self.expert_sentinel = SentinelAgent(observation_dim, action_dim=2)
        self.expert_payloadgen = PayloadGenAgent(observation_dim, action_dim=128)
        self.expert_mutator = MutatorAgent(observation_dim, action_dim=128,
                                           sentinel_expert=self.expert_sentinel,
                                           generator_expert=self.expert_payloadgen)

        self.expert_mimic = MimicAgent(observation_dim, action_dim=2)
        self.expert_ghost = GhostAgent(observation_dim, action_dim=5)
        self.expert_stego = StegoAgent(observation_dim, action_dim=64)
        self.expert_cleaner = CleanerAgent(observation_dim, action_dim=10)

        self.experts = nn.ModuleList([
            self.expert_cartographer,
            self.expert_deepscope,
            self.expert_chronos,
            self.expert_payloadgen,
            self.expert_mutator,
            self.expert_sentinel,
            self.expert_mimic,
            self.expert_ghost,
            self.expert_stego,
            self.expert_cleaner
        ])

        # 3. Gating
        self.gating_network = NoisyGatingNetwork(observation_dim, num_experts=len(self.experts))

    def forward(self, raw_logs: List[Dict], top_k: int = 3) -> HiveResults:
        """
        Executes the main forward pass of the swarm.

        1. Encodes raw log stream into a heterogeneous graph.
        2. Computes a global state embedding.
        3. Routes the state through the Noisy Gating Network.
        4. Activates and executes the Top-K experts.

        Args:
            raw_logs: List of log dictionaries (e.g., {'src_ip': ..., 'port': ...}).
            top_k: Number of experts to activate.

        Returns:
            A HiveResults dictionary containing outputs from active agents.
        """
        # 1. Encode
        data = self.log_encoder.update(raw_logs)

        # Global State Embedding from HeteroData
        # Aggregate IP nodes?
        if 'ip' in data.node_types and data['ip'].x.size(0) > 0:
            global_state = torch.mean(data['ip'].x, dim=0, keepdim=True)
        else:
            global_state = torch.zeros(1, self.observation_dim, device=next(self.parameters()).device)

        # 2. Gating
        weights, logits = self.gating_network(global_state, training=self.training)

        # 3. Select Top-K
        top_k_vals, top_k_indices = torch.topk(weights, k=top_k, dim=-1)
        active_indices = top_k_indices[0].tolist()

        results = {}

        for expert in self.experts:
            expert.is_active = False

        # 4. Execute Active Experts
        for idx in active_indices:
            expert = self.experts[idx]
            expert.is_active = True

            try:
                # Handling HeteroData vs Tensor inputs based on Expert type
                if expert.name == "Cartographer":
                    # Expects HeteroData
                    out = expert(data)
                    results["topology"] = out

                elif expert.name == "DeepScope":
                    out = expert(global_state)
                    results["constraints"] = out

                elif expert.name == "Chronos":
                    dummy_times = torch.randn(1, 10, device=global_state.device)
                    out = expert(dummy_times)
                    results["timing"] = out

                elif expert.name == "PayloadGen":
                    out = expert(global_state)
                    results["raw_payload"] = out

                elif expert.name == "Mutator":
                    out = expert(global_state)
                    results["optimized_payload"] = out

                elif expert.name == "Sentinel":
                    out = expert(global_state.unsqueeze(1))
                    results["defense_score"] = out

                elif expert.name == "Mimic":
                    out = expert(global_state)
                    results["traffic_shape"] = out

                elif expert.name == "Ghost":
                    out = expert(global_state)
                    results["hiding_spot"] = out

                elif expert.name == "Stego":
                    dummy_data = torch.rand(1, self.observation_dim, device=global_state.device)
                    out = expert(dummy_data)
                    results["covert_channel"] = out

                elif expert.name == "Cleaner":
                    out = expert(global_state)
                    results["cleanup"] = out

            except Exception as e:
                self.logger.error(f"Execution failed for {expert.name}: {e}")

        results["gating_weights"] = weights
        return results
