from typing import Dict, List, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_zero_core.agents.attack_experts import MutatorAgent, PayloadGenAgent, SentinelAgent
from hive_zero_core.agents.post_experts import CleanerAgent, GhostAgent, MimicAgent, StegoAgent
from hive_zero_core.agents.recon_experts import CartographerAgent, ChronosAgent, DeepScopeAgent
from hive_zero_core.memory.graph_store import HeteroLogEncoder
from hive_zero_core.utils.logging_config import setup_logger


class NoisyGatingNetwork(nn.Module):
    """
    Implements a Noisy Gating Network for Sparse Mixture-of-Experts.

    Provides learnable routing with exploration noise to prevent
    expert collapse during training.
    """

    def __init__(
        self, input_dim: int, num_experts: int, hidden_dim: int = 64, noise_epsilon: float = 1e-2
    ):
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

    **Thread Safety:** This module is NOT thread-safe. The `is_active` flags on
    experts are modified during forward passes. Use a single HiveMind instance
    per thread, or add external synchronization (e.g., threading.Lock) if
    concurrent forward passes are required.
    """

    def __init__(
        self, observation_dim: int = 64, load_hf_models: bool = True, local_files_only: bool = False
    ):
        """
        Initialize HiveMind controller.

        Args:
            observation_dim: Dimension of observation space.
            load_hf_models: If False, defer loading of HuggingFace models (SentinelAgent, PayloadGenAgent).
                           Set to False for offline environments or when models aren't needed.
            local_files_only: If True, only use locally cached HuggingFace models (no network downloads).
                             Requires models to be pre-downloaded. Ignored if load_hf_models=False.
        """
        super().__init__()
        self.logger = setup_logger("HiveMind")
        self.observation_dim = observation_dim

        # 1. Data Layer (Updated to Hetero)
        self.log_encoder = HeteroLogEncoder(node_embed_dim=observation_dim)

        # 2. Experts (non-HF experts always loaded)
        self.expert_cartographer = CartographerAgent(observation_dim, action_dim=observation_dim)
        self.expert_deepscope = DeepScopeAgent(observation_dim, action_dim=10)
        self.expert_chronos = ChronosAgent(1, action_dim=1)

        # HuggingFace-dependent experts (conditionally loaded)
        if load_hf_models:
            try:
                self.expert_sentinel = SentinelAgent(
                    observation_dim, action_dim=2, local_files_only=local_files_only
                )
                self.expert_payloadgen = PayloadGenAgent(
                    observation_dim, action_dim=128, local_files_only=local_files_only
                )
                self.expert_mutator = MutatorAgent(
                    observation_dim,
                    action_dim=128,
                    sentinel_expert=self.expert_sentinel,
                    generator_expert=self.expert_payloadgen,
                )
            except (OSError, RuntimeError) as e:
                self.logger.warning(
                    f"Failed to load HuggingFace models: {e}. "
                    "Continuing with non-HF experts only. "
                    "To avoid network access, use load_hf_models=False or local_files_only=True with cached models."
                )
                # Create placeholder None values for HF experts
                self.expert_sentinel = None
                self.expert_payloadgen = None
                self.expert_mutator = None
        else:
            self.logger.info("Skipping HuggingFace model loading (load_hf_models=False)")
            self.expert_sentinel = None
            self.expert_payloadgen = None
            self.expert_mutator = None

        self.expert_mimic = MimicAgent(observation_dim, action_dim=2)
        self.expert_ghost = GhostAgent(observation_dim, action_dim=5)
        self.expert_stego = StegoAgent(observation_dim, action_dim=64)
        self.expert_cleaner = CleanerAgent(observation_dim, action_dim=10)

        # Build experts list, filtering out None (unloaded HF experts)
        all_experts = [
            self.expert_cartographer,
            self.expert_deepscope,
            self.expert_chronos,
            self.expert_payloadgen,
            self.expert_mutator,
            self.expert_sentinel,
            self.expert_mimic,
            self.expert_ghost,
            self.expert_stego,
            self.expert_cleaner,
        ]
        self.experts = nn.ModuleList([e for e in all_experts if e is not None])

        # 3. Gating
        self.gating_network = NoisyGatingNetwork(observation_dim, num_experts=len(self.experts))

        # 4. Projection layer for Sentinel (observation_dim -> BERT hidden_size)
        # The Sentinel's BERT backbone expects hidden_size embeddings
        # Only create projection if Sentinel is loaded
        if self.expert_sentinel is not None:
            try:
                sentinel_hidden_size = self.expert_sentinel.backbone.config.hidden_size
                self.sentinel_projection = nn.Linear(observation_dim, sentinel_hidden_size)
            except (AttributeError, RuntimeError) as e:
                self.logger.warning(
                    f"Failed to create Sentinel projection layer: {e}. Using default hidden_size=128"
                )
                self.sentinel_projection = nn.Linear(observation_dim, 128)
        else:
            # No sentinel loaded, use None or dummy projection
            self.sentinel_projection = None

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
        if "ip" in data.node_types and data["ip"].x.size(0) > 0:
            global_state = torch.mean(data["ip"].x, dim=0, keepdim=True)
        else:
            global_state = torch.zeros(
                1, self.observation_dim, device=next(self.parameters()).device
            )

        # 2. Gating
        weights, logits = self.gating_network(global_state, training=self.training)

        # 3. Select Top-K with validation
        num_experts = len(self.experts)
        if num_experts == 0:
            raise ValueError("No experts are available for routing; cannot select top_k experts.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(
                f"Invalid top_k={top_k}. Expected top_k to be between 1 and {num_experts} "
                "to safely select experts using torch.topk."
            )

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
                    # Project observation_dim to BERT's hidden_size to avoid shape mismatch
                    projected_state = self.sentinel_projection(global_state)
                    out = expert(projected_state.unsqueeze(1))
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
