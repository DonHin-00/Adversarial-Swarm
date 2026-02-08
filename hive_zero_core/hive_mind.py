import torch  # noqa: I001
import torch.nn as nn  # noqa: PLR0402
import torch.nn.functional as F  # noqa: N812
from typing import List, Dict, Optional, Tuple, Any, cast  # noqa: F401
from hive_zero_core.utils.logging_config import setup_logger
from hive_zero_core.memory.graph_store import HeteroLogEncoder
from hive_zero_core.agents.recon_experts import Agent_Cartographer, Agent_DeepScope, Agent_Chronos
from hive_zero_core.agents.attack_experts import Agent_Sentinel, Agent_PayloadGen, Agent_Mutator
from hive_zero_core.agents.post_experts import Agent_Mimic, Agent_Ghost, Agent_Stego, Agent_Cleaner
from hive_zero_core.agents.base_expert import BaseExpert
from hive_zero_core.agents.defense_experts import Agent_Tarpit

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, TypedDict
from hive_zero_core.utils.logging_config import setup_logger
from hive_zero_core.memory.graph_store import HeteroLogEncoder
from hive_zero_core.agents.recon_experts import CartographerAgent, DeepScopeAgent, ChronosAgent
from hive_zero_core.agents.attack_experts import SentinelAgent, PayloadGenAgent, MutatorAgent
from hive_zero_core.agents.post_experts import MimicAgent, GhostAgent, StegoAgent, CleanerAgent

class NoisyGatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64, noise_epsilon: float = 1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.noise_epsilon = noise_epsilon

        self.w_gating = nn.Linear(input_dim, num_experts)
        self.w_noise = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, observation_dim: int = 64):
        super().__init__()
        self.logger = setup_logger("HiveMind")
        self.observation_dim = observation_dim

        self.log_encoder = HeteroLogEncoder(node_embed_dim=observation_dim)

        # 2. The 11 Experts (Cluster A, B, C + Active Defense)
        # Define dimensions carefully. For prototype, we use unified dims or specific ones mapped by adapters.
        # We'll use a standard 'action_dim' for most, or expert-specific return types handled by aggregation.

        # Cluster A: Recon
        self.expert_cartographer = Agent_Cartographer(observation_dim, action_dim=observation_dim)
        self.expert_deepscope = Agent_DeepScope(observation_dim, action_dim=10)
        self.expert_chronos = Agent_Chronos(1, action_dim=1)

        # Cluster B: Attack
        self.expert_sentinel = Agent_Sentinel(observation_dim, action_dim=2)
        self.expert_payloadgen = Agent_PayloadGen(observation_dim, action_dim=128)
        self.expert_mutator = Agent_Mutator(
            observation_dim,
            action_dim=128,
            sentinel_expert=self.expert_sentinel,
            generator_expert=self.expert_payloadgen,
        )

        # Cluster C: Post-Exploit
        self.expert_mimic = Agent_Mimic(observation_dim, action_dim=2)
        self.expert_ghost = Agent_Ghost(observation_dim, action_dim=5)
        self.expert_stego = Agent_Stego(observation_dim, action_dim=64)
        self.expert_cleaner = Agent_Cleaner(observation_dim, action_dim=10)
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

        # Cluster D: Active Defense (The Hunter)
        # Action dim 64 matches observation dim to simulate "port" coverage or full-spectrum noise
        self.expert_tarpit = Agent_Tarpit(observation_dim, action_dim=observation_dim)

        # Order matters for indexing in GatingNetwork outputs
        self.experts = nn.ModuleList(
            [
                self.expert_cartographer,  # 0
                self.expert_deepscope,  # 1
                self.expert_chronos,  # 2
                self.expert_payloadgen,  # 3
                self.expert_mutator,  # 4
                self.expert_sentinel,  # 5
                self.expert_mimic,  # 6
                self.expert_ghost,  # 7
                self.expert_stego,  # 8
                self.expert_cleaner,  # 9
                self.expert_tarpit,  # 10
            ]
        )

        self.gating_network = NoisyGatingNetwork(observation_dim, num_experts=len(self.experts))

    def forward(self, raw_logs: List[Dict], top_k: int = 3) -> Dict[str, Any]:  # noqa: PLR0912, PLR0915
    def forward(self, raw_logs: List[Dict], top_k: int = 3) -> HiveResults:
        """
        Main Forward Pass with Noisy Gating.
        """
        # 1. Encode
        data = self.log_encoder.update(raw_logs)

        device = next(self.parameters()).device
        if "ip" in data.node_types and hasattr(data["ip"], "x") and data["ip"].x.size(0) > 0:
            global_state = torch.mean(data["ip"].x, dim=0, keepdim=True)
        else:
            global_state = torch.zeros(1, self.observation_dim, device=device)

        weights, logits = self.gating_network(global_state, training=self.training)

        top_k_vals, top_k_indices = torch.topk(weights, k=top_k, dim=-1)
        active_indices = top_k_indices[0].tolist()

        results: Dict[str, Any] = {}

        for module in self.experts:
            expert = cast(BaseExpert, module)
            expert.is_active = False

        for idx in active_indices:
            module = self.experts[idx]
            expert = cast(BaseExpert, module)
            expert.is_active = True

            try:
                if expert.name == "Cartographer":
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

                elif expert.name == "Tarpit":
                    # The Hunter needs maximum view (global_state) to deploy traps
                    out = expert(global_state)
                    results["active_defense"] = out

            except Exception as e:
                self.logger.error(f"Execution failed for {expert.name}: {e}")

        results["gating_weights"] = weights
        return results
