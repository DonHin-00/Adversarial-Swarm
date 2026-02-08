import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from hive_zero_core.utils.logging_config import setup_logger
from hive_zero_core.memory.graph_store import LogEncoder
from hive_zero_core.agents.recon_experts import Agent_Cartographer, Agent_DeepScope, Agent_Chronos
from hive_zero_core.agents.attack_experts import Agent_Sentinel, Agent_PayloadGen, Agent_Mutator
from hive_zero_core.agents.post_experts import Agent_Mimic, Agent_Ghost, Agent_Stego, Agent_Cleaner

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Global state/context embedding
        logits = self.net(x)
        # Softmax for weights
        weights = F.softmax(logits, dim=-1)
        return weights

class HiveMind(nn.Module):
    def __init__(self, observation_dim: int = 64):
        super().__init__()
        self.logger = setup_logger("HiveMind")
        self.observation_dim = observation_dim

        # 1. Shared Latent Space / Data Layer
        self.log_encoder = LogEncoder(node_feature_dim=observation_dim)

        # 2. The 10 Experts
        # Define dimensions carefully. For prototype, we use unified dims or specific ones mapped by adapters.
        # We'll use a standard 'action_dim' for most, or expert-specific return types handled by aggregation.

        # Cluster A
        self.expert_cartographer = Agent_Cartographer(observation_dim, action_dim=observation_dim)
        self.expert_deepscope = Agent_DeepScope(observation_dim, action_dim=10) # 10 discrete actions?
        self.expert_chronos = Agent_Chronos(1, action_dim=1) # Time input

        # Cluster B
        self.expert_sentinel = Agent_Sentinel(observation_dim, action_dim=2)
        self.expert_payloadgen = Agent_PayloadGen(observation_dim, action_dim=128) # Seq len
        self.expert_mutator = Agent_Mutator(observation_dim, action_dim=128,
                                           sentinel_expert=self.expert_sentinel,
                                           generator_expert=self.expert_payloadgen)

        # Cluster C
        self.expert_mimic = Agent_Mimic(observation_dim, action_dim=2)
        self.expert_ghost = Agent_Ghost(observation_dim, action_dim=5)
        self.expert_stego = Agent_Stego(observation_dim, action_dim=64)
        self.expert_cleaner = Agent_Cleaner(observation_dim, action_dim=10)

        # Order matters for indexing in GatingNetwork outputs
        self.experts = nn.ModuleList([
            self.expert_cartographer, # 0
            self.expert_deepscope,    # 1
            self.expert_chronos,      # 2
            self.expert_payloadgen,   # 3
            self.expert_mutator,      # 4
            self.expert_sentinel,     # 5
            self.expert_mimic,        # 6
            self.expert_ghost,        # 7
            self.expert_stego,        # 8
            self.expert_cleaner       # 9
        ])

        # 3. Gating Mechanism
        # Routes based on Global Graph Embedding (mean of node features)
        self.gating_network = GatingNetwork(observation_dim, num_experts=len(self.experts))

    def forward(self, raw_logs: List[Dict], top_k: int = 3) -> Dict[str, Any]:
        """
        Main Forward Pass:
        1. Process Logs -> Graph -> Embedding
        2. Gating Network -> Weights
        3. Select Top-K Experts
        4. Execute Active Experts
        5. Aggregate (Return dictionary of results)
        """
        # 1. Encode
        data = self.log_encoder.update(raw_logs)

        # Global State Embedding: Mean pool of node features
        # Handle empty graph case
        if data.x.size(0) > 0:
            global_state = torch.mean(data.x, dim=0, keepdim=True) # [1, dim]
        else:
            global_state = torch.zeros(1, self.observation_dim)

        # 2. Gating
        weights = self.gating_network(global_state) # [1, num_experts]

        # 3. Select Top-K
        # Get indices
        top_k_vals, top_k_indices = torch.topk(weights, k=top_k, dim=-1)
        active_indices = top_k_indices[0].tolist()

        results = {}

        # Reset all to inactive
        for expert in self.experts:
            expert.is_active = False

        # 4. Execute Active Experts
        for idx in active_indices:
            expert = self.experts[idx]
            expert.is_active = True

            # Prepare Input based on Expert Type
            # This is a critical orchestration step. Different experts need different slices of data.
            # Simplified for prototype: pass global node features or specific inputs based on role.

            try:
                if expert.name == "Cartographer":
                    # Needs Node Features + Edge Index
                    out = expert(data.x, context=data.edge_index)
                    results["topology"] = out

                elif expert.name == "DeepScope":
                    # Needs global state to decide masks? Or per-node?
                    # Let's say it masks actions for the system.
                    out = expert(global_state, mask=None)
                    results["constraints"] = out

                elif expert.name == "Chronos":
                    # Needs time series. We don't have it in graph yet.
                    # Mock input for now or extract from logs if timestamps existed
                    dummy_times = torch.randn(1, 10) # Batch 1, seq 10
                    out = expert(dummy_times)
                    results["timing"] = out

                elif expert.name == "PayloadGen":
                    # Needs context.
                    out = expert(global_state) # Token IDs
                    results["raw_payload"] = out

                elif expert.name == "Mutator":
                    # Needs context for generator
                    # Mutator calls generator internally
                    out = expert(global_state)
                    results["optimized_payload"] = out

                elif expert.name == "Sentinel":
                    # Usually called by Mutator, but if routed directly:
                    # Assess current state?
                    out = expert(global_state.unsqueeze(1)) # [1, 1, dim]
                    results["defense_score"] = out

                elif expert.name == "Mimic":
                    out = expert(global_state)
                    results["traffic_shape"] = out

                elif expert.name == "Ghost":
                    out = expert(global_state)
                    results["hiding_spot"] = out

                elif expert.name == "Stego":
                    dummy_data = torch.rand(1, self.observation_dim)
                    out = expert(dummy_data)
                    results["covert_channel"] = out

                elif expert.name == "Cleaner":
                    out = expert(global_state)
                    results["cleanup"] = out

            except Exception as e:
                self.logger.error(f"Execution failed for {expert.name}: {e}")
                # Don't crash the whole swarm

        return results
