import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from hive_zero_core.utils.logging_config import setup_logger
from hive_zero_core.memory.graph_store import LogEncoder
from hive_zero_core.memory.foundation import KnowledgeLoader, WeightInitializer
from hive_zero_core.agents.recon_experts import Agent_Cartographer, Agent_DeepScope, Agent_Chronos
from hive_zero_core.agents.attack_experts import Agent_Sentinel, Agent_PayloadGen, Agent_Mutator
from hive_zero_core.agents.post_experts import Agent_Mimic, Agent_Ghost, Agent_Stego, Agent_Cleaner
from hive_zero_core.agents.defense_experts import Agent_Tarpit
from hive_zero_core.agents.offensive_defense import Agent_FeedbackLoop, Agent_Flashbang, Agent_GlassHouse

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
    def __init__(self, observation_dim: int = 64, pretrained: bool = False):
        super().__init__()
        self.logger = setup_logger("HiveMind")
        self.observation_dim = observation_dim

        # 1. Shared Latent Space / Data Layer
        self.log_encoder = LogEncoder(node_feature_dim=observation_dim)

        # 2. The 14 Experts (Cluster A, B, C + Active Defense + Kill Chain)

        # Cluster A: Recon
        self.expert_cartographer = Agent_Cartographer(observation_dim, action_dim=observation_dim)
        self.expert_deepscope = Agent_DeepScope(observation_dim, action_dim=10) # 10 discrete actions?
        self.expert_chronos = Agent_Chronos(1, action_dim=1) # Time input

        # Cluster B: Attack
        self.expert_sentinel = Agent_Sentinel(observation_dim, action_dim=2)
        self.expert_payloadgen = Agent_PayloadGen(observation_dim, action_dim=128) # Seq len
        self.expert_mutator = Agent_Mutator(observation_dim, action_dim=128,
                                           sentinel_expert=self.expert_sentinel,
                                           generator_expert=self.expert_payloadgen)

        # Cluster C: Post-Exploit
        self.expert_mimic = Agent_Mimic(observation_dim, action_dim=2)
        self.expert_ghost = Agent_Ghost(observation_dim, action_dim=5)
        self.expert_stego = Agent_Stego(observation_dim, action_dim=64)
        self.expert_cleaner = Agent_Cleaner(observation_dim, action_dim=10)

        # Cluster D: Active Defense (The Hunter)
        self.expert_tarpit = Agent_Tarpit(observation_dim, action_dim=observation_dim)

        # Cluster E: Kill Chain (The Synergizers)
        self.expert_feedback = Agent_FeedbackLoop(observation_dim, action_dim=observation_dim)
        self.expert_flashbang = Agent_Flashbang(observation_dim, action_dim=observation_dim)
        self.expert_glasshouse = Agent_GlassHouse(observation_dim, action_dim=observation_dim)

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
            self.expert_cleaner,      # 9
            self.expert_tarpit,       # 10
            self.expert_feedback,     # 11
            self.expert_flashbang,    # 12
            self.expert_glasshouse    # 13
        ])

        # 3. Gating Mechanism
        self.gating_network = GatingNetwork(observation_dim, num_experts=len(self.experts))

        # 4. Foundation / Knowledge Bootstrap
        if pretrained:
            self.bootstrap_knowledge()

    def bootstrap_knowledge(self, years: int = 2):
        """
        Injects 1-2 years of 'Mastery' knowledge and instinctual biases.
        """
        self.logger.info(f"Bootstrapping {years} years of foundational operational knowledge...")
        WeightInitializer.inject_instincts(self.gating_network)
        self.logger.info("Instincts injected: Bias towards Active Defense established.")
        self.knowledge_loader = KnowledgeLoader(self.observation_dim, self.observation_dim)

    def forward(self, raw_logs: List[Dict], top_k: int = 3) -> Dict[str, Any]:
        """
        Main Forward Pass with fixed Quad-Strike Logic.
        """
        data = self.log_encoder.update(raw_logs)

        if data.x.size(0) > 0:
            global_state = torch.mean(data.x, dim=0, keepdim=True)
        else:
            global_state = torch.zeros(1, self.observation_dim)

        weights = self.gating_network(global_state)

        top_k_vals, top_k_indices = torch.topk(weights, k=top_k, dim=-1)
        active_indices = top_k_indices[0].tolist()

        # SYNERGY LOGIC: Force-Enable Kill Chain
        tarpit_idx = 10
        if tarpit_idx in active_indices:
            if 11 not in active_indices: active_indices.append(11) # Feedback
            if 12 not in active_indices: active_indices.append(12) # Flashbang
            if 13 not in active_indices: active_indices.append(13) # GlassHouse

        results = {}

        # Reset all to inactive
        for expert in self.experts:
            expert.is_active = False

        # Execute Active Experts using index-based dispatch for better performance
        for idx in active_indices:
            expert = self.experts[idx]
            expert.is_active = True

            try:
                # Index-based expert execution (more efficient than string comparison)
                if idx == 0:  # Cartographer
                    out = expert(data.x, context=data.edge_index)
                    results["topology"] = out

                elif idx == 1:  # DeepScope
                    out = expert(global_state, mask=None)
                    results["constraints"] = out

                elif idx == 2:  # Chronos
                    dummy_times = torch.randn(1, 10)
                    out = expert(dummy_times)
                    results["timing"] = out

                elif idx == 3:  # PayloadGen
                    out = expert(global_state)
                    results["raw_payload"] = out

                elif idx == 4:  # Mutator
                    out = expert(global_state)
                    results["optimized_payload"] = out

                elif idx == 5:  # Sentinel
                    out = expert(global_state.unsqueeze(1))
                    results["defense_score"] = out

                elif idx == 6:  # Mimic
                    out = expert(global_state)
                    results["traffic_shape"] = out

                elif idx == 7:  # Ghost
                    out = expert(global_state)
                    results["hiding_spot"] = out

                elif idx == 8:  # Stego
                    dummy_data = torch.rand(1, self.observation_dim)
                    out = expert(dummy_data)
                    results["covert_channel"] = out

                elif idx == 9:  # Cleaner
                    out = expert(global_state)
                    results["cleanup"] = out

                elif idx == 10:  # Tarpit
                    out = expert(global_state)
                    results["active_defense"] = out

                elif idx == 11:  # FeedbackLoop
                    out = expert(global_state)
                    results["counter_strike"] = out

                elif idx == 12:  # Flashbang
                    out = expert(global_state)
                    results["overload"] = out

                elif idx == 13:  # GlassHouse
                    out = expert(global_state)
                    results["total_exposure"] = out

            except Exception as e:
                self.logger.error(f"Execution failed for expert {idx}: {e}")

        return results
