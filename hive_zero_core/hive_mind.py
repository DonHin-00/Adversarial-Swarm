import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from hive_zero_core.utils.logging_config import setup_logger
from hive_zero_core.memory.graph_store import LogEncoder
from hive_zero_core.memory.foundation import KnowledgeLoader, WeightInitializer
from hive_zero_core.memory.threat_intel_db import ThreatIntelDB
from hive_zero_core.agents.recon_experts import Agent_Cartographer, Agent_DeepScope, Agent_Chronos
from hive_zero_core.agents.attack_experts import Agent_Sentinel, Agent_PayloadGen, Agent_Mutator
from hive_zero_core.agents.post_experts import Agent_Mimic, Agent_Ghost, Agent_Stego, Agent_Cleaner
from hive_zero_core.agents.defense_experts import Agent_Tarpit
from hive_zero_core.agents.offensive_defense import Agent_FeedbackLoop, Agent_Flashbang, Agent_GlassHouse
from hive_zero_core.agents.blue_team import Agent_WAF, Agent_EDR, Agent_SIEM, Agent_IDS
from hive_zero_core.agents.red_booster import Agent_PreAttackBooster

class GatingNetwork(nn.Module):
    """
    Sparse Mixture-of-Experts gating network.

    Maps a global state embedding to expert selection weights via a learned
    two-layer MLP. During training, Gaussian noise is injected into logits
    to encourage exploration across experts, and a load-balancing auxiliary
    loss penalises routing collapse onto a single expert.
    """

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64,
                 noise_std: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.noise_std = noise_std

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, num_experts)
        )

        # Running estimate of expert utilisation for load balancing
        self.register_buffer(
            "_expert_counts",
            torch.zeros(num_experts, dtype=torch.float32),
        )
        self._total_routes: int = 0

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Global state embedding [batch, input_dim].
            top_k: If set, applies top-k sparse routing — only the top_k
                   experts receive non-zero weight while preserving gradients
                   through a straight-through estimator.

        Returns:
            Normalised expert weights [batch, num_experts].
        """
        logits = self.net(x)

        # Exploration noise during training
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        weights = F.softmax(logits, dim=-1)

        # Top-k sparse routing with straight-through gradient
        if top_k is not None and 0 < top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(weights, k=top_k, dim=-1)
            sparse = torch.zeros_like(weights)
            sparse.scatter_(1, topk_idx, topk_vals)
            # Re-normalise the sparse weights
            sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
            # Straight-through: forward uses sparse, backward uses dense
            weights = weights + (sparse - weights).detach()

        return weights

    def load_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss that penalises uneven expert utilisation.

        Computes the coefficient of variation of mean routing weights across
        experts. A perfectly balanced router yields CV ≈ 0.  Also updates
        the running ``_expert_counts`` buffer for diagnostics.

        Args:
            weights: Expert weights [batch, num_experts] from the last forward.

        Returns:
            Scalar loss tensor.
        """
        mean_weights = weights.mean(dim=0)  # [num_experts]

        # Update running utilisation counts
        with torch.no_grad():
            self._expert_counts += mean_weights.detach()
            self._total_routes += 1

        cv = mean_weights.std() / (mean_weights.mean() + 1e-8)
        return cv

    @torch.no_grad()
    def utilisation_stats(self) -> torch.Tensor:
        """Return normalised per-expert utilisation over training so far."""
        if self._total_routes == 0:
            return torch.zeros(self.num_experts)
        return self._expert_counts / self._total_routes

class HiveMind(nn.Module):
    def __init__(self, observation_dim: int = 64, pretrained: bool = False):
        super().__init__()
        self.logger = setup_logger("HiveMind")
        self.observation_dim = observation_dim

        # 1. Shared Latent Space / Data Layer
        self.log_encoder = LogEncoder(node_feature_dim=observation_dim)

        # 2. Evolving Threat Intelligence Database
        self.threat_intel = ThreatIntelDB(embedding_dim=observation_dim)

        # 3. Expert Clusters (19 experts across 7 clusters)

        # Cluster A: Recon
        self.expert_cartographer = Agent_Cartographer(observation_dim, action_dim=observation_dim)
        self.expert_deepscope = Agent_DeepScope(observation_dim, action_dim=10)
        self.expert_chronos = Agent_Chronos(1, action_dim=1)

        # Cluster B: Attack
        self.expert_sentinel = Agent_Sentinel(observation_dim, action_dim=2)
        self.expert_payloadgen = Agent_PayloadGen(observation_dim, action_dim=128)
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

        # Cluster F: Blue Team Detection Stack
        self.expert_waf = Agent_WAF(observation_dim, action_dim=2)
        self.expert_edr = Agent_EDR(observation_dim, action_dim=2)
        self.expert_siem = Agent_SIEM(observation_dim, action_dim=2)
        self.expert_ids = Agent_IDS(observation_dim, action_dim=2)

        # Cluster G: Red Team Pre-Attack Booster
        self.expert_booster = Agent_PreAttackBooster(
            observation_dim, action_dim=observation_dim
        )
        # Wire the booster to the blue team so it can adversarially refine
        self.expert_booster.register_blue_team([
            self.expert_waf, self.expert_edr, self.expert_siem, self.expert_ids,
        ])

        # Order matters for indexing in GatingNetwork outputs
        self.experts = nn.ModuleList([
            self.expert_cartographer,  # 0   Recon
            self.expert_deepscope,     # 1   Recon
            self.expert_chronos,       # 2   Recon
            self.expert_payloadgen,    # 3   Attack
            self.expert_mutator,       # 4   Attack
            self.expert_sentinel,      # 5   Attack
            self.expert_mimic,         # 6   Post-Exploit
            self.expert_ghost,         # 7   Post-Exploit
            self.expert_stego,         # 8   Post-Exploit
            self.expert_cleaner,       # 9   Post-Exploit
            self.expert_tarpit,        # 10  Active Defense
            self.expert_feedback,      # 11  Kill Chain
            self.expert_flashbang,     # 12  Kill Chain
            self.expert_glasshouse,    # 13  Kill Chain
            self.expert_waf,           # 14  Blue Team
            self.expert_edr,           # 15  Blue Team
            self.expert_siem,          # 16  Blue Team
            self.expert_ids,           # 17  Blue Team
            self.expert_booster,       # 18  Red Booster
        ])

        # 4. Gating Mechanism
        self.gating_network = GatingNetwork(observation_dim, num_experts=len(self.experts))

        # 5. Foundation / Knowledge Bootstrap
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

    def compute_global_state(self, data) -> torch.Tensor:
        """Compute the mean-pooled global state from a PyG Data object."""
        if data.x.size(0) > 0:
            return torch.mean(data.x, dim=0, keepdim=True)
        return torch.zeros(1, self.observation_dim, device=data.x.device)

    def forward(self, raw_logs: List[Dict], top_k: int = 3) -> Dict[str, Any]:
        """
        Main Forward Pass with fixed Quad-Strike Logic and sparse gating.

        The gating network produces top-k sparse weights with straight-through
        gradient estimation, ensuring only the selected experts receive
        gradient signal while keeping the full softmax differentiable.

        Args:
            raw_logs: List of dicts with keys 'src_ip', 'dst_ip', 'port', 'proto'.
            top_k: Number of top experts to activate per forward pass.

        Returns:
            Dictionary mapping result names to output tensors.
        """
        if not isinstance(raw_logs, list):
            raise TypeError(f"raw_logs must be a list, got {type(raw_logs)}")

        data = self.log_encoder.update(raw_logs)
        global_state = self.compute_global_state(data)

        # Sparse gating: top-k straight-through routing
        num_experts = len(self.experts)
        effective_k = max(1, min(top_k, num_experts))
        weights = self.gating_network(global_state, top_k=effective_k)

        # Determine which experts have non-zero routing weight
        active_indices = (weights[0] > 0).nonzero(as_tuple=True)[0].tolist()

        # SYNERGY LOGIC: Force-Enable Kill Chain using name-based lookup
        # instead of hardcoded indices to prevent breakage if expert order changes
        tarpit_active = any(
            self.experts[idx].name == "Tarpit" for idx in active_indices
        )
        if tarpit_active:
            synergy_names = {"FeedbackLoop", "Flashbang", "GlassHouse"}
            for i, expert in enumerate(self.experts):
                if expert.name in synergy_names and i not in active_indices:
                    active_indices.append(i)

        results = {}

        # Reset all to inactive
        for expert in self.experts:
            expert.is_active = False

        # Execute Active Experts
        for idx in active_indices:
            expert = self.experts[idx]
            expert.is_active = True

            try:
                if expert.name == "Cartographer":
                    out = expert(data.x, context=data.edge_index)
                    results["topology"] = out

                elif expert.name == "DeepScope":
                    out = expert(global_state, mask=None)
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
                    out = expert(global_state)
                    results["active_defense"] = out

                elif expert.name == "FeedbackLoop":
                    out = expert(global_state)
                    results["counter_strike"] = out

                elif expert.name == "Flashbang":
                    out = expert(global_state)
                    results["overload"] = out

                elif expert.name == "GlassHouse":
                    out = expert(global_state)
                    results["total_exposure"] = out

                # --- Blue Team Detection Stack ---
                elif expert.name == "WAF":
                    out = expert(global_state)
                    results["waf_verdict"] = out

                elif expert.name == "EDR":
                    out = expert(global_state)
                    results["edr_verdict"] = out

                elif expert.name == "SIEM":
                    out = expert(global_state)
                    results["siem_verdict"] = out

                elif expert.name == "IDS":
                    out = expert(global_state)
                    results["ids_verdict"] = out

                # --- Red Team Pre-Attack Booster ---
                elif expert.name == "PreAttackBooster":
                    # Feed the best available payload through the booster
                    payload = results.get(
                        "optimized_payload",
                        results.get("raw_payload", global_state),
                    )
                    payload_input = self.expert_booster.ensure_dimension(
                        payload, self.observation_dim
                    )
                    out = expert(payload_input)
                    results["hardened_payload"] = out

            except Exception as e:
                self.logger.error(f"Execution failed for {expert.name}: {e}")

        return results

    # ------------------------------------------------------------------
    # Co-Evolutionary Threat Intelligence
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evolve_threat_intel(self, results: Dict[str, Any]):
        """
        Feed forward-pass results into the ThreatIntelDB and update
        blue-team detectors to drive red/blue co-evolution.

        The database records:
        * **successful evasions** — payloads that the blue-team stack
          scored as "Allowed" (attack bank, for red-team imitation).
        * **caught payloads** — payloads that triggered at least one
          detector (defense bank, for blue-team pattern learning).

        Additionally, caught payloads are fed into the WAF signature bank
        so the WAF continuously learns from new attack patterns.

        Should be called after every ``forward()`` during training.
        """
        # Determine the best available payload embedding
        payload = results.get(
            "hardened_payload",
            results.get("optimized_payload",
                        results.get("raw_payload")),
        )
        if payload is None:
            return

        payload_flat = payload.view(payload.size(0), -1)
        payload_flat = self.expert_booster.ensure_dimension(
            payload_flat, self.observation_dim
        )

        # Aggregate blue-team verdicts (P(Blocked) from each detector)
        blue_keys = ("waf_verdict", "edr_verdict", "siem_verdict", "ids_verdict")
        block_probs = []
        for key in blue_keys:
            if key in results:
                logits = results[key]
                probs = F.softmax(logits, dim=-1)
                block_probs.append(probs[:, 0])  # P(Blocked) is index 0

        if not block_probs:
            return

        # A payload is "caught" if ANY detector gives P(Blocked) > 0.5
        max_block = torch.stack(block_probs, dim=0).max(dim=0).values  # [B]
        evaded = max_block < 0.5
        caught = ~evaded

        if evaded.any():
            self.threat_intel.record_attack_success(payload_flat[evaded])
        if caught.any():
            self.threat_intel.record_attack_failure(payload_flat[caught])
            # Feed caught payloads into WAF signature bank for continuous
            # adaptation — WAF learns from every new attack pattern it sees.
            caught_encoded = self.expert_waf.encoder(payload_flat[caught])
            self.expert_waf.update_signatures(caught_encoded)

        evasion_rate = float(evaded.float().mean().item())
        self.threat_intel.step_generation(evasion_rate)
