import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union, cast
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert
from hive_zero_core.knowledge.exploit_db import ExploitDB
from hive_zero_core.knowledge.mitre_kb import MitreKnowledgeBase
import random
import re

class Agent_Sentinel(BaseExpert):
    """
    Expert 6: Local WAF Verifier (OWASP CRS Regex + BERT Ensemble)
    """
import random
import math

class Agent_Sentinel(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=hidden_dim, output_hidden_states=True)
            self.head = nn.Linear(hidden_dim, 2)

            self.rules = [
                r"(?i)<script",
                r"(?i)union.*select",
                r"(?i)/etc/passwd",
                r"(?i)\.\./",
                r"(?i)eval\(",
            ]
            self.head1 = nn.Linear(hidden_dim, 2)
            self.head2 = nn.Linear(hidden_dim, 2)
            self.head3 = nn.Linear(hidden_dim, 2)
        except Exception as e:
            self.logger.error(f"Failed to load Sentinel: {e}")
            raise e

    def check_waf(self, payload_str: str) -> float:
        for rule in self.rules:
            if re.search(rule, payload_str):
                return 1.0
        return 0.0

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] > 1:
             outputs = self.backbone(inputs_embeds=x)
        else:
             outputs = self.backbone(input_ids=x.long())

        features = F.relu(outputs.logits)
        probs = torch.softmax(self.head(features), dim=-1)
        return probs

class Agent_PayloadGen(BaseExpert):
    """
    Expert 4: Real RAG Payload Generator
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "t5-small", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.db = ExploitDB()
            self.mitre = MitreKnowledgeBase()
            self.context_encoder = nn.Linear(observation_dim, self.model.config.d_model)
        features = F.relu(outputs.logits)
        probs1 = torch.softmax(self.head1(features), dim=-1)
        probs2 = torch.softmax(self.head2(features), dim=-1)
        probs3 = torch.softmax(self.head3(features), dim=-1)
        return (probs1 + probs2 + probs3) / 3.0

class Agent_PayloadGen(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "t5-small", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.db_keys = nn.Parameter(torch.randn(10, 64), requires_grad=False)
            self.db_values = nn.Parameter(torch.randint(0, 1000, (10, 20)), requires_grad=False)
            self.query_proj = nn.Linear(observation_dim, 64)
        except Exception as e:
            self.logger.error(f"Failed to load PayloadGen: {e}")
            raise e

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        soft_prompt = self.context_encoder(x).unsqueeze(1)
        encoder_out = self.model.encoder(inputs_embeds=soft_prompt)
        outputs = self.model.generate(encoder_outputs=encoder_out, max_length=64)
        return outputs

    def generate_real_exploit(self, target_service: str) -> str:
        """
        Non-differentiable generation for actual execution.
        """
        techniques = self.mitre.map_service_to_technique(target_service)
        query = f"{target_service} {' '.join(techniques)} exploit"

        docs = self.db.query(query, n_results=1)
        if not docs:
            return "NO_EXPLOIT_FOUND"

        # ExploitDB.query returns list of dicts with 'metadata' key
        # Metadata contains 'code'
        metadata = docs[0].get('metadata', {})
        template = metadata.get('code', "NO_CODE_IN_METADATA")

        return template

class Agent_Mutator(BaseExpert):
    """
    Expert 5: WAF Bypass Mutator
    """
    def _retrieve(self, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query, self.db_keys.t())
        indices = torch.argmax(scores, dim=1)
        return self.db_values[indices]

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.query_proj(x)
        retrieved_tokens = self._retrieve(query)
        encoder_out = self.model.encoder(input_ids=retrieved_tokens.long())
        return self.model.generate(encoder_outputs=encoder_out, max_length=64)

class Agent_Mutator(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, sentinel_expert: BaseExpert, generator_expert: BaseExpert, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Mutator", hidden_dim=hidden_dim)
        self.sentinel = sentinel_expert
        self.generator = generator_expert

        self.mutations = [
            lambda s: s.replace(" ", "/**/"),
            lambda s: s.replace("SELECT", "SeLeCt"),
            lambda s: s.replace("<script>", "%3Cscript%3E"),
            lambda s: s + " --",
            lambda s: s.replace("UNION", "/*!UNION*/")
        ]

    def _mutate_string(self, payload: str) -> str:
        m = random.choice(self.mutations)
        try:
            return m(payload)
        except:
            return payload
    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0

    def _evaluate(self, embedding):
        with torch.no_grad():
            # Pass dummy mask/context
            probs = self.sentinel._forward_impl(embedding, None, None)
        return probs[:, 1].item()

    def _expand(self, node):
        for _ in range(3):
            noise = torch.randn_like(node.state) * 0.1
            child_state = node.state + noise
            child = self.Node(child_state, parent=node)
            node.children.append(child)

    def _run_mcts(self, root_embedding, simulations=5):
        root = self.Node(root_embedding)
        for _ in range(simulations):
            node = root
            if node.children:
                 node = max(node.children, key=lambda c: c.value / (c.visits + 1e-6) + 2.0 * math.sqrt(math.log(root.visits + 1) / (c.visits + 1e-6)))
            if node.visits > 0 or not node.children:
                self._expand(node)
                if node.children: node = random.choice(node.children)
            score = self._evaluate(node.state)
            curr = node
            while curr:
                curr.visits += 1
                curr.value += score
                curr = curr.parent
        best_child = max(root.children, key=lambda c: c.visits) if root.children else root
        return best_child.state

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            initial_token_ids = self.generator._forward_impl(x, context, mask)

        # Access internals of Sentinel (assuming Agent_Sentinel type)
        # Using type ignore or cast for mypy
        sentinel_impl = cast(Agent_Sentinel, self.sentinel)
        embed_layer = sentinel_impl.backbone.get_input_embeddings()
        vocab_size = sentinel_impl.backbone.config.vocab_size

        initial_token_ids = torch.clamp(initial_token_ids.long(), 0, vocab_size - 1)
        if initial_token_ids.shape[-1] > 512:
             initial_token_ids = initial_token_ids[:, :512]

        current_embeddings = embed_layer(initial_token_ids).clone().detach()
        current_embeddings.requires_grad_(True)

        optimizer = optim.SGD([current_embeddings], lr=0.1)

        for i in range(5):
            optimizer.zero_grad()
            probs = self.sentinel._forward_impl(current_embeddings, None, None)
            loss = -torch.log(probs[:, 1] + 1e-8).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([current_embeddings], max_norm=1.0)
            optimizer.step()

        return current_embeddings.detach()
        current_embeddings = embed_layer(initial_token_ids).detach()
        if current_embeddings.size(0) == 1:
            optimized_embeddings = self._run_mcts(current_embeddings)
        else:
            optimized_embeddings = current_embeddings
        return optimized_embeddings
