import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union, cast
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert
import random
import math

class Agent_Sentinel(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=hidden_dim, output_hidden_states=True)
            self.head1 = nn.Linear(hidden_dim, 2)
            self.head2 = nn.Linear(hidden_dim, 2)
            self.head3 = nn.Linear(hidden_dim, 2)
        except Exception as e:
            self.logger.error(f"Failed to load Sentinel: {e}")
            raise e

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-1] > 1:
             outputs = self.backbone(inputs_embeds=x)
        else:
             outputs = self.backbone(input_ids=x.long())
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

        current_embeddings = embed_layer(initial_token_ids).detach()
        if current_embeddings.size(0) == 1:
            optimized_embeddings = self._run_mcts(current_embeddings)
        else:
            optimized_embeddings = current_embeddings
        return optimized_embeddings
