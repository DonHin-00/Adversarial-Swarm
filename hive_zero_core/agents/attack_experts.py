import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert
import random
import math

class Agent_Sentinel(BaseExpert):
    """
    Expert 6: Ensemble Discriminator
    Runs 3 distinct BERT models (mocked as one model with 3 heads/perturbations for prototype)
    and votes on the outcome.
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensemble of 3 heads on top of shared base
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=hidden_dim, output_hidden_states=True)

            # 3 Expert Heads
            self.head1 = nn.Linear(hidden_dim, 2)
            self.head2 = nn.Linear(hidden_dim, 2)
            self.head3 = nn.Linear(hidden_dim, 2)

        except Exception as e:
            self.logger.error(f"Failed to load Sentinel: {e}")
            raise e

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Base Features
        # x: [Batch, Seq, Dim] or [Batch, Seq]

        if x.dim() == 3 and x.shape[-1] > 1:
             outputs = self.backbone(inputs_embeds=x)
        else:
             outputs = self.backbone(input_ids=x.long())

        # Backbone outputs [Batch, hidden_dim] because num_labels=hidden_dim
        features = F.relu(outputs.logits)

        # 2. Ensemble Voting
        logits1 = self.head1(features)
        logits2 = self.head2(features)
        logits3 = self.head3(features)

        probs1 = torch.softmax(logits1, dim=-1)
        probs2 = torch.softmax(logits2, dim=-1)
        probs3 = torch.softmax(logits3, dim=-1)

        # Average probability (Soft Voting)
        avg_probs = (probs1 + probs2 + probs3) / 3.0

        return avg_probs

class Agent_PayloadGen(BaseExpert):
    """
    Expert 4: RAG-Enhanced Payload Generator
    Retrieves "Exploit Templates" from a mock VectorDB before generation.
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "t5-small", hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Mock Vector DB (Keys: Embeddings, Values: Template Tokens)
            # 10 templates, embedding dim 64
            self.db_keys = nn.Parameter(torch.randn(10, 64), requires_grad=False)
            self.db_values = nn.Parameter(torch.randint(0, 1000, (10, 20)), requires_grad=False) # 20 tokens long

            self.query_proj = nn.Linear(observation_dim, 64)

        except Exception as e:
            self.logger.error(f"Failed to load PayloadGen: {e}")
            raise e

    def _retrieve(self, query: torch.Tensor) -> torch.Tensor:
        # query: [B, 64]
        scores = torch.matmul(query, self.db_keys.t())
        indices = torch.argmax(scores, dim=1) # [B]

        # Retrieve templates
        templates = self.db_values[indices] # [B, 20]
        return templates

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. RAG Retrieval
        query = self.query_proj(x)
        retrieved_tokens = self._retrieve(query)

        # 2. Generation with Context
        # Using mock encoder outputs to condition generation
        # We simulate this by passing retrieval embedding as 'encoder_outputs'

        # Embed retrieval
        # T5 encoder expects input_ids.
        # We need to run the full encoder stack to get valid encoder_outputs
        encoder_out = self.model.encoder(input_ids=retrieved_tokens.long())

        # Generate
        outputs = self.model.generate(
            encoder_outputs=encoder_out,
            max_length=64
        )

        return outputs

class Agent_Mutator(BaseExpert):
    """
    Expert 5: MCTS Mutator
    Uses Monte Carlo Tree Search to plan discrete mutations.
    """
    def __init__(self, observation_dim: int, action_dim: int, sentinel_expert: BaseExpert, generator_expert: BaseExpert, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Mutator", hidden_dim=hidden_dim)
        self.sentinel = sentinel_expert
        self.generator = generator_expert

    class Node:
        def __init__(self, state, parent=None):
            self.state = state # Embedding tensor
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0

    def _evaluate(self, embedding):
        # We need to detach embedding to avoid gradient graph issues in MCTS?
        # Actually MCTS is non-differentiable search.
        with torch.no_grad():
            probs = self.sentinel._forward_impl(embedding, context=None)
        return probs[:, 1].item() # P(Allowed)

    def _expand(self, node):
        for _ in range(3):
            # Noise in embedding space as "actions"
            noise = torch.randn_like(node.state) * 0.1
            child_state = node.state + noise
            child = self.Node(child_state, parent=node)
            node.children.append(child)

    def _run_mcts(self, root_embedding, simulations=5):
        root = self.Node(root_embedding)

        for _ in range(simulations):
            # Selection
            node = root
            # Simple Selection: just pick random if exists or expand
            if node.children:
                 # UCB
                 node = max(node.children, key=lambda c: c.value / (c.visits + 1e-6) + 2.0 * math.sqrt(math.log(root.visits + 1) / (c.visits + 1e-6)))

            # Expansion
            if node.visits > 0 or not node.children:
                self._expand(node)
                if node.children:
                    node = random.choice(node.children)

            # Simulation/Evaluation
            score = self._evaluate(node.state)

            # Backprop
            curr = node
            while curr:
                curr.visits += 1
                curr.value += score
                curr = curr.parent

        best_child = max(root.children, key=lambda c: c.visits) if root.children else root
        return best_child.state

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Initial Generation
        with torch.no_grad():
            initial_token_ids = self.generator._forward_impl(x, context)

        embed_layer = self.sentinel.backbone.get_input_embeddings()
        vocab_size = self.sentinel.backbone.config.vocab_size
        initial_token_ids = torch.clamp(initial_token_ids.long(), 0, vocab_size - 1)

        if initial_token_ids.shape[-1] > 512:
             initial_token_ids = initial_token_ids[:, :512]

        current_embeddings = embed_layer(initial_token_ids).detach()

        # 2. MCTS Optimization
        if current_embeddings.size(0) == 1:
            optimized_embeddings = self._run_mcts(current_embeddings)
        else:
            # Fallback for batch > 1
            optimized_embeddings = current_embeddings

        return optimized_embeddings
