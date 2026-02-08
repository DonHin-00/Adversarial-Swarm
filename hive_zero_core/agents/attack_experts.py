import torch  # noqa: I001
import torch.nn as nn  # noqa: PLR0402
import torch.optim as optim  # noqa: F401, PLR0402
import torch.nn.functional as F  # noqa: N812
from typing import Optional, Dict, Tuple, List, Union, cast  # noqa: F401
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert
from hive_zero_core.knowledge.exploit_db import ExploitDB
from hive_zero_core.knowledge.mitre_kb import MitreKnowledgeBase
import random
import math
import re


class Agent_Sentinel(BaseExpert):  # noqa: N801
    """
    Expert 6: Local WAF Verifier (OWASP CRS Regex + BERT Ensemble)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, Union
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from hive_zero_core.agents.base_expert import BaseExpert

class SentinelAgent(BaseExpert):
    """
    Expert 6: Stateful Ensemble Discriminator
    Uses a history GRU to model alert thresholds across sequences.
    """

    def __init__(
        self, observation_dim: int, action_dim: int, model_name: str = "prajjwal1/bert-tiny", hidden_dim: int = 64
    ):
        super().__init__(observation_dim, action_dim, name="Sentinel", hidden_dim=hidden_dim)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=hidden_dim, output_hidden_states=True
            )
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
            # Fallback if model load fails (e.g. no internet in restricted CI)
            print(f"Failed to load Sentinel model: {e}")
            self.backbone = None  # Handle in forward

    def check_waf(self, payload_str: str) -> float:
        for rule in self.rules:
            if re.search(rule, payload_str):
                return 1.0
        return 0.0

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.backbone is None:
            return torch.zeros(x.size(0), 2, device=x.device)

        if x.dim() == 3 and x.shape[-1] > 1:  # noqa: PLR2004
            outputs = self.backbone(inputs_embeds=x)
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=hidden_dim, output_hidden_states=True)

            # Ensemble Heads
            self.head1 = nn.Linear(hidden_dim, 2)
            self.head2 = nn.Linear(hidden_dim, 2)
            self.head3 = nn.Linear(hidden_dim, 2)

            # Stateful Threshold Modeling
            self.history_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        except (ImportError, RuntimeError, OSError) as e:
            self.logger.error(f"Failed to load Sentinel model/tokenizer: {str(e)}")
            raise RuntimeError(f"Sentinel initialization failed: {str(e)}")

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [Batch, Seq, Dim] or [Batch, Seq]
        if x.dim() == 3 and x.shape[-1] > 1:
             outputs = self.backbone(inputs_embeds=x)
        else:
            outputs = self.backbone(input_ids=x.long())

        features = F.relu(outputs.logits)
        probs = torch.softmax(self.head(features), dim=-1)
        return probs


class Agent_PayloadGen(BaseExpert):  # noqa: N801
    """
    Expert 4: Real RAG Payload Generator
    """

    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "t5-small", hidden_dim: int = 64):
        features = F.relu(outputs.logits) # [Batch, hidden_dim]

        # Stateful refinement
        h_seq = features.unsqueeze(1)
        h_out, _ = self.history_gru(h_seq)
        refined_features = h_out.squeeze(1)

        # Ensemble Voting
        logits1 = self.head1(refined_features)
        logits2 = self.head2(refined_features)
        logits3 = self.head3(refined_features)

        avg_probs = (torch.softmax(logits1, -1) + torch.softmax(logits2, -1) + torch.softmax(logits3, -1)) / 3.0
        return avg_probs

class PayloadGenAgent(BaseExpert):
    """
    Expert 4: RAG-Enhanced Payload Generator
    Retrieves "Exploit Templates" from a VectorDB before generation.
    """
    def __init__(self, observation_dim: int, action_dim: int, model_name: str = "t5-small",
                 hidden_dim: int = 64, rag_db: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__(observation_dim, action_dim, name="PayloadGen", hidden_dim=hidden_dim)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.db = ExploitDB()
            self.mitre = MitreKnowledgeBase()
            # Map observation to T5 embedding dimension
            self.context_encoder = nn.Linear(observation_dim, self.model.config.d_model)
        except Exception as e:
            print(f"Failed to load PayloadGen: {e}")
            self.model = None

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.model is None:
            return torch.zeros(x.size(0), 10, device=x.device)  # Dummy output

        # Project observation to soft prompt embeddings
        soft_prompt = self.context_encoder(x).unsqueeze(1)

        # We need to generate token IDs.
        # Using model.generate() is non-differentiable usually.
        # But here we return the 'generated' output?
        # Actually, for training we might use encoder outputs or something.
        # This implementation seems to return generate() output which are token IDs (long).

        encoder_out = self.model.encoder(inputs_embeds=soft_prompt)
        outputs = self.model.generate(encoder_outputs=encoder_out, max_length=64)
        return outputs

    def generate_real_exploit(self, target_service: str) -> str:
        """
        Non-differentiable generation for actual execution.
        """
        techniques = self.mitre.map_service_to_technique(target_service)
        query = f"{target_service} {' '.join(techniques)} exploit"
            # Vector DB (Keys: Embeddings, Values: Template Tokens)
            if rag_db:
                self.db_keys = nn.Parameter(rag_db["keys"], requires_grad=False)
                self.db_values = nn.Parameter(rag_db["values"], requires_grad=False)
            else:
                # Default mock DB
                self.db_keys = nn.Parameter(torch.randn(10, 64), requires_grad=False)
                self.db_values = nn.Parameter(torch.randint(0, 1000, (10, 20)), requires_grad=False)

        docs = self.db.query(query, n_results=1)
        if not docs:
            return "NO_EXPLOIT_FOUND"

        metadata = docs[0].get("metadata", {})
        template = metadata.get("code", "NO_CODE_IN_METADATA")
        except (ImportError, RuntimeError, OSError) as e:
            self.logger.error(f"Failed to load PayloadGen model/tokenizer: {str(e)}")
            raise RuntimeError(f"PayloadGen initialization failed: {str(e)}")

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

        return template


class Agent_Mutator(BaseExpert):  # noqa: N801
    """
    Expert 5: WAF Bypass Mutator
class MutatorAgent(BaseExpert):
    """
    Expert 5: Hybrid Search Mutator
    Implements Inference-Time Search using gradient descent + noise injection.
    Optimizes payloads to evade SentinelAgent.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        sentinel_expert: BaseExpert,
        generator_expert: BaseExpert,
        hidden_dim: int = 64,
    ):
        super().__init__(observation_dim, action_dim, name="Mutator", hidden_dim=hidden_dim)
        self.sentinel = sentinel_expert
        self.generator = generator_expert

        self.mutations = [
            lambda s: s.replace(" ", "/**/"),
            lambda s: s.replace("SELECT", "SeLeCt"),
            lambda s: s.replace("<script>", "%3Cscript%3E"),
            lambda s: s + " --",
            lambda s: s.replace("UNION", "/*!UNION*/"),
        ]

    def _mutate_string(self, payload: str) -> str:
        m = random.choice(self.mutations)
        try:
            return m(payload)
        except:  # noqa: E722
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
                node = max(
                    node.children,
                    key=lambda c: (
                        c.value / (c.visits + 1e-6) + 2.0 * math.sqrt(math.log(root.visits + 1) / (c.visits + 1e-6))
                    ),
                )
            if node.visits > 0 or not node.children:
                self._expand(node)
                if node.children:
                    node = random.choice(node.children)  # noqa: E701
            score = self._evaluate(node.state)
            curr = node
            while curr:
                curr.visits += 1
                curr.value += score
                curr = curr.parent
        best_child = max(root.children, key=lambda c: c.visits) if root.children else root
        return best_child.state

    def _forward_impl(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            initial_token_ids = self.generator._forward_impl(x, context, mask)

        # Access internals of Sentinel (assuming Agent_Sentinel type)
        sentinel_impl = cast(Agent_Sentinel, self.sentinel)
        if sentinel_impl.backbone is None:
            return torch.zeros(x.size(0), self.action_dim, device=x.device)

        embed_layer = sentinel_impl.backbone.get_input_embeddings()
        vocab_size = sentinel_impl.backbone.config.vocab_size

        initial_token_ids = torch.clamp(initial_token_ids.long(), 0, vocab_size - 1)
        if initial_token_ids.shape[-1] > 512:  # noqa: PLR2004
            initial_token_ids = initial_token_ids[:, :512]

        current_embeddings = embed_layer(initial_token_ids).clone().detach()

        if current_embeddings.size(0) == 1:
            optimized_embeddings = self._run_mcts(current_embeddings)
        else:
            optimized_embeddings = current_embeddings

        return optimized_embeddings
    def _inner_loop_search(self, embedding: torch.Tensor, iterations: int = 5) -> torch.Tensor:
        # embedding: [1, seq, dim]
        optimized_emb = embedding.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([optimized_emb], lr=0.01)

        for _ in range(iterations):
            optimizer.zero_grad()
            # We want to MAXIMIZE probability of evasion (P(Allowed))
            probs = self.sentinel(optimized_emb)
            loss = -torch.log(probs[:, 1] + 1e-8).mean()
            loss.backward()
            optimizer.step()

            # Discrete Noise Injection (Hybrid)
            with torch.no_grad():
                noise = torch.randn_like(optimized_emb) * 0.05
                optimized_emb.add_(noise)

        return optimized_emb.detach()

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Base Generation
        with torch.no_grad():
            gen_out = self.generator(x, context)
            # Decode T5 tokens to text and re-encode to BERT tokens to avoid vocab mismatch
            # gen_out shape: [Batch, Seq]
            gen_text = self.generator.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
            # Re-encode for Sentinel (BERT)
            sentinel_inputs = self.sentinel.tokenizer(gen_text, return_tensors="pt", padding=True, truncation=True).to(x.device)
            initial_token_ids = sentinel_inputs["input_ids"]

        embed_layer = self.sentinel.backbone.get_input_embeddings()
        current_embeddings = embed_layer(initial_token_ids).detach()

        # 2. Inference-Time Search
        if current_embeddings.size(0) == 1:
            return self._inner_loop_search(current_embeddings)
        return current_embeddings
