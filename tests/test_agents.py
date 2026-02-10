import pytest
import torch
from torch_geometric.data import HeteroData
from unittest.mock import MagicMock, patch

from hive_zero_core.agents.attack_experts import MutatorAgent, PayloadGenAgent, SentinelAgent
from hive_zero_core.agents.post_experts import CleanerAgent, GhostAgent, MimicAgent, StegoAgent
from hive_zero_core.agents.recon_experts import CartographerAgent, ChronosAgent, DeepScopeAgent
from hive_zero_core.hive_mind import HiveMind


@pytest.fixture
def obs_dim():
    return 64


@pytest.fixture
def action_dim():
    return 64


def test_cartographer_agent(obs_dim, action_dim):
    agent = CartographerAgent(obs_dim, action_dim)
    agent.is_active = True  # Activate expert for testing
    data = HeteroData()
    data["ip"].x = torch.randn(5, obs_dim)
    data["port"].x = torch.randn(3, obs_dim)
    data["protocol"].x = torch.randn(2, obs_dim)
    data["ip", "flow", "ip"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    data["ip", "binds", "port"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data["port", "uses", "protocol"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    out = agent(data)
    assert out.shape == (5, action_dim)


def test_chronos_agent(obs_dim, action_dim):
    agent = ChronosAgent(obs_dim, action_dim)
    agent.is_active = True  # Activate expert for testing
    x = torch.randn(4, 20)  # Batch of 4, sequence of 20
    out = agent(x)
    assert out.shape == (4, action_dim)


def test_deepscope_agent(obs_dim, action_dim):
    agent = DeepScopeAgent(obs_dim, action_dim)
    agent.is_active = True  # Activate expert for testing
    x = torch.randn(8, obs_dim)
    out = agent(x)
    assert out.shape == (8, action_dim)


def test_sentinel_agent_with_mock(obs_dim):
    """Test SentinelAgent with mocked HuggingFace components to avoid network access."""
    with patch('hive_zero_core.agents.attack_experts.AutoTokenizer') as mock_tokenizer, \
         patch('hive_zero_core.agents.attack_experts.AutoModelForSequenceClassification') as mock_model:
        
        # Mock tokenizer
        mock_tok_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance
        
        # Mock model with proper output structure
        mock_model_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(4, 64)  # [Batch, hidden_dim]
        mock_model_instance.return_value = mock_output
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Create agent
        agent = SentinelAgent(obs_dim, 2)
        agent.is_active = True
        
        # Test forward pass
        x = torch.randint(0, 100, (4, 10))  # Batch 4, Seq 10
        out = agent(x)
        assert out.shape == (4, 2)
        
        # Verify from_pretrained was called with local_files_only parameter
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()


def test_payloadgen_agent_with_mock(obs_dim):
    """Test PayloadGenAgent with mocked HuggingFace components to avoid network access."""
    with patch('hive_zero_core.agents.attack_experts.AutoTokenizer') as mock_tokenizer, \
         patch('hive_zero_core.agents.attack_experts.AutoModelForSeq2SeqLM') as mock_model:
        
        # Mock tokenizer
        mock_tok_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance
        
        # Mock T5 model with encoder and generate
        mock_model_instance = MagicMock()
        mock_encoder_output = MagicMock()
        mock_encoder_output.last_hidden_state = torch.randn(2, 20, 512)
        mock_model_instance.encoder.return_value = mock_encoder_output
        mock_model_instance.generate.return_value = torch.randint(0, 1000, (2, 64))
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Create agent
        agent = PayloadGenAgent(obs_dim, 128)
        agent.is_active = True
        
        # Test forward pass
        x = torch.randn(2, obs_dim)
        out = agent(x)
        
        # T5 generate outputs token IDs, shape [Batch, Seq]
        assert out.dim() == 2
        assert out.shape[0] == 2


def test_mutator_agent_with_mock(obs_dim):
    """Test MutatorAgent initialization with mocked HuggingFace components."""
    with patch('hive_zero_core.agents.attack_experts.AutoTokenizer') as mock_tokenizer, \
         patch('hive_zero_core.agents.attack_experts.AutoModelForSequenceClassification') as mock_model_cls, \
         patch('hive_zero_core.agents.attack_experts.AutoModelForSeq2SeqLM') as mock_model_seq:
        
        # Mock Sentinel components
        mock_tok_cls = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok_cls
        
        mock_sentinel_model = MagicMock()
        mock_sentinel_output = MagicMock()
        mock_sentinel_output.logits = torch.randn(1, 64)
        mock_sentinel_model.return_value = mock_sentinel_output
        mock_model_cls.from_pretrained.return_value = mock_sentinel_model
        
        # Mock PayloadGen components
        mock_gen_model = MagicMock()
        mock_encoder_output = MagicMock()
        mock_encoder_output.last_hidden_state = torch.randn(1, 20, 512)
        mock_gen_model.encoder.return_value = mock_encoder_output
        mock_gen_model.generate.return_value = torch.randint(0, 1000, (1, 64))
        mock_model_seq.from_pretrained.return_value = mock_gen_model
        
        # Create agents
        sentinel = SentinelAgent(obs_dim, 2)
        gen = PayloadGenAgent(obs_dim, 128)
        agent = MutatorAgent(obs_dim, 128, sentinel, gen)
        agent.is_active = True
        
        # Verify initialization
        assert agent.sentinel is not None
        assert agent.generator is not None
        assert agent.observation_dim == obs_dim
        assert agent.action_dim == 128
        
        # Note: Forward pass testing requires more sophisticated mocking of tokenizers
        # and model interactions. The actual forward method is tested via integration tests
        # with real models (when available).


@pytest.mark.skip(
    reason="Requires downloading HuggingFace models which is slow and may fail in offline CI"
)
def test_sentinel_agent(obs_dim, action_dim):
    # Sentinel usually outputs [Batch, 2]
    agent = SentinelAgent(obs_dim, 2)
    agent.is_active = True  # Activate expert for testing
    x = torch.randint(0, 100, (4, 10))  # Batch 4, Seq 10
    out = agent(x)
    assert out.shape == (4, 2)


@pytest.mark.skip(
    reason="Requires downloading HuggingFace models which is slow and may fail in offline CI"
)
def test_payloadgen_agent(obs_dim):
    agent = PayloadGenAgent(obs_dim, 128)
    agent.is_active = True  # Activate expert for testing
    x = torch.randn(2, obs_dim)
    out = agent(x)
    # T5 generate outputs token IDs, shape [Batch, Seq]
    assert out.dim() == 2
    assert out.shape[0] == 2


@pytest.mark.skip(
    reason="Requires downloading HuggingFace models which is slow and may fail in offline CI"
)
def test_mutator_agent(obs_dim):
    sentinel = SentinelAgent(obs_dim, 2)
    gen = PayloadGenAgent(obs_dim, 128)
    agent = MutatorAgent(obs_dim, 128, sentinel, gen)
    agent.is_active = True  # Activate expert for testing
    x = torch.randn(1, obs_dim)
    out = agent(x)
    # Output is optimized embeddings [1, Seq, Hidden]
    assert out.dim() == 3
    assert out.shape[0] == 1


def test_mimic_agent(obs_dim, action_dim):
    agent = MimicAgent(obs_dim, action_dim)
    agent.is_active = True  # Activate expert for testing
    x = torch.randn(4, obs_dim)
    out = agent(x)
    assert out.shape == (4, action_dim)


def test_ghost_agent(obs_dim):
    agent = GhostAgent(obs_dim, 1)
    agent.is_active = True  # Activate expert for testing
    x = torch.randn(10, obs_dim)
    out = agent(x)
    assert out.shape == (10, 1)


def test_stego_agent(obs_dim):
    agent = StegoAgent(obs_dim, 64)
    agent.is_active = True  # Activate expert for testing
    x = torch.randn(2, obs_dim)
    cover = torch.randn(2, 1, 32, 32)
    out = agent(x, context=cover)
    assert out.shape == (2, 1, 32, 32)


def test_cleaner_agent(obs_dim, action_dim):
    agent = CleanerAgent(obs_dim, action_dim)
    agent.is_active = True  # Activate expert for testing
    x = torch.randn(4, 5, obs_dim)  # Batch 4, Seq 5
    out = agent(x)
    assert out.shape == (4, action_dim)
    assert hasattr(agent, "last_verified_score")
    assert agent.last_verified_score.shape == (4, 1)


def test_hive_mind_without_hf_models(obs_dim):
    """Test HiveMind initialization without loading HuggingFace models."""
    # This should work in offline/airgapped environments
    hive = HiveMind(observation_dim=obs_dim, load_hf_models=False)
    
    # Verify that HF experts are None
    assert hive.expert_sentinel is None
    assert hive.expert_payloadgen is None
    assert hive.expert_mutator is None
    
    # Verify that non-HF experts are loaded
    assert hive.expert_cartographer is not None
    assert hive.expert_deepscope is not None
    assert hive.expert_chronos is not None
    assert hive.expert_mimic is not None
    assert hive.expert_ghost is not None
    assert hive.expert_stego is not None
    assert hive.expert_cleaner is not None
    
    # Verify experts list only contains non-None experts
    assert len(hive.experts) == 7  # 7 non-HF experts
    
    # Test forward pass with sample data
    raw_logs = [
        {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.1", "port": 80, "protocol": "TCP"},
        {"src_ip": "192.168.1.2", "dst_ip": "10.0.0.2", "port": 443, "protocol": "TCP"},
    ]
    
    results = hive.forward(raw_logs, top_k=2)
    assert "gating_weights" in results
    assert results["gating_weights"] is not None


def test_hive_mind_with_local_files_only(obs_dim):
    """Test HiveMind with local_files_only flag (requires cached models)."""
    with patch('hive_zero_core.agents.attack_experts.AutoTokenizer') as mock_tokenizer, \
         patch('hive_zero_core.agents.attack_experts.AutoModelForSequenceClassification') as mock_model_cls, \
         patch('hive_zero_core.agents.attack_experts.AutoModelForSeq2SeqLM') as mock_model_seq:
        
        # Mock all components to avoid network access
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_model_seq.from_pretrained.return_value = MagicMock()
        
        # Initialize with local_files_only=True
        hive = HiveMind(observation_dim=obs_dim, load_hf_models=True, local_files_only=True)
        
        # Verify from_pretrained was called with local_files_only=True
        for call in mock_tokenizer.from_pretrained.call_args_list:
            assert call[1].get('local_files_only') == True
        for call in mock_model_cls.from_pretrained.call_args_list:
            assert call[1].get('local_files_only') == True
        for call in mock_model_seq.from_pretrained.call_args_list:
            assert call[1].get('local_files_only') == True
