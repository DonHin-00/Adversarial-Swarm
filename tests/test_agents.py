import torch
import pytest
from torch_geometric.data import HeteroData
from hive_zero_core.agents.recon_experts import CartographerAgent, ChronosAgent, DeepScopeAgent
from hive_zero_core.agents.attack_experts import SentinelAgent, PayloadGenAgent, MutatorAgent
from hive_zero_core.agents.post_experts import MimicAgent, GhostAgent, StegoAgent, CleanerAgent

@pytest.fixture
def obs_dim():
    return 64

@pytest.fixture
def action_dim():
    return 64

def test_cartographer_agent(obs_dim, action_dim):
    agent = CartographerAgent(obs_dim, action_dim)
    data = HeteroData()
    data['ip'].x = torch.randn(5, obs_dim)
    data['port'].x = torch.randn(3, obs_dim)
    data['protocol'].x = torch.randn(2, obs_dim)
    data['ip', 'flow', 'ip'].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    data['ip', 'binds', 'port'].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data['port', 'uses', 'protocol'].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    out = agent(data)
    assert out.shape == (5, action_dim)

def test_chronos_agent(obs_dim, action_dim):
    agent = ChronosAgent(obs_dim, action_dim)
    x = torch.randn(4, 20) # Batch of 4, sequence of 20
    out = agent(x)
    assert out.shape == (4, action_dim)

def test_deepscope_agent(obs_dim, action_dim):
    agent = DeepScopeAgent(obs_dim, action_dim)
    x = torch.randn(8, obs_dim)
    out = agent(x)
    assert out.shape == (8, action_dim)

def test_sentinel_agent(obs_dim, action_dim):
    # Sentinel usually outputs [Batch, 2]
    agent = SentinelAgent(obs_dim, 2)
    x = torch.randint(0, 100, (4, 10)) # Batch 4, Seq 10
    out = agent(x)
    assert out.shape == (4, 2)

def test_payloadgen_agent(obs_dim):
    agent = PayloadGenAgent(obs_dim, 128)
    x = torch.randn(2, obs_dim)
    out = agent(x)
    # T5 generate outputs token IDs, shape [Batch, Seq]
    assert out.dim() == 2
    assert out.shape[0] == 2

def test_mutator_agent(obs_dim):
    sentinel = SentinelAgent(obs_dim, 2)
    gen = PayloadGenAgent(obs_dim, 128)
    agent = MutatorAgent(obs_dim, 128, sentinel, gen)
    x = torch.randn(1, obs_dim)
    out = agent(x)
    # Output is optimized embeddings [1, Seq, Hidden]
    assert out.dim() == 3
    assert out.shape[0] == 1

def test_mimic_agent(obs_dim, action_dim):
    agent = MimicAgent(obs_dim, action_dim)
    x = torch.randn(4, obs_dim)
    out = agent(x)
    assert out.shape == (4, action_dim)

def test_ghost_agent(obs_dim):
    agent = GhostAgent(obs_dim, 1)
    x = torch.randn(10, obs_dim)
    out = agent(x)
    assert out.shape == (10, 1)

def test_stego_agent(obs_dim):
    agent = StegoAgent(obs_dim, 64)
    x = torch.randn(2, obs_dim)
    cover = torch.randn(2, 1, 32, 32)
    out = agent(x, context=cover)
    assert out.shape == (2, 1, 32, 32)

def test_cleaner_agent(obs_dim, action_dim):
    agent = CleanerAgent(obs_dim, action_dim)
    x = torch.randn(4, 5, obs_dim) # Batch 4, Seq 5
    out = agent(x)
    assert out.shape == (4, action_dim)
    assert hasattr(agent, "last_verified_score")
    assert agent.last_verified_score.shape == (4, 1)
