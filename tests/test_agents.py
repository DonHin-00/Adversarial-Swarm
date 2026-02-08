import torch
import pytest
from torch_geometric.data import HeteroData
from hive_zero_core.agents.recon_experts import Agent_Cartographer, Agent_Chronos, Agent_DeepScope
from hive_zero_core.agents.attack_experts import Agent_Sentinel, Agent_PayloadGen, Agent_Mutator
from hive_zero_core.agents.post_experts import Agent_Mimic, Agent_Ghost, Agent_Stego, Agent_Cleaner

@pytest.fixture
def obs_dim():
    return 64

@pytest.fixture
def action_dim():
    return 64

def test_agent_cartographer(obs_dim, action_dim):
    agent = Agent_Cartographer(obs_dim, action_dim)
    data = HeteroData()
    data['ip'].x = torch.randn(5, obs_dim)
    data['port'].x = torch.randn(3, obs_dim)
    data['protocol'].x = torch.randn(2, obs_dim)
    data['ip', 'flow', 'ip'].edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    data['ip', 'binds', 'port'].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data['port', 'uses', 'protocol'].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    out = agent(data)
    assert out.shape == (5, action_dim)

def test_agent_chronos(obs_dim, action_dim):
    agent = Agent_Chronos(obs_dim, action_dim)
    x = torch.randn(4, 20) # Batch of 4, sequence of 20
    out = agent(x)
    assert out.shape == (4, action_dim)

def test_agent_deepscope(obs_dim, action_dim):
    agent = Agent_DeepScope(obs_dim, action_dim)
    x = torch.randn(8, obs_dim)
    out = agent(x)
    assert out.shape == (8, action_dim)

def test_agent_sentinel(obs_dim, action_dim):
    # Sentinel usually outputs [Batch, 2]
    agent = Agent_Sentinel(obs_dim, 2)
    x = torch.randint(0, 100, (4, 10)) # Batch 4, Seq 10
    out = agent(x)
    assert out.shape == (4, 2)

def test_agent_payloadgen(obs_dim):
    agent = Agent_PayloadGen(obs_dim, 128)
    x = torch.randn(2, obs_dim)
    out = agent(x)
    # T5 generate outputs token IDs, shape [Batch, Seq]
    assert out.dim() == 2
    assert out.shape[0] == 2

def test_agent_mutator(obs_dim):
    sentinel = Agent_Sentinel(obs_dim, 2)
    gen = Agent_PayloadGen(obs_dim, 128)
    agent = Agent_Mutator(obs_dim, 128, sentinel, gen)
    x = torch.randn(1, obs_dim)
    out = agent(x)
    # Output is optimized embeddings [1, Seq, Hidden]
    assert out.dim() == 3
    assert out.shape[0] == 1

def test_agent_mimic(obs_dim, action_dim):
    agent = Agent_Mimic(obs_dim, action_dim)
    x = torch.randn(4, obs_dim)
    out = agent(x)
    assert out.shape == (4, action_dim)

def test_agent_ghost(obs_dim):
    agent = Agent_Ghost(obs_dim, 1)
    x = torch.randn(10, obs_dim)
    out = agent(x)
    assert out.shape == (10, 1)

def test_agent_stego(obs_dim):
    agent = Agent_Stego(obs_dim, 64)
    x = torch.randn(2, obs_dim)
    cover = torch.randn(2, 1, 32, 32)
    out = agent(x, context=cover)
    assert out.shape == (2, 1, 32, 32)

def test_agent_cleaner(obs_dim, action_dim):
    agent = Agent_Cleaner(obs_dim, action_dim)
    x = torch.randn(4, 5, obs_dim) # Batch 4, Seq 5
    out = agent(x)
    assert out.shape == (4, action_dim)
    assert hasattr(agent, "last_verified_score")
    assert agent.last_verified_score.shape == (4, 1)
