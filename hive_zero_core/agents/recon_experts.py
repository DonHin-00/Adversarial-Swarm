import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from typing import Optional, Dict, Union, List
from hive_zero_core.agents.base_expert import BaseExpert
from hive_zero_core.scanners.nmap_adapter import NmapAdapter
from hive_zero_core.knowledge.mitre_kb import MitreKnowledgeBase

class Agent_Cartographer(BaseExpert):
    """
    Expert 1: Real-World Mapper (Nmap + HGT + MITRE)
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Cartographer", hidden_dim=hidden_dim)

        self.scanner = NmapAdapter()
        self.kb = MitreKnowledgeBase()

        self.metadata = (
            ['ip', 'port', 'technique'],
            [('ip', 'flow', 'ip'), ('ip', 'binds', 'port'), ('ip', 'exhibits', 'technique')]
        )

        self.conv1 = HGTConv(observation_dim, hidden_dim, self.metadata, heads=4)
        self.conv2 = HGTConv(hidden_dim, action_dim, self.metadata, heads=2)

    def scan_and_map(self, target_ip: str) -> Dict:
        logs = self.scanner.scan_target(target_ip)
        mitre_ctx = []
        for log in logs:
            service = log.get('service', 'unknown')
            techniques = self.kb.map_service_to_technique(service)
            for t in techniques:
                mitre_ctx.append({
                    'ip': log['dst_ip'],
                    'technique': t
                })
        return {'logs': logs, 'mitre_context': mitre_ctx}

    def _forward_impl(self, x: Union[torch.Tensor, HeteroData], context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device

        if isinstance(x, HeteroData):
            x_dict = x.x_dict
            edge_index_dict = x.edge_index_dict

            out_dict = self.conv1(x_dict, edge_index_dict)
            out_dict = {key: torch.relu(val) for key, val in out_dict.items()}
            out_dict = self.conv2(out_dict, edge_index_dict)

            return out_dict.get('ip', torch.zeros(0, self.action_dim, device=device))
        else:
            return torch.zeros(x.size(0), self.action_dim, device=device)

class Agent_DeepScope(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="DeepScope", hidden_dim=hidden_dim)
        self.priority_net = nn.Linear(observation_dim, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.priority_net(x)

class Agent_Chronos(BaseExpert):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, name="Chronos", hidden_dim=hidden_dim)
        self.encoder = nn.Linear(observation_dim + 10, hidden_dim)
        self.head = nn.Linear(hidden_dim, action_dim)

    def _forward_impl(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device

        fft = torch.fft.rfft(x, dim=1)
        magnitudes = torch.abs(fft)
        feats = magnitudes[:, 1:11]

        if feats.size(1) < 10:
            feats = torch.cat([feats, torch.zeros(x.size(0), 10 - feats.size(1), device=device)], dim=1)

        inp = torch.cat([x[:, -1:], feats], dim=1)
        out = self.encoder(inp)
        return self.head(torch.relu(out))
