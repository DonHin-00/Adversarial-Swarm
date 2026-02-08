import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import List, Dict, Optional, Tuple
import ipaddress
import hashlib

class HeteroLogEncoder(nn.Module):
    """
    Advanced Log Encoder producing Heterogeneous Graphs.
    Nodes:
        - Network: IP, Port, Protocol
        - MITRE: Tactic, Technique
        - Diamond: Adversary, Infrastructure, Capability, Victim
    Edges:
        - Network Flows: (IP, flow, IP), (IP, binds, Port)
        - Attribution: (IP, maps_to, Infrastructure), (Technique, uses, Capability)
    """
    def __init__(self, node_embed_dim: int = 64):
        super().__init__()
        self.node_embed_dim = node_embed_dim

        # Encoders for Network Entities
        self.ip_encoder = nn.Linear(32, node_embed_dim)
        self.port_encoder = nn.Embedding(65536, node_embed_dim)
        self.proto_encoder = nn.Embedding(256, node_embed_dim)

        # Encoders for Framework Entities (MITRE / Diamond)
        # Using simple embeddings for IDs
        self.tactic_encoder = nn.Embedding(14, node_embed_dim) # 14 MITRE Tactics
        self.technique_encoder = nn.Embedding(1000, node_embed_dim) # ~600+ Techniques
        self.diamond_encoder = nn.Embedding(4, node_embed_dim) # 4 Diamond Corners (Type Embedding)

        # Maps (Persist across updates? For prototype, reset per batch usually, but KB is static)
        # We'll handle dynamic mapping in update()

    def _get_idx(self, key, mapping):
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    def _ip_to_tensor(self, ip_str: str) -> torch.Tensor:
        try:
            ip_int = int(ipaddress.IPv4Address(ip_str))
            bits = [float(x) for x in format(ip_int, '032b')]
            return torch.tensor(bits, dtype=torch.float32)
        except:
            return torch.zeros(32, dtype=torch.float32)

    def update(self, logs: List[Dict], mitre_context: Optional[List[Dict]] = None) -> HeteroData:
        data = HeteroData()

        # Mappings
        local_ip_map: Dict[str, int] = {}
        local_port_map: Dict[int, int] = {}
        local_proto_map: Dict[int, int] = {}
        local_tech_map: Dict[str, int] = {} # Technique ID -> Index

        # Network Edges
        ip_src, ip_dst = [], []
        ip_port_src, ip_port_dst = [], []

        # Process Logs
        for log in logs:
            src = log.get('src_ip', '0.0.0.0')
            dst = log.get('dst_ip', '0.0.0.0')
            port = int(log.get('port', 0))

            s_idx = self._get_idx(src, local_ip_map)
            d_idx = self._get_idx(dst, local_ip_map)
            p_idx = self._get_idx(port, local_port_map)

            ip_src.append(s_idx)
            ip_dst.append(d_idx)

            ip_port_src.append(d_idx) # Dst IP owns Port
            ip_port_dst.append(p_idx)

        # Process MITRE Context (if provided, e.g. from Nmap Service detection)
        # Format: {'ip': '1.2.3.4', 'technique': 'T1021'}
        tech_node_indices = []
        ip_tech_src, ip_tech_dst = [], []

        if mitre_context:
            for ctx in mitre_context:
                ip = ctx.get('ip', '0.0.0.0')
                tech_id = ctx.get('technique', 'T0000')

                if ip in local_ip_map:
                    ip_idx = local_ip_map[ip]
                    t_idx = self._get_idx(tech_id, local_tech_map)

                    # Edge: IP -> Exhibit -> Technique
                    ip_tech_src.append(ip_idx)
                    ip_tech_dst.append(t_idx)

        # Build Features
        device = next(self.parameters()).device

        # IP Features
        ip_feats = [self._ip_to_tensor(ip) for ip, _ in sorted(local_ip_map.items(), key=lambda x: x[1])]
        if ip_feats:
            data['ip'].x = self.ip_encoder(torch.stack(ip_feats).to(device))
        else:
            data['ip'].x = torch.zeros(0, self.node_embed_dim, device=device)

        # Port Features
        port_indices = torch.tensor([p for p, _ in sorted(local_port_map.items(), key=lambda x: x[1])], dtype=torch.long, device=device)
        if len(port_indices) > 0:
            data['port'].x = self.port_encoder(port_indices)
        else:
            data['port'].x = torch.zeros(0, self.node_embed_dim, device=device)

        # Technique Features (Mock hash to index 0-999)
        tech_indices = []
        for t_id, _ in sorted(local_tech_map.items(), key=lambda x: x[1]):
            # Simple hash mapping for prototype
            idx = int(hashlib.md5(t_id.encode()).hexdigest(), 16) % 1000
            tech_indices.append(idx)

        if tech_indices:
            data['technique'].x = self.technique_encoder(torch.tensor(tech_indices, dtype=torch.long, device=device))
        else:
            data['technique'].x = torch.zeros(0, self.node_embed_dim, device=device)

        # Edges
        if ip_src:
            data['ip', 'flow', 'ip'].edge_index = torch.tensor([ip_src, ip_dst], dtype=torch.long, device=device)
        else:
            data['ip', 'flow', 'ip'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        if ip_port_src:
            data['ip', 'binds', 'port'].edge_index = torch.tensor([ip_port_src, ip_port_dst], dtype=torch.long, device=device)
        else:
            data['ip', 'binds', 'port'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        if ip_tech_src:
            data['ip', 'exhibits', 'technique'].edge_index = torch.tensor([ip_tech_src, ip_tech_dst], dtype=torch.long, device=device)
        else:
            data['ip', 'exhibits', 'technique'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        return data
