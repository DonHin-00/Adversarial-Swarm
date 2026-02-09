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
        self.ip_map: Dict[str, int] = {}
        self.port_map: Dict[int, int] = {}
        self.proto_map: Dict[int, int] = {}

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
    def update(self, logs: List[Dict]) -> HeteroData:
        data = HeteroData()

        ip_src_indices: List[int] = []
        ip_dst_indices: List[int] = []

        ip_to_port_src: List[int] = []
        ip_to_port_dst: List[int] = []

        port_to_proto_src: List[int] = []
        port_to_proto_dst: List[int] = []

        local_ip_map: Dict[str, int] = {}
        local_port_map: Dict[int, int] = {}
        local_proto_map: Dict[int, int] = {}

        for log in logs:
            src_ip = log.get('src_ip', '0.0.0.0')
            dst_ip = log.get('dst_ip', '0.0.0.0')
            dport = int(log.get('port', 0))
            proto = int(log.get('proto', 6))

            s_idx = self._get_idx(src_ip, local_ip_map)
            d_idx = self._get_idx(dst_ip, local_ip_map)
            p_idx = self._get_idx(dport, local_port_map)
            pr_idx = self._get_idx(proto, local_proto_map)

            ip_src_indices.append(s_idx)
            ip_dst_indices.append(d_idx)

            ip_to_port_src.append(d_idx)
            ip_to_port_dst.append(p_idx)

            port_to_proto_src.append(p_idx)
            port_to_proto_dst.append(pr_idx)

        ip_features = []
        sorted_ips = sorted(local_ip_map.items(), key=lambda x: x[1])
        for ip, _ in sorted_ips:
            ip_features.append(self._ip_to_tensor(ip))

        # Ensure device consistency by inferring from model parameters
        device = next(self.parameters()).device

        if ip_features:
            x_ip_raw = torch.stack(ip_features).to(device)
            x_ip = self.ip_encoder(x_ip_raw)
        else:
            x_ip = torch.zeros((0, self.node_embed_dim), device=device)

        sorted_ports = sorted(local_port_map.items(), key=lambda x: x[1])
        port_indices = torch.tensor([p for p, _ in sorted_ports], dtype=torch.long, device=device)
        if len(port_indices) > 0:
            x_port = self.port_encoder(port_indices)
        else:
            x_port = torch.zeros((0, self.node_embed_dim), device=device)

        sorted_protos = sorted(local_proto_map.items(), key=lambda x: x[1])
        proto_indices = torch.tensor([p for p, _ in sorted_protos], dtype=torch.long, device=device)
        if len(proto_indices) > 0:
            x_proto = self.proto_encoder(proto_indices)
        else:
            x_proto = torch.zeros((0, self.node_embed_dim), device=device)

        data['ip'].x = x_ip
        data['port'].x = x_port
        data['protocol'].x = x_proto

        if ip_src_indices:
            data['ip', 'flow', 'ip'].edge_index = torch.tensor([ip_src_indices, ip_dst_indices], dtype=torch.long, device=device)
        else:
            data['ip', 'flow', 'ip'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        if ip_to_port_src:
            data['ip', 'binds', 'port'].edge_index = torch.tensor([ip_to_port_src, ip_to_port_dst], dtype=torch.long, device=device)
        else:
            data['ip', 'binds', 'port'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        if port_to_proto_src:
            data['port', 'uses', 'protocol'].edge_index = torch.tensor([port_to_proto_src, port_to_proto_dst], dtype=torch.long, device=device)
        else:
            data['port', 'uses', 'protocol'].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        return data
