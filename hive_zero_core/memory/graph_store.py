import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import List, Dict, Optional, Tuple
import ipaddress
import hashlib

class HeteroLogEncoder(nn.Module):
    """
    Advanced Log Encoder producing Heterogeneous Graphs.
    """
    def __init__(self, node_embed_dim: int = 64):
        super().__init__()
        self.node_embed_dim = node_embed_dim

        self.ip_encoder = nn.Linear(32, node_embed_dim)
        self.port_encoder = nn.Embedding(65536, node_embed_dim)
        self.proto_encoder = nn.Embedding(256, node_embed_dim)

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
