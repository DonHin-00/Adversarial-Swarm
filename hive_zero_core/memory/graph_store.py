import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import List, Dict, Optional, Any
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

        self.tactic_encoder = nn.Embedding(14, node_embed_dim)
        self.technique_encoder = nn.Embedding(1000, node_embed_dim)
        self.diamond_encoder = nn.Embedding(4, node_embed_dim)

    def _get_idx(self, key, mapping):
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    def _ip_to_tensor(self, ip_str: str) -> torch.Tensor:
        try:
            ip_int = int(ipaddress.IPv4Address(ip_str))
            bits = [float(x) for x in format(ip_int, '032b')]
            return torch.tensor(bits, dtype=torch.float32)
        except (ValueError, ipaddress.AddressValueError):
            return torch.zeros(32, dtype=torch.float32)

    def update(self, logs: List[Dict], mitre_context: Optional[List[Dict]] = None) -> HeteroData:
        data = HeteroData()

        local_ip_map: Dict[str, int] = {}
        local_port_map: Dict[int, int] = {}
        local_tech_map: Dict[str, int] = {}

        ip_src, ip_dst = [], []
        ip_port_src, ip_port_dst = [], []
        ip_tech_src, ip_tech_dst = [], []

        for log in logs:
            src = log.get('src_ip', '0.0.0.0')
            dst = log.get('dst_ip', '0.0.0.0')
            port = int(log.get('port', 0))

            s_idx = self._get_idx(src, local_ip_map)
            d_idx = self._get_idx(dst, local_ip_map)
            p_idx = self._get_idx(port, local_port_map)

            ip_src.append(s_idx)
            ip_dst.append(d_idx)
            ip_port_src.append(d_idx)
            ip_port_dst.append(p_idx)

        if mitre_context:
            for ctx in mitre_context:
                ip = ctx.get('ip', '0.0.0.0')
                tech_id = ctx.get('technique', 'T0000')
                if ip in local_ip_map:
                    ip_idx = local_ip_map[ip]
                    t_idx = self._get_idx(tech_id, local_tech_map)
                    ip_tech_src.append(ip_idx)
                    ip_tech_dst.append(t_idx)

        device = next(self.parameters()).device

        # Features
        ip_feats = [self._ip_to_tensor(ip) for ip, _ in sorted(local_ip_map.items(), key=lambda x: x[1])]
        if ip_feats:
            data['ip'].x = self.ip_encoder(torch.stack(ip_feats).to(device))
        else:
            data['ip'].x = torch.zeros(0, self.node_embed_dim, device=device)

        port_indices = torch.tensor([p for p, _ in sorted(local_port_map.items(), key=lambda x: x[1])], dtype=torch.long, device=device)
        if len(port_indices) > 0:
            data['port'].x = self.port_encoder(port_indices)
        else:
            data['port'].x = torch.zeros(0, self.node_embed_dim, device=device)

        tech_indices = []
        for t_id, _ in sorted(local_tech_map.items(), key=lambda x: x[1]):
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

        # Store maps for Viz
        self.last_maps = {
            'ip': {v: k for k, v in local_ip_map.items()},
            'port': {v: k for k, v in local_port_map.items()},
            'technique': {v: k for k, v in local_tech_map.items()}
        }

        return data

    def to_cytoscape_json(self, data: HeteroData) -> Dict[str, Any]:
        """
        Exports current graph to Cytoscape.js format.
        """
        elements = {"nodes": [], "edges": []}

        # Nodes
        if hasattr(self, 'last_maps'):
            for idx, ip in self.last_maps.get('ip', {}).items():
                elements["nodes"].append({"data": {"id": f"ip_{idx}", "label": ip, "type": "ip"}})
            for idx, port in self.last_maps.get('port', {}).items():
                elements["nodes"].append({"data": {"id": f"port_{idx}", "label": str(port), "type": "port"}})
            for idx, tech in self.last_maps.get('technique', {}).items():
                elements["nodes"].append({"data": {"id": f"tech_{idx}", "label": tech, "type": "technique"}})

        # Edges
        # IP->IP
        if 'ip' in data.node_types and ('ip', 'flow', 'ip') in data.edge_types:
            edge_index = data['ip', 'flow', 'ip'].edge_index
            if edge_index.size(1) > 0:
                srcs, dsts = edge_index[0].tolist(), edge_index[1].tolist()
                for s, d in zip(srcs, dsts):
                    elements["edges"].append({"data": {"source": f"ip_{s}", "target": f"ip_{d}", "label": "flow"}})

        # IP->Port
        if ('ip', 'binds', 'port') in data.edge_types:
            edge_index = data['ip', 'binds', 'port'].edge_index
            if edge_index.size(1) > 0:
                srcs, dsts = edge_index[0].tolist(), edge_index[1].tolist()
                for s, d in zip(srcs, dsts):
                    elements["edges"].append({"data": {"source": f"ip_{s}", "target": f"port_{d}", "label": "binds"}})

        return elements
