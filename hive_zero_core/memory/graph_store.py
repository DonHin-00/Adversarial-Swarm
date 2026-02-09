import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import List, Dict, Union
import ipaddress
import hashlib

class HeteroLogEncoder(nn.Module):
    """
    Advanced Log Encoder producing Heterogeneous Graphs.
    Nodes: IP, Port, Protocol
    Edges: (IP, CONNECTS_TO, Port), (Port, USES, Protocol), (IP, COMMUNICATES_WITH, IP)
    """
    def __init__(self, node_embed_dim: int = 64):
        super().__init__()
        self.node_embed_dim = node_embed_dim

        # Encoders for different node types
        self.ip_encoder = nn.Linear(32, node_embed_dim)
        self.port_encoder = nn.Embedding(65536, node_embed_dim)
        self.proto_encoder = nn.Embedding(256, node_embed_dim)

        # Mappings
        self.ip_map = {}
        self.port_map = {}
        self.proto_map = {}

    def _get_idx(self, key: Union[str, int], mapping: Dict) -> int:
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    def _ip_to_tensor(self, ip_str: str) -> torch.Tensor:
        """Convert IP address string to binary tensor representation.
        
        Args:
            ip_str: IPv4 address string (e.g., '192.168.1.1')
            
        Returns:
            32-dimensional binary tensor representation, or zeros if invalid
        """
        try:
            ip_int = int(ipaddress.IPv4Address(ip_str))
            bits = [float(x) for x in format(ip_int, '032b')]
            return torch.tensor(bits, dtype=torch.float32)
        except (ValueError, ipaddress.AddressValueError):
            # Return zero tensor for invalid IP addresses
            return torch.zeros(32, dtype=torch.float32)

    def update(self, logs: List[Dict]) -> HeteroData:
        """Update the heterogeneous graph with new log entries.
        
        Args:
            logs: List of log dictionaries with keys: 'src_ip', 'dst_ip', 'port', 'proto'
            
        Returns:
            HeteroData object representing the network graph
        """
        if not logs:
            # Return empty graph for empty input
            data = HeteroData()
            data['ip'].x = torch.zeros(0, self.node_embed_dim)
            data['port'].x = torch.zeros(0, self.node_embed_dim)
            data['protocol'].x = torch.zeros(0, self.node_embed_dim)
            data['ip', 'flow', 'ip'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['ip', 'binds', 'port'].edge_index = torch.empty(2, 0, dtype=torch.long)
            data['port', 'uses', 'protocol'].edge_index = torch.empty(2, 0, dtype=torch.long)
            return data
            
        data = HeteroData()

        # Lists for edges
        ip_src_indices = []
        ip_dst_indices = []

        ip_to_port_src = []
        ip_to_port_dst = []

        port_to_proto_src = []
        port_to_proto_dst = []

        # Reset mappings for batch (or persistent? For prototype, batch-local)
        self.ip_map = {}
        # Keep port/proto maps implies they are global concepts, but for graph indices
        # we need 0..N for this batch's graph.
        # Actually, PyG HeteroData expects features for nodes present in graph.
        # So we rebuild local maps.

        local_ip_map = {}
        local_port_map = {}
        local_proto_map = {}

        for log in logs:
            # Validate and extract log fields with defaults
            src_ip = log.get('src_ip', '0.0.0.0')
            dst_ip = log.get('dst_ip', '0.0.0.0')
            
            # Validate port is within valid range (0-65535)
            try:
                port_val = int(log.get('port', 0))
                dport = max(0, min(65535, port_val))  # Clamp to valid range
            except (ValueError, TypeError):
                dport = 0
            
            # Validate protocol is within valid range (0-255)
            try:
                proto_val = int(log.get('proto', 6))  # Default to TCP (6)
                proto = max(0, min(255, proto_val))  # Clamp to valid range
            except (ValueError, TypeError):
                proto = 6

            s_idx = self._get_idx(src_ip, local_ip_map)
            d_idx = self._get_idx(dst_ip, local_ip_map)
            p_idx = self._get_idx(dport, local_port_map)
            pr_idx = self._get_idx(proto, local_proto_map)

            # Edges
            # IP -> IP (Communication Flow)
            ip_src_indices.append(s_idx)
            ip_dst_indices.append(d_idx)

            # IP -> Port (Destination Service)
            ip_to_port_src.append(d_idx) # Destination IP owns the port
            ip_to_port_dst.append(p_idx)

            # Port -> Protocol
            port_to_proto_src.append(p_idx)
            port_to_proto_dst.append(pr_idx)

        # Build Node Features
        # IP Nodes
        ip_features = []
        # Sort by index to match
        sorted_ips = sorted(local_ip_map.items(), key=lambda x: x[1])
        for ip, _ in sorted_ips:
            ip_features.append(self._ip_to_tensor(ip))

        if ip_features:
            x_ip = self.ip_encoder(torch.stack(ip_features))
        else:
            x_ip = torch.zeros(0, self.node_embed_dim)

        # Port Nodes
        sorted_ports = sorted(local_port_map.items(), key=lambda x: x[1])
        port_indices = torch.tensor([p for p, _ in sorted_ports], dtype=torch.long)
        if len(port_indices) > 0:
            x_port = self.port_encoder(port_indices)
        else:
            x_port = torch.zeros(0, self.node_embed_dim)

        # Proto Nodes
        sorted_protos = sorted(local_proto_map.items(), key=lambda x: x[1])
        proto_indices = torch.tensor([p for p, _ in sorted_protos], dtype=torch.long)
        if len(proto_indices) > 0:
            x_proto = self.proto_encoder(proto_indices)
        else:
            x_proto = torch.zeros(0, self.node_embed_dim)

        # Assign to Data
        data['ip'].x = x_ip
        data['port'].x = x_port
        data['protocol'].x = x_proto

        # Assign Edges
        # flow: IP -> IP
        if ip_src_indices:
            data['ip', 'flow', 'ip'].edge_index = torch.tensor([ip_src_indices, ip_dst_indices], dtype=torch.long)
        else:
            data['ip', 'flow', 'ip'].edge_index = torch.empty(2, 0, dtype=torch.long)

        # binds: IP -> Port
        if ip_to_port_src:
            data['ip', 'binds', 'port'].edge_index = torch.tensor([ip_to_port_src, ip_to_port_dst], dtype=torch.long)
        else:
            data['ip', 'binds', 'port'].edge_index = torch.empty(2, 0, dtype=torch.long)

        # uses: Port -> Protocol
        if port_to_proto_src:
            data['port', 'uses', 'protocol'].edge_index = torch.tensor([port_to_proto_src, port_to_proto_dst], dtype=torch.long)
        else:
            data['port', 'uses', 'protocol'].edge_index = torch.empty(2, 0, dtype=torch.long)

        return data
