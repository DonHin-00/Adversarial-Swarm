import torch
import torch.nn as nn
import logging
from torch_geometric.data import Data
from typing import List, Dict, Optional, Tuple, Set
import ipaddress

# Configure logger
logger = logging.getLogger(__name__)

class LogEncoder(nn.Module):
    def __init__(self, node_feature_dim: int = 64, edge_feature_dim: int = 32):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # Encoders for node features
        # IP as 32 bits -> embedded to node_feature_dim
        self.ip_projection = nn.Linear(32, node_feature_dim)

        # Encoders for edge features
        self.port_embedding = nn.Embedding(65536, edge_feature_dim) # 0-65535 ports
        self.proto_embedding = nn.Embedding(256, edge_feature_dim) # 0-255 protocols

        # Mappings
        self.ip_to_idx: Dict[str, int] = {}
        self.idx_to_ip: Dict[int, str] = {}
        self.next_idx = 0

    def _ip_to_bits(self, ip_str: str) -> torch.Tensor:
        try:
            ip_int = int(ipaddress.IPv4Address(ip_str))
            # Convert to 32 bits binary representation
            # format(ip_int, '032b') creates a string like '11000000101010000000000100000001'
            bits = [float(x) for x in format(ip_int, '032b')]
            return torch.tensor(bits, dtype=torch.float32)
        except (ipaddress.AddressValueError, ValueError):
            logger.warning(f"Invalid IP address encountered: {ip_str}. Using 0.0.0.0")
            return torch.zeros(32, dtype=torch.float32)

    def _get_node_idx(self, ip: str) -> int:
        if ip not in self.ip_to_idx:
            idx = self.next_idx
            self.ip_to_idx[ip] = idx
            self.idx_to_ip[idx] = ip
            self.next_idx += 1
        return self.ip_to_idx[ip]

    def update(self, logs: List[Dict]) -> Data:
        """
        Converts a list of raw log dictionaries into a PyG Data object.

        Args:
            logs: List of dicts with keys 'src_ip', 'dst_ip', 'port', 'proto'

        Returns:
            torch_geometric.data.Data object with:
            - x: Node features [num_nodes, node_feature_dim]
            - edge_index: Graph connectivity [2, num_edges]
            - edge_attr: Edge features [num_edges, edge_feature_dim * 2]
        """
        if not logs:
            # Return empty graph with correct feature dimensions
            # Even with 0 nodes, feature dim must match expectation
            return Data(
                x=torch.zeros((0, self.node_feature_dim)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, self.edge_feature_dim * 2))
            )

        src_indices = []
        dst_indices = []
        edge_attr_inputs = [] # List of tuples (port, proto)

        # Process logs to build edges and register nodes
        for log in logs:
            src = log.get('src_ip', '0.0.0.0')
            dst = log.get('dst_ip', '0.0.0.0')
            port = log.get('port', 0)
            proto = log.get('proto', 6) # TCP default

            # Input Validation Hardening
            try:
                port = int(port)
                if not (0 <= port <= 65535):
                    port = 0
            except (ValueError, TypeError):
                port = 0

            try:
                proto = int(proto)
                if not (0 <= proto <= 255):
                    proto = 6
            except (ValueError, TypeError):
                proto = 6

            src_idx = self._get_node_idx(src)
            dst_idx = self._get_node_idx(dst)

            src_indices.append(src_idx)
            dst_indices.append(dst_idx)

            edge_attr_inputs.append((port, proto))

        # Create Node Features Tensor
        # Batch process all registered nodes for better performance
        num_nodes = self.next_idx
        
        if num_nodes == 0:
             return Data(
                x=torch.zeros((0, self.node_feature_dim)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, self.edge_feature_dim * 2))
            )
        
        # Vectorized IP to bits conversion
        x_tensor = torch.zeros((num_nodes, 32), dtype=torch.float32)
        for i in range(num_nodes):
            ip_str = self.idx_to_ip[i]
            x_tensor[i] = self._ip_to_bits(ip_str)
        
        x_embedded = self.ip_projection(x_tensor) # [num_nodes, node_feature_dim]

        # Create Edge Index Tensor
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

        # Create Edge Attributes Tensor - vectorized for better performance
        if edge_attr_inputs:
            ports_list, protos_list = zip(*edge_attr_inputs)
            ports = torch.tensor(ports_list, dtype=torch.long)
            protos = torch.tensor(protos_list, dtype=torch.long)

            port_embeds = self.port_embedding(ports) # [num_edges, edge_feature_dim]
            proto_embeds = self.proto_embedding(protos) # [num_edges, edge_feature_dim]

            edge_attr = torch.cat([port_embeds, proto_embeds], dim=1) # [num_edges, edge_feature_dim * 2]
        else:
            edge_attr = torch.empty((0, self.edge_feature_dim * 2))

        return Data(x=x_embedded, edge_index=edge_index, edge_attr=edge_attr)
