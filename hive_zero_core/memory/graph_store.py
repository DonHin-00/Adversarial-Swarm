import torch
import torch.nn as nn
import logging
from torch_geometric.data import Data
from typing import List, Dict, Optional, Tuple, Set
import ipaddress
import hashlib

# Configure logger
logger = logging.getLogger(__name__)

class LogEncoder(nn.Module):
    def __init__(self, node_feature_dim: int = 64, edge_feature_dim: int = 32):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # Encoders for node features
        self.ip_projection = nn.Linear(32, node_feature_dim)

        # Encoders for edge features
        self.port_embedding = nn.Embedding(65536, edge_feature_dim)
        self.proto_embedding = nn.Embedding(256, edge_feature_dim)

        # Stateful Flow Tracking (TCP Stream Reassembly Simulation)
        self.flow_state_dim = 16
        # Input to GRU is [edge_feat_dim * 2] (port + proto)
        self.flow_tracker = nn.GRUCell(edge_feature_dim * 2, self.flow_state_dim)

        # Mappings
        self.ip_to_idx: Dict[str, int] = {}
        self.idx_to_ip: Dict[int, str] = {}
        self.next_idx = 0

        # Flow State Cache: hash(5-tuple) -> tensor state
        self.flow_states: Dict[str, torch.Tensor] = {}

    def _ip_to_bits(self, ip_str: str) -> torch.Tensor:
        try:
            ip_int = int(ipaddress.IPv4Address(ip_str))
            bits = [float(x) for x in format(ip_int, '032b')]
            return torch.tensor(bits, dtype=torch.float32)
        except (ipaddress.AddressValueError, ValueError):
            return torch.zeros(32, dtype=torch.float32)

    def _get_node_idx(self, ip: str) -> int:
        if ip not in self.ip_to_idx:
            idx = self.next_idx
            self.ip_to_idx[ip] = idx
            self.idx_to_ip[idx] = ip
            self.next_idx += 1
        return self.ip_to_idx[ip]

    def _get_flow_key(self, src, dst, sport, dport, proto) -> str:
        return hashlib.md5(f"{src}:{sport}-{dst}:{dport}/{proto}".encode()).hexdigest()

    def update(self, logs: List[Dict]) -> Data:
        """
        Converts logs to Graph Data. Updates flow states.
        """
        if not logs:
            return Data(
                x=torch.zeros((0, self.node_feature_dim)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, self.edge_feature_dim * 2 + self.flow_state_dim))
            )

        src_indices = []
        dst_indices = []
        edge_attr_inputs = []
        flow_keys_list = []

        # Register nodes and prepare edges
        for log in logs:
            src = log.get('src_ip', '0.0.0.0')
            dst = log.get('dst_ip', '0.0.0.0')
            sport = int(log.get('src_port', 0))
            dport = int(log.get('port', 0))
            proto = int(log.get('proto', 6))

            # Validation
            sport = max(0, min(65535, sport))
            dport = max(0, min(65535, dport))
            proto = max(0, min(255, proto))

            src_idx = self._get_node_idx(src)
            dst_idx = self._get_node_idx(dst)

            src_indices.append(src_idx)
            dst_indices.append(dst_idx)

            edge_attr_inputs.append((dport, proto))

            flow_key = self._get_flow_key(src, dst, sport, dport, proto)
            flow_keys_list.append(flow_key)

        # Create Node Features
        num_nodes = self.next_idx
        x_raw_list = []
        for i in range(num_nodes):
            x_raw_list.append(self._ip_to_bits(self.idx_to_ip[i]))

        if not x_raw_list:
             # Should be covered by empty check, but safe fallback
             x_tensor = torch.zeros(0, 32)
        else:
             x_tensor = torch.stack(x_raw_list)

        # Device consistency - assume cpu for encoding logic then model moves
        # But if model is on GPU, these layers are on GPU.
        # We need to respect self.device
        device = next(self.parameters()).device
        x_tensor = x_tensor.to(device)

        x_embedded = self.ip_projection(x_tensor)

        # Edge Index
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long, device=device)

        # Edge Attributes
        ports = torch.tensor([attr[0] for attr in edge_attr_inputs], dtype=torch.long, device=device)
        protos = torch.tensor([attr[1] for attr in edge_attr_inputs], dtype=torch.long, device=device)

        port_embeds = self.port_embedding(ports)
        proto_embeds = self.proto_embedding(protos)

        basic_edge_attr = torch.cat([port_embeds, proto_embeds], dim=1)

        # Update Flow States in Batch
        # Prepare previous states tensor
        prev_states_list = []
        for key in flow_keys_list:
            if key not in self.flow_states:
                self.flow_states[key] = torch.zeros(self.flow_state_dim, device=device)
            # Ensure cached state is on correct device (in case of movement)
            if self.flow_states[key].device != device:
                self.flow_states[key] = self.flow_states[key].to(device)

            prev_states_list.append(self.flow_states[key])

        if prev_states_list:
            prev_stack = torch.stack(prev_states_list)

            # Run GRU Cell
            new_states = self.flow_tracker(basic_edge_attr, prev_stack)

            # Update cache
            for i, key in enumerate(flow_keys_list):
                self.flow_states[key] = new_states[i].detach()

            final_edge_attr = torch.cat([basic_edge_attr, new_states], dim=1)
        else:
            final_edge_attr = torch.zeros((0, self.edge_feature_dim * 2 + self.flow_state_dim), device=device)

        return Data(x=x_embedded, edge_index=edge_index, edge_attr=final_edge_attr)
