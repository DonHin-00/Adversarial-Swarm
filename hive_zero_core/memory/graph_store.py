import ipaddress
import logging
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

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
        self.port_embedding = nn.Embedding(65536, edge_feature_dim)  # 0-65535 ports
        self.proto_embedding = nn.Embedding(256, edge_feature_dim)  # 0-255 protocols

        # Mappings
        self.ip_to_idx: Dict[str, int] = {}
        self.idx_to_ip: Dict[int, str] = {}
        self.next_idx = 0

        # Temporal tracking for Chronos agent
        self.timestamps: List[float] = []
        self.last_timestamp: Optional[float] = None

    def _ip_to_bits(self, ip_str: str) -> torch.Tensor:
        try:
            ip_int = int(ipaddress.IPv4Address(ip_str))
            # Convert to 32 bits binary representation
            # format(ip_int, '032b') creates a string like '11000000101010000000000100000001'
            bits = [float(x) for x in format(ip_int, "032b")]
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

    def _parse_timestamp(self, timestamp) -> Optional[float]:
        """
        Parse timestamp-like input to Unix epoch time.
        Supports ISO 8601 strings, common string variations, numeric epoch
        seconds, and datetime objects.
        """
        # Handle explicit None early
        if timestamp is None:
            return None

        # If it's already a datetime instance, convert directly
        if isinstance(timestamp, datetime):
            return timestamp.timestamp()

        # If it's numeric, treat it as epoch seconds
        if isinstance(timestamp, (int, float)):
            return float(timestamp)

        # Fallback: work with a string representation
        try:
            timestamp_str = str(timestamp)
        except Exception:
            logger.debug(f"Could not convert timestamp to string: {timestamp!r}")
            return None

        if not timestamp_str:
            return None

        try:
            # Try ISO 8601 format
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError, AttributeError):
            # Try other common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt.timestamp()
                except (ValueError, TypeError):
                    continue
            logger.debug(f"Could not parse timestamp: {timestamp_str!r}")
            return None

    def reset(self):
        """Clear the IP-to-index mapping to free memory between episodes."""
        self.ip_to_idx.clear()
        self.idx_to_ip.clear()
        self.next_idx = 0
        self.timestamps.clear()
        self.last_timestamp = None

    def _empty_graph(self) -> Data:
        """Return a zero-node graph on the module's device."""
        device = self.ip_projection.weight.device
        return Data(
            x=torch.zeros((0, self.node_feature_dim), device=device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
            edge_attr=torch.empty((0, self.edge_feature_dim * 2), device=device),
        )

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
        # Get device from module parameters for device-aware tensor creation
        device = self.ip_projection.weight.device

        if not logs:
            # Return empty graph with correct feature dimensions on correct device
            return Data(
                x=torch.zeros((0, self.node_feature_dim), device=device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                edge_attr=torch.empty((0, self.edge_feature_dim * 2), device=device),
            )
            return self._empty_graph()

        src_indices = []
        dst_indices = []
        edge_attr_inputs = []  # List of tuples (port, proto)

        # Process logs to build edges and register nodes
        for log in logs:
            src = log.get("src_ip", "0.0.0.0")
            dst = log.get("dst_ip", "0.0.0.0")
            port = log.get("port", 0)
            proto = log.get("proto", 6)  # TCP default

            # Extract and track timestamp if present
            if "timestamp" in log:
                ts = self._parse_timestamp(log["timestamp"])
                if ts is not None:
                    self.timestamps.append(ts)
                    self.last_timestamp = ts

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
        # Vectorized batch processing for better performance
        num_nodes = self.next_idx

        # Get device from module parameters
        device = self.ip_projection.weight.device

        if num_nodes == 0:
            return Data(
                x=torch.zeros((0, self.node_feature_dim), device=device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                edge_attr=torch.empty((0, self.edge_feature_dim * 2), device=device),
            )

        # Vectorized IP to bits conversion using list comprehension and stack
        # This is more efficient than loop-based tensor assignment
        x_raw_list = [self._ip_to_bits(self.idx_to_ip[i]) for i in range(num_nodes)]
        x_tensor = torch.stack(x_raw_list).to(device)  # [num_nodes, 32]

        x_embedded = self.ip_projection(x_tensor)  # [num_nodes, node_feature_dim]
        x_raw_list = []
        for i in range(num_nodes):
            ip_str = self.idx_to_ip[i]
            x_raw_list.append(self._ip_to_bits(ip_str))

        if not x_raw_list:
            return self._empty_graph()

        x_tensor = torch.stack(x_raw_list)  # [num_nodes, 32]
        x_embedded = self.ip_projection(x_tensor)  # [num_nodes, node_feature_dim]

        # Create Edge Index Tensor
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long, device=device)

        # Create Edge Attributes Tensor - vectorized for better performance
        if edge_attr_inputs:
            ports_list, protos_list = zip(*edge_attr_inputs)
            ports = torch.tensor(ports_list, dtype=torch.long, device=device)
            protos = torch.tensor(protos_list, dtype=torch.long, device=device)

            port_embeds = self.port_embedding(ports)  # [num_edges, edge_feature_dim]
            proto_embeds = self.proto_embedding(protos)  # [num_edges, edge_feature_dim]

            edge_attr = torch.cat(
                [port_embeds, proto_embeds], dim=1
            )  # [num_edges, edge_feature_dim * 2]
        else:
            edge_attr = torch.empty((0, self.edge_feature_dim * 2), device=device)
        # Create Edge Attributes Tensor on the same device as the embedding layers
        # to avoid device-mismatch errors when LogEncoder is moved to GPU.
        emb_device = self.port_embedding.weight.device
        ports = torch.clamp(
            torch.tensor(
                [attr[0] for attr in edge_attr_inputs], dtype=torch.long, device=emb_device
            ),
            0,
            65535,
        )
        protos = torch.clamp(
            torch.tensor(
                [attr[1] for attr in edge_attr_inputs], dtype=torch.long, device=emb_device
            ),
            0,
            255,
        )

        port_embeds = self.port_embedding(ports)  # [num_edges, edge_feature_dim]
        proto_embeds = self.proto_embedding(protos)  # [num_edges, edge_feature_dim]

        edge_attr = torch.cat(
            [port_embeds, proto_embeds], dim=1
        )  # [num_edges, edge_feature_dim * 2]

        return Data(x=x_embedded, edge_index=edge_index, edge_attr=edge_attr)

    def get_inter_arrival_times(self, max_len: int = 100) -> torch.Tensor:
        """
        Extract inter-arrival times from tracked timestamps for temporal analysis.

        Computes time differences between consecutive logs in the order they were
        received, preserving the actual event sequence.

        Args:
            max_len: Maximum number of intervals to return (must be >= 2 to produce any intervals)

        Returns:
            Tensor of inter-arrival times [1, seq_len] suitable for Chronos agent.
            The sequence length will be min(len(timestamps)-1, max_len), representing
            the actual number of intervals available without artificial padding.
            Returns zeros with shape [1, max(max_len, 2)] if insufficient timestamps available.

        Note:
            The Chronos agent handles variable-length sequences via its transformer
            architecture, so no padding is needed. This preserves the true temporal
            characteristics of the data.
        """
        device = self.ip_projection.weight.device

        # Ensure max_len is at least 2 to produce meaningful intervals
        max_len = max(2, max_len)

        if len(self.timestamps) < 2:
            # Not enough data, return zeros with consistent shape
            return torch.zeros(1, max_len, device=device)

        # Use most recent timestamps in the order they were added
        # Note: We need max_len+1 timestamps to produce max_len intervals
        recent_times = self.timestamps[-(max_len + 1) :]

        # Compute inter-arrival times (differences between consecutive timestamps)
        inter_arrivals = [
            recent_times[i + 1] - recent_times[i] for i in range(len(recent_times) - 1)
        ]

        # Convert to tensor
        if inter_arrivals:
            inter_arrival_tensor = torch.tensor(inter_arrivals, dtype=torch.float32, device=device)
            return inter_arrival_tensor.unsqueeze(0)  # [1, seq_len]
        else:
            # Fallback with consistent shape
            return torch.zeros(1, max_len, device=device)
