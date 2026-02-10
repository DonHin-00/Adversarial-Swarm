import hashlib
import ipaddress
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


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
        try:
            ip_int = int(ipaddress.IPv4Address(ip_str))
            bits = [float(x) for x in format(ip_int, "032b")]
            return torch.tensor(bits, dtype=torch.float32)
        except (ValueError, ipaddress.AddressValueError, TypeError) as e:
            # Log malformed IP addresses for debugging
            logging.getLogger(__name__).warning(f"Invalid IP address '{ip_str}': {e}")
            return torch.zeros(32, dtype=torch.float32)

    def update(self, logs: List[Dict]) -> HeteroData:
        data = HeteroData()

        # Infer device from module parameters with safe fallback
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

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
            src_ip = log.get("src_ip", "0.0.0.0")
            dst_ip = log.get("dst_ip", "0.0.0.0")

            # Validate and clamp port to valid range
            try:
                dport = int(log.get("port", 0))
                if dport < 0 or dport > 65535:
                    logging.getLogger(__name__).warning(
                        f"Port {dport} out of range [0, 65535], clamping to valid range"
                    )
                dport = max(0, min(dport, 65535))  # Clamp to valid port range
            except (ValueError, TypeError) as e:
                logging.getLogger(__name__).warning(f"Invalid port value: {e}, using default 0")
                dport = 0

            # Validate and clamp protocol to valid range for IP protocol field (0-255)
            # The protocol number is an 8-bit field in the IP header, thus limited to 0-255
            try:
                proto = int(log.get("proto", 6))
                if proto < 0 or proto > 255:
                    logging.getLogger(__name__).warning(
                        f"Protocol {proto} out of range [0, 255], clamping to valid range"
                    )
                proto = max(0, min(proto, 255))  # Clamp to valid protocol range
            except (ValueError, TypeError) as e:
                logging.getLogger(__name__).warning(
                    f"Invalid protocol value: {e}, using default 6 (TCP)"
                )
                proto = 6  # Default to TCP

            s_idx = self._get_idx(src_ip, local_ip_map)
            d_idx = self._get_idx(dst_ip, local_ip_map)
            p_idx = self._get_idx(dport, local_port_map)
            pr_idx = self._get_idx(proto, local_proto_map)

            # Edges
            # IP -> IP (Communication Flow)
            ip_src_indices.append(s_idx)
            ip_dst_indices.append(d_idx)

            # IP -> Port (Destination Service)
            ip_to_port_src.append(d_idx)  # Destination IP owns the port
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
            x_ip = self.ip_encoder(torch.stack(ip_features).to(device))
        else:
            x_ip = torch.zeros(0, self.node_embed_dim, device=device)

        # Port Nodes
        sorted_ports = sorted(local_port_map.items(), key=lambda x: x[1])
        port_indices = torch.tensor([p for p, _ in sorted_ports], dtype=torch.long, device=device)
        if len(port_indices) > 0:
            x_port = self.port_encoder(port_indices)
        else:
            x_port = torch.zeros(0, self.node_embed_dim, device=device)

        # Proto Nodes
        sorted_protos = sorted(local_proto_map.items(), key=lambda x: x[1])
        proto_indices = torch.tensor([p for p, _ in sorted_protos], dtype=torch.long, device=device)
        if len(proto_indices) > 0:
            x_proto = self.proto_encoder(proto_indices)
        else:
            x_proto = torch.zeros(0, self.node_embed_dim, device=device)

        # Assign to Data
        data["ip"].x = x_ip
        data["port"].x = x_port
        data["protocol"].x = x_proto

        # Assign Edges
        # flow: IP -> IP
        if ip_src_indices:
            data["ip", "flow", "ip"].edge_index = torch.tensor(
                [ip_src_indices, ip_dst_indices], dtype=torch.long, device=device
            )
        else:
            data["ip", "flow", "ip"].edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

        # binds: IP -> Port
        if ip_to_port_src:
            data["ip", "binds", "port"].edge_index = torch.tensor(
                [ip_to_port_src, ip_to_port_dst], dtype=torch.long, device=device
            )
        else:
            data["ip", "binds", "port"].edge_index = torch.empty(
                2, 0, dtype=torch.long, device=device
            )

        # uses: Port -> Protocol
        if port_to_proto_src:
            data["port", "uses", "protocol"].edge_index = torch.tensor(
                [port_to_proto_src, port_to_proto_dst], dtype=torch.long, device=device
            )
        else:
            data["port", "uses", "protocol"].edge_index = torch.empty(
                2, 0, dtype=torch.long, device=device
            )

        return data
