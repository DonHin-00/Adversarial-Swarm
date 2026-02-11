"""
Data loading utilities for training the HiveMind system.
Supports both real network logs and synthetic data generation.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class NetworkLogDataset:
    """
    Dataset for network logs that can be used with the HiveMind system.
    Supports both real log files and synthetic data generation.
    """

    def __init__(
        self,
        data_source: Optional[Union[str, Path, List[Dict]]] = None,
        batch_size: int = 32,
        synthetic: bool = False,
        num_synthetic_samples: int = 1000,
    ):
        """
        Initialize the dataset.

        Args:
            data_source: Path to log file or list of log dictionaries. If None, uses synthetic data.
            batch_size: Number of samples per batch
            synthetic: If True, generate synthetic data instead of loading from file
            num_synthetic_samples: Number of synthetic samples to generate
        """
        self.batch_size = batch_size
        self.synthetic = synthetic
        self.num_synthetic_samples = num_synthetic_samples

        if synthetic or data_source is None:
            logger.info(f"Generating {num_synthetic_samples} synthetic network logs")
            self.data = self._generate_synthetic_logs(num_synthetic_samples)
        elif isinstance(data_source, list):
            logger.info(f"Using provided list of {len(data_source)} logs")
            self.data = data_source
        else:
            logger.info(f"Loading logs from {data_source}")
            self.data = self._load_from_file(data_source)

        logger.info(f"Dataset initialized with {len(self.data)} samples")

    def _generate_synthetic_logs(self, num_samples: int) -> List[Dict]:
        """Generate synthetic network logs for training."""
        logs = []

        # Common network patterns
        src_ips = [
            "192.168.1.1", "192.168.1.2", "192.168.1.100",
            "10.0.0.1", "10.0.0.5", "172.16.0.1"
        ]
        dst_ips = [
            "10.0.0.5", "8.8.8.8", "1.1.1.1",
            "172.217.0.0", "142.250.0.0"
        ]
        ports = [80, 443, 53, 22, 3389, 8080, 8443]
        protocols = [6, 17]  # TCP, UDP
        events = ["connection", "request", "response", "error", "timeout"]

        for i in range(num_samples):
            log = {
                "timestamp": f"2024-01-01T{i % 24:02d}:{i % 60:02d}:{i % 60:02d}",
                "event": events[i % len(events)],
                "src_ip": src_ips[i % len(src_ips)],
                "dst_ip": dst_ips[i % len(dst_ips)],
                "port": ports[i % len(ports)],
                "proto": protocols[i % len(protocols)],
                "bytes": torch.randint(100, 10000, (1,)).item(),
                "flags": torch.randint(0, 256, (1,)).item(),
            }
            logs.append(log)

        return logs

    def _load_from_file(self, filepath: Union[str, Path]) -> List[Dict]:
        """
        Load network logs from a file.
        Supports common log formats (placeholder for future implementation).
        """
        # Placeholder for real file loading
        # In production, this would parse actual log files (CSV, JSON, PCAP, etc.)
        logger.warning(
            "Real file loading not yet implemented. "
            "Falling back to synthetic data generation."
        )
        return self._generate_synthetic_logs(self.num_synthetic_samples)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __iter__(self) -> Iterator[List[Dict]]:
        """Iterate over batches of log data."""
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : i + self.batch_size]
            yield batch

    def get_batch(self, idx: int) -> List[Dict]:
        """Get a specific batch by index."""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))
        return self.data[start_idx:end_idx]

    @property
    def num_batches(self) -> int:
        """Return the number of batches in the dataset."""
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class DataConfig:
    """Configuration for data loading."""

    def __init__(
        self,
        data_source: Optional[Union[str, Path, List[Dict]]] = None,
        batch_size: int = 32,
        synthetic: bool = True,
        num_synthetic_samples: int = 1000,
        shuffle: bool = True,
    ):
        """
        Initialize data configuration.

        Args:
            data_source: Path to log file or list of log dictionaries
            batch_size: Number of samples per batch
            synthetic: If True, generate synthetic data
            num_synthetic_samples: Number of synthetic samples to generate
            shuffle: If True, shuffle the data
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.synthetic = synthetic
        self.num_synthetic_samples = num_synthetic_samples
        self.shuffle = shuffle

    def create_dataset(self) -> NetworkLogDataset:
        """Create a dataset based on this configuration."""
        return NetworkLogDataset(
            data_source=self.data_source,
            batch_size=self.batch_size,
            synthetic=self.synthetic,
            num_synthetic_samples=self.num_synthetic_samples,
        )
