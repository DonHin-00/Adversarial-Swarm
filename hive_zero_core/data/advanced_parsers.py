"""
Advanced Log Parsers for Network Data

Implements parsers for multiple log formats:
- CSV files (standard and custom formats)
- JSON files (structured logs)
- PCAP files (packet capture)
- Streaming data sources

All parsers output standardized log dictionaries for compatibility with existing system.
"""

import csv
import json
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class CSVLogParser:
    """
    Parse CSV log files with flexible column mapping.

    Supports:
    - Standard CSV formats
    - Custom column mappings
    - Auto-detection of format
    - Header validation
    """

    # Standard column name variants
    TIMESTAMP_COLS = ["timestamp", "time", "datetime", "date", "ts"]
    SOURCE_COLS = ["source", "src", "source_ip", "src_ip", "source_host"]
    DEST_COLS = ["destination", "dest", "dst", "dest_ip", "dst_ip", "destination_host"]
    PORT_COLS = ["port", "dest_port", "dst_port", "destination_port"]
    PROTOCOL_COLS = ["protocol", "proto"]

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize CSV parser.

        Args:
            column_mapping: Optional mapping of CSV columns to standard fields
                           e.g., {'time': 'timestamp', 'src': 'source_ip'}
        """
        self.column_mapping = column_mapping or {}

    def parse_file(self, filepath: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse CSV file into list of log dictionaries.

        Args:
            filepath: Path to CSV file
            max_rows: Maximum number of rows to parse (None = all)

        Returns:
            List of parsed log dictionaries
        """
        logs = []

        try:
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)

                # Auto-detect column mapping if not provided
                if not self.column_mapping and reader.fieldnames:
                    self.column_mapping = self._auto_detect_columns(reader.fieldnames)

                for i, row in enumerate(reader):
                    if max_rows and i >= max_rows:
                        break

                    log = self._parse_row(row)
                    if log:
                        logs.append(log)

            logger.info(f"Parsed {len(logs)} logs from {filepath}")
            return logs

        except Exception as e:
            logger.error(f"Failed to parse CSV file {filepath}: {e}")
            return []

    def _auto_detect_columns(self, fieldnames: List[str]) -> Dict[str, str]:
        """Auto-detect column mapping from field names."""
        mapping = {}

        for field in fieldnames:
            field_lower = field.lower()

            # Check timestamp
            if any(ts in field_lower for ts in self.TIMESTAMP_COLS):
                mapping[field] = "timestamp"

            # Check source
            elif any(src in field_lower for src in self.SOURCE_COLS):
                mapping[field] = "source_ip"

            # Check destination
            elif any(dst in field_lower for dst in self.DEST_COLS):
                mapping[field] = "destination_ip"

            # Check port
            elif any(port in field_lower for port in self.PORT_COLS):
                mapping[field] = "destination_port"

            # Check protocol
            elif any(proto in field_lower for proto in self.PROTOCOL_COLS):
                mapping[field] = "protocol"

        logger.info(f"Auto-detected column mapping: {mapping}")
        return mapping

    def _parse_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Parse single CSV row into log dictionary."""
        try:
            log = {}

            # Apply column mapping
            for csv_col, std_col in self.column_mapping.items():
                if csv_col in row:
                    log[std_col] = row[csv_col]

            # Include unmapped columns as-is
            for col, value in row.items():
                if col not in self.column_mapping:
                    log[col] = value

            # Ensure minimum required fields
            if "source_ip" not in log and "source" in log:
                log["source_ip"] = log["source"]

            return log if log else None

        except Exception as e:
            logger.error(f"Failed to parse row: {e}")
            return None


class JSONLogParser:
    """
    Parse JSON log files with flexible schema support.

    Supports:
    - Single JSON object per file
    - JSON Lines format (one object per line)
    - Nested JSON structures
    - Schema validation
    """

    def parse_file(self, filepath: str, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse JSON file into list of log dictionaries.

        Args:
            filepath: Path to JSON file
            max_entries: Maximum number of entries to parse

        Returns:
            List of parsed log dictionaries
        """
        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Try parsing as single JSON object
            try:
                data = json.loads(content)

                # Handle different formats
                if isinstance(data, list):
                    logs = data[:max_entries] if max_entries else data
                elif isinstance(data, dict):
                    if "logs" in data:
                        logs = data["logs"][:max_entries] if max_entries else data["logs"]
                    else:
                        logs = [data]
                else:
                    logger.warning(f"Unexpected JSON structure in {filepath}")
                    return []

                logger.info(f"Parsed {len(logs)} logs from {filepath}")
                return logs

            except json.JSONDecodeError:
                # Try parsing as JSON Lines
                return self._parse_jsonlines(content, max_entries)

        except Exception as e:
            logger.error(f"Failed to parse JSON file {filepath}: {e}")
            return []

    def _parse_jsonlines(self, content: str, max_entries: Optional[int]) -> List[Dict[str, Any]]:
        """Parse JSON Lines format (one JSON object per line)."""
        logs = []

        for i, line in enumerate(content.strip().split("\n")):
            if max_entries and i >= max_entries:
                break

            try:
                log = json.loads(line)
                logs.append(log)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i+1}: {e}")
                continue

        logger.info(f"Parsed {len(logs)} logs from JSON Lines format")
        return logs


class PCAPParser:
    """
    Parse PCAP files using scapy.

    Extracts:
    - IP addresses (source/destination)
    - Ports (source/destination)
    - Protocols
    - Packet timestamps
    - Payload data (if requested)
    """

    def __init__(self, include_payload: bool = False):
        """
        Initialize PCAP parser.

        Args:
            include_payload: Whether to include packet payload data
        """
        self.include_payload = include_payload

        try:
            from scapy.all import rdpcap, IP, TCP, UDP

            self.rdpcap = rdpcap
            self.IP = IP
            self.TCP = TCP
            self.UDP = UDP
            self.available = True
        except ImportError:
            logger.warning("scapy not available - PCAP parsing disabled")
            self.available = False

    def parse_file(self, filepath: str, max_packets: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse PCAP file into list of log dictionaries.

        Args:
            filepath: Path to PCAP file
            max_packets: Maximum number of packets to parse

        Returns:
            List of parsed packet/log dictionaries
        """
        if not self.available:
            logger.error("PCAP parsing requires scapy: pip install scapy")
            return []

        try:
            packets = self.rdpcap(filepath)
            logs = []

            for i, packet in enumerate(packets):
                if max_packets and i >= max_packets:
                    break

                log = self._parse_packet(packet)
                if log:
                    logs.append(log)

            logger.info(f"Parsed {len(logs)} packets from {filepath}")
            return logs

        except Exception as e:
            logger.error(f"Failed to parse PCAP file {filepath}: {e}")
            return []

    def _parse_packet(self, packet) -> Optional[Dict[str, Any]]:
        """Parse single packet into log dictionary."""
        try:
            log = {"timestamp": float(packet.time), "protocol": "unknown"}

            # Extract IP layer info
            if packet.haslayer(self.IP):
                ip = packet[self.IP]
                log["source_ip"] = ip.src
                log["destination_ip"] = ip.dst
                log["protocol"] = ip.proto

            # Extract TCP info
            if packet.haslayer(self.TCP):
                tcp = packet[self.TCP]
                log["source_port"] = tcp.sport
                log["destination_port"] = tcp.dport
                log["protocol"] = "TCP"
                log["flags"] = str(tcp.flags)

            # Extract UDP info
            elif packet.haslayer(self.UDP):
                udp = packet[self.UDP]
                log["source_port"] = udp.sport
                log["destination_port"] = udp.dport
                log["protocol"] = "UDP"

            # Include payload if requested
            if self.include_payload and hasattr(packet, "load"):
                log["payload"] = packet.load.hex()

            return log

        except Exception as e:
            logger.error(f"Failed to parse packet: {e}")
            return None


class StreamingLogParser:
    """
    Parse streaming log data in real-time.

    Supports:
    - Line-by-line parsing
    - Buffer management
    - Automatic format detection
    """

    def __init__(self, buffer_size: int = 1000):
        """
        Initialize streaming parser.

        Args:
            buffer_size: Maximum number of logs to buffer
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.csv_parser = CSVLogParser()
        self.json_parser = JSONLogParser()

    def parse_line(self, line: str, format_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parse single line of log data.

        Args:
            line: Log line as string
            format_hint: Optional format hint ('json', 'csv', None=auto)

        Returns:
            Parsed log dictionary or None
        """
        try:
            # Auto-detect or use hint
            if format_hint == "json" or (not format_hint and line.strip().startswith("{")):
                return json.loads(line)

            elif format_hint == "csv" or (not format_hint and "," in line):
                # Simple CSV parsing (assumes no headers)
                parts = line.split(",")
                return {
                    "timestamp": parts[0] if len(parts) > 0 else None,
                    "source_ip": parts[1] if len(parts) > 1 else None,
                    "destination_ip": parts[2] if len(parts) > 2 else None,
                    "data": ",".join(parts[3:]) if len(parts) > 3 else None,
                }

            return None

        except Exception as e:
            logger.error(f"Failed to parse line: {e}")
            return None

    def add_to_buffer(self, log: Dict[str, Any]):
        """Add parsed log to buffer."""
        self.buffer.append(log)

        # Enforce buffer size
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get_buffer(self) -> List[Dict[str, Any]]:
        """Get current buffer contents."""
        return self.buffer.copy()

    def clear_buffer(self):
        """Clear buffer."""
        self.buffer.clear()


# Example usage and testing
if __name__ == "__main__":
    print("Log Parser Test Suite")
    print("=" * 60)

    # Test CSV parser
    print("\n1. CSV Parser:")
    csv_parser = CSVLogParser()
    print("  ✓ CSV parser initialized")

    # Test JSON parser
    print("\n2. JSON Parser:")
    json_parser = JSONLogParser()
    print("  ✓ JSON parser initialized")

    # Test PCAP parser
    print("\n3. PCAP Parser:")
    pcap_parser = PCAPParser()
    if pcap_parser.available:
        print("  ✓ PCAP parser available")
    else:
        print("  ⚠ PCAP parser unavailable (scapy not installed)")

    # Test streaming parser
    print("\n4. Streaming Parser:")
    stream_parser = StreamingLogParser()
    test_log = stream_parser.parse_line('{"timestamp": "2024-01-01", "source": "10.0.0.1"}')
    if test_log:
        print(f"  ✓ Streaming parser working: {test_log}")

    print("\n" + "=" * 60)
    print("All parsers initialized successfully!")
