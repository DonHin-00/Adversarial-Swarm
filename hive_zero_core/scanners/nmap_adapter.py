import logging
from typing import Any, Dict, List

import nmap


class NmapAdapter:
    def __init__(self):
        self.nm = nmap.PortScanner()
        self.logger = logging.getLogger(__name__)

    def scan_target(self, target: str, arguments: str = "-sS -T4 -p 1-1000") -> List[Dict[str, Any]]:
        """
        Executes Nmap scan and returns structured logs for HeteroLogEncoder.
        """
        self.logger.info(f"Scanning target: {target} with args: {arguments}")
        try:
            self.nm.scan(hosts=target, arguments=arguments)
        except nmap.PortScannerError as e:
            self.logger.error(f"Nmap scan failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return []

        logs = []
        for host in self.nm.all_hosts():
            if self.nm[host].state() != 'up':
                continue

            for proto in self.nm[host].all_protocols():
                ports = self.nm[host][proto].keys()
                for port in ports:
                    service = self.nm[host][proto][port]
                    # Map to Log Structure
                    logs.append({
                        'src_ip': '127.0.0.1', # Local scanner IP
                        'dst_ip': host,
                        'port': port,
                        'proto': 6 if proto == 'tcp' else 17, # Simplify
                        'service': service.get('name', ''),
                        'state': service.get('state', '')
                    })
        return logs

    def parse_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parses raw XML content (for offline/test usage).
        """
        # python-nmap usually parses files or execution.
        # We can simulate by saving to tmp file or using internal methods if exposed.
        # Simpler: Mock the internal dictionary structure for tests.
        pass
