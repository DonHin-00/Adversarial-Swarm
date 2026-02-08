import time
import random
import logging

class TrafficShaper:
    """
    Middleware to enforce stealthy networking.
    Adds jitter and fragmentation to evade IDS.
    """
    def __init__(self, min_jitter: float = 0.1, max_jitter: float = 0.5):
        self.min_jitter = min_jitter
        self.max_jitter = max_jitter
        self.logger = logging.getLogger("TrafficShaper")

    def apply_jitter(self):
        """Sleeps for a random interval."""
        delay = random.uniform(self.min_jitter, self.max_jitter)
        time.sleep(delay)

    def fragment_payload(self, payload: bytes, mtu: int = 1400) -> list[bytes]:
        """Splits payload into chunks to avoid signature detection on full packets."""
        chunks = [payload[i:i+mtu] for i in range(0, len(payload), mtu)]
        self.logger.info(f"Fragmented payload into {len(chunks)} packets")
        return chunks
