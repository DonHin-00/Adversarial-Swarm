import time
import logging
from typing import Callable, Any, Dict

class CircuitBreaker:
    """
    Self-Healing wrapper for Experts.
    """
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED" # CLOSED (Normal), OPEN (Tripped), HALF-OPEN (Testing)
        self.logger = logging.getLogger("CircuitBreaker")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF-OPEN"
                self.logger.info("Circuit Half-Open: Retrying...")
            else:
                self.logger.warning("Circuit Open: Call blocked.")
                return None # or raise Exception

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
                self.logger.info("Circuit Closed (Recovered)")
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            self.logger.error(f"Call failed: {e}. Failures: {self.failures}")

            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.critical("Circuit Tripped! Entering Cool-down.")

            raise e
