"""
Adversarial Swarm - Security-Focused AI Framework

A production-ready framework for autonomous security operations with
multi-agent AI, graph-based reasoning, and comprehensive security features.

Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Adversarial Swarm Team"
__license__ = "MIT"

from src.security_core.config import SecureConfig
from src.security_core.audit import AuditLogger
from src.orchestration.coordinator import AgentCoordinator

__all__ = [
    "SecureConfig",
    "AuditLogger",
    "AgentCoordinator",
]
