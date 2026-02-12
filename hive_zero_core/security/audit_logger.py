"""
Audit logging for security-relevant events.

Provides comprehensive, tamper-evident logging of all security operations.
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityEvent(Enum):
    """Types of security events to audit."""
    # Authentication & Authorization
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    
    # Data Operations
    DATA_COLLECTED = "data_collected"
    DATA_EXFILTRATED = "data_exfiltrated"
    DATA_ENCRYPTED = "data_encrypted"
    DATA_DECRYPTED = "data_decrypted"
    
    # Key Operations
    KEY_GENERATED = "key_generated"
    KEY_ROTATED = "key_rotated"
    KEY_EXPIRED = "key_expired"
    KEY_COMPROMISED = "key_compromised"
    
    # Variant Operations
    VARIANT_CREATED = "variant_created"
    VARIANT_MERGED = "variant_merged"
    VARIANT_DIED = "variant_died"
    
    # Attack Operations
    PAYLOAD_GENERATED = "payload_generated"
    PAYLOAD_MUTATED = "payload_mutated"
    WAF_BYPASS_ATTEMPTED = "waf_bypass_attempted"
    
    # Security Violations
    INJECTION_ATTEMPT = "injection_attempt"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    
    # System Events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    CONFIG_CHANGED = "config_changed"


@dataclass
class AuditEntry:
    """Represents a single audit log entry."""
    timestamp: datetime
    event_type: SecurityEvent
    actor_id: str  # Who performed the action
    action: str  # What action was performed
    resource: Optional[str] = None  # What was acted upon
    result: str = "success"  # "success", "failure", "error"
    details: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None  # Hash of previous entry for tamper detection
    entry_hash: Optional[str] = None  # Hash of this entry
    
    def __post_init__(self):
        """Compute hash of this entry."""
        if self.entry_hash is None:
            self.entry_hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of entry for tamper detection.
        
        Includes timestamp, event type, actor, action, resource, result,
        and previous hash to create a chain.
        """
        hash_input = (
            f"{self.timestamp.isoformat()}|"
            f"{self.event_type.value}|"
            f"{self.actor_id}|"
            f"{self.action}|"
            f"{self.resource or ''}|"
            f"{self.result}|"
            f"{json.dumps(self.details, sort_keys=True)}|"
            f"{self.previous_hash or ''}"
        )
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    def verify_hash(self) -> bool:
        """Verify that entry hash is correct (not tampered)."""
        expected_hash = self.compute_hash()
        return expected_hash == self.entry_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor_id': self.actor_id,
            'action': self.action,
            'resource': self.resource,
            'result': self.result,
            'details': self.details,
            'previous_hash': self.previous_hash,
            'entry_hash': self.entry_hash,
        }


class AuditLogger:
    """
    Tamper-evident audit logger.
    
    Features:
    - Cryptographic chaining of log entries
    - Tamper detection via hash verification
    - Structured logging with metadata
    - Automatic obfuscation of sensitive data
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.entries: List[AuditEntry] = []
        self.log_file = log_file
        self.last_hash: Optional[str] = None
        logger.info(f"AuditLogger initialized (file: {log_file or 'memory-only'})")
    
    def log_event(self, event_type: SecurityEvent, actor_id: str, action: str,
                  resource: Optional[str] = None, result: str = "success",
                  details: Optional[Dict[str, Any]] = None):
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            actor_id: Identifier of the actor performing the action
            action: Description of the action
            resource: Optional resource identifier
            result: Result of the action ("success", "failure", "error")
            details: Additional details (will be sanitized)
        """
        if details is None:
            details = {}
        
        # Obfuscate sensitive data in details
        sanitized_details = self._sanitize_details(details)
        
        # Create audit entry with chain
        entry = AuditEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            actor_id=actor_id,
            action=action,
            resource=resource,
            result=result,
            details=sanitized_details,
            previous_hash=self.last_hash
        )
        
        # Add to entries
        self.entries.append(entry)
        self.last_hash = entry.entry_hash
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(entry)
        
        # Also log to standard logger at appropriate level
        log_level = logging.INFO if result == "success" else logging.WARNING
        logger.log(log_level, 
                  f"[AUDIT] {event_type.value}: {action} by {actor_id} -> {result}")
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or obfuscate sensitive fields from details."""
        sanitized = {}
        sensitive_keys = {'password', 'secret', 'key', 'token', 'credential'}
        
        for key, value in details.items():
            key_lower = key.lower()
            # Check if key contains sensitive terms
            if any(term in key_lower for term in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                # Truncate long strings
                sanitized[key] = value[:1000] + "...[truncated]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _write_to_file(self, entry: AuditEntry):
        """Write audit entry to file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log to file: {e}")
    
    def verify_integrity(self) -> Tuple[bool, List[int]]:
        """
        Verify integrity of audit log chain.
        
        Returns:
            Tuple of (all_valid, list_of_invalid_indices)
        """
        invalid_indices = []
        
        for i, entry in enumerate(self.entries):
            # Verify hash
            if not entry.verify_hash():
                invalid_indices.append(i)
                logger.error(f"Audit entry {i} has invalid hash!")
            
            # Verify chain
            if i > 0:
                expected_prev_hash = self.entries[i-1].entry_hash
                if entry.previous_hash != expected_prev_hash:
                    invalid_indices.append(i)
                    logger.error(f"Audit entry {i} has broken chain!")
        
        all_valid = len(invalid_indices) == 0
        return all_valid, invalid_indices
    
    def get_events_by_type(self, event_type: SecurityEvent) -> List[AuditEntry]:
        """Get all events of a specific type."""
        return [e for e in self.entries if e.event_type == event_type]
    
    def get_events_by_actor(self, actor_id: str) -> List[AuditEntry]:
        """Get all events by a specific actor."""
        return [e for e in self.entries if e.actor_id == actor_id]
    
    def get_failed_events(self) -> List[AuditEntry]:
        """Get all failed events."""
        return [e for e in self.entries if e.result != "success"]
    
    def export_log(self, filename: str):
        """Export audit log to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump([e.to_dict() for e in self.entries], f, indent=2)
            logger.info(f"Exported {len(self.entries)} audit entries to {filename}")
        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger


def log_security_event(event_type: SecurityEvent, actor_id: str, action: str,
                      resource: Optional[str] = None, result: str = "success",
                      details: Optional[Dict[str, Any]] = None):
    """Convenience function to log security event to global logger."""
    get_audit_logger().log_event(event_type, actor_id, action, resource, result, details)
