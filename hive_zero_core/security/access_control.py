"""
Access control and authorization for operations.

Implements role-based access control (RBAC) and operation authorization.
"""

import logging
from typing import Dict, Set, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be controlled."""

    # Data operations
    DATA_COLLECT = "data_collect"
    DATA_EXFILTRATE = "data_exfiltrate"
    DATA_ENCRYPT = "data_encrypt"
    DATA_DECRYPT = "data_decrypt"

    # Variant operations
    VARIANT_CREATE = "variant_create"
    VARIANT_MERGE = "variant_merge"
    VARIANT_TERMINATE = "variant_terminate"

    # Attack operations
    PAYLOAD_GENERATE = "payload_generate"
    PAYLOAD_EXECUTE = "payload_execute"
    WAF_BYPASS = "waf_bypass"

    # System operations
    CONFIG_MODIFY = "config_modify"
    KEY_GENERATE = "key_generate"
    KEY_ROTATE = "key_rotate"

    # Administrative
    ADMIN_ACCESS = "admin_access"
    AUDIT_VIEW = "audit_view"


class AccessLevel(Enum):
    """Access levels for role-based access control."""

    NONE = 0  # No access
    READ = 1  # Read-only
    WRITE = 2  # Read and write
    EXECUTE = 3  # Read, write, and execute
    ADMIN = 4  # Full administrative access


@dataclass
class Role:
    """Represents a role with specific permissions."""

    role_name: str
    access_level: AccessLevel
    allowed_operations: Set[OperationType] = field(default_factory=set)
    rate_limits: Dict[OperationType, int] = field(default_factory=dict)  # operations per minute
    description: str = ""

    def can_perform(self, operation: OperationType) -> bool:
        """Check if role can perform operation."""
        # Admin can do everything
        if self.access_level == AccessLevel.ADMIN:
            return True
        return operation in self.allowed_operations


@dataclass
class Actor:
    """Represents an actor (user, variant, system component)."""

    actor_id: str
    role: Role
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    operation_counts: Dict[OperationType, List[datetime]] = field(default_factory=dict)
    is_active: bool = True
    session_timeout_minutes: int = 60

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_active = datetime.now()

    def is_session_valid(self) -> bool:
        """Check if session hasn't timed out."""
        if not self.is_active:
            return False
        timeout = timedelta(minutes=self.session_timeout_minutes)
        return (datetime.now() - self.last_active) < timeout

    def record_operation(self, operation: OperationType):
        """Record an operation for rate limiting."""
        if operation not in self.operation_counts:
            self.operation_counts[operation] = []
        self.operation_counts[operation].append(datetime.now())
        self.update_activity()

    def get_operation_rate(self, operation: OperationType, window_minutes: int = 1) -> int:
        """Get operation count in time window."""
        if operation not in self.operation_counts:
            return 0

        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_ops = [t for t in self.operation_counts[operation] if t >= cutoff]

        # Clean old entries
        self.operation_counts[operation] = recent_ops

        return len(recent_ops)


class AccessController:
    """
    Role-based access control system.

    Features:
    - Role-based permissions
    - Operation authorization
    - Rate limiting
    - Session management
    """

    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.actors: Dict[str, Actor] = {}
        self._initialize_default_roles()
        logger.info("AccessController initialized")

    def _initialize_default_roles(self):
        """Initialize default roles for the system."""
        # Read-only role
        self.register_role(
            Role(
                role_name="readonly",
                access_level=AccessLevel.READ,
                allowed_operations={OperationType.AUDIT_VIEW},
                description="Read-only access",
            )
        )

        # Operator role
        self.register_role(
            Role(
                role_name="operator",
                access_level=AccessLevel.EXECUTE,
                allowed_operations={
                    OperationType.DATA_COLLECT,
                    OperationType.VARIANT_CREATE,
                    OperationType.PAYLOAD_GENERATE,
                    OperationType.AUDIT_VIEW,
                },
                rate_limits={
                    OperationType.DATA_COLLECT: 100,  # per minute
                    OperationType.VARIANT_CREATE: 10,
                    OperationType.PAYLOAD_GENERATE: 50,
                },
                description="Standard operator with execute permissions",
            )
        )

        # Advanced operator role
        self.register_role(
            Role(
                role_name="advanced",
                access_level=AccessLevel.EXECUTE,
                allowed_operations={
                    OperationType.DATA_COLLECT,
                    OperationType.DATA_EXFILTRATE,
                    OperationType.DATA_ENCRYPT,
                    OperationType.DATA_DECRYPT,
                    OperationType.VARIANT_CREATE,
                    OperationType.VARIANT_MERGE,
                    OperationType.PAYLOAD_GENERATE,
                    OperationType.PAYLOAD_EXECUTE,
                    OperationType.WAF_BYPASS,
                    OperationType.AUDIT_VIEW,
                },
                rate_limits={
                    OperationType.DATA_EXFILTRATE: 50,
                    OperationType.VARIANT_MERGE: 20,
                    OperationType.PAYLOAD_EXECUTE: 100,
                },
                description="Advanced operator with exfiltration and merge capabilities",
            )
        )

        # Admin role
        self.register_role(
            Role(
                role_name="admin",
                access_level=AccessLevel.ADMIN,
                allowed_operations=set(OperationType),  # All operations
                description="Full administrative access",
            )
        )

    def register_role(self, role: Role):
        """Register a new role."""
        self.roles[role.role_name] = role
        logger.debug(f"Registered role: {role.role_name}")

    def register_actor(
        self, actor_id: str, role_name: str, session_timeout_minutes: int = 60
    ) -> Actor:
        """
        Register a new actor with a role.

        Args:
            actor_id: Unique identifier for actor
            role_name: Name of role to assign
            session_timeout_minutes: Session timeout in minutes

        Returns:
            Actor object
        """
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' not found")

        role = self.roles[role_name]
        actor = Actor(actor_id=actor_id, role=role, session_timeout_minutes=session_timeout_minutes)

        self.actors[actor_id] = actor
        logger.info(f"Registered actor {actor_id} with role {role_name}")
        return actor

    def authorize_operation(self, actor_id: str, operation: OperationType) -> bool:
        """
        Check if actor is authorized to perform operation.

        Args:
            actor_id: Actor identifier
            operation: Operation to authorize

        Returns:
            True if authorized, False otherwise
        """
        # Get actor
        actor = self.actors.get(actor_id)
        if actor is None:
            logger.warning(f"Unknown actor: {actor_id}")
            return False

        # Check if session is valid
        if not actor.is_session_valid():
            logger.warning(f"Session expired for actor: {actor_id}")
            return False

        # Check if role allows operation
        if not actor.role.can_perform(operation):
            logger.warning(f"Actor {actor_id} not authorized for {operation.value}")
            return False

        # Check rate limiting
        if operation in actor.role.rate_limits:
            rate_limit = actor.role.rate_limits[operation]
            current_rate = actor.get_operation_rate(operation)

            if current_rate >= rate_limit:
                logger.warning(
                    f"Rate limit exceeded for {actor_id}: {operation.value} "
                    f"({current_rate}/{rate_limit} per minute)"
                )
                return False

        # Record operation
        actor.record_operation(operation)

        return True

    def revoke_actor(self, actor_id: str):
        """Revoke access for an actor."""
        if actor_id in self.actors:
            self.actors[actor_id].is_active = False
            logger.info(f"Revoked access for actor: {actor_id}")

    def get_actor_stats(self, actor_id: str) -> Dict:
        """Get statistics for an actor."""
        actor = self.actors.get(actor_id)
        if actor is None:
            return {}

        return {
            "actor_id": actor.actor_id,
            "role": actor.role.role_name,
            "access_level": actor.role.access_level.name,
            "is_active": actor.is_active,
            "session_valid": actor.is_session_valid(),
            "created_at": actor.created_at.isoformat(),
            "last_active": actor.last_active.isoformat(),
            "operation_counts": {
                op.value: len(times) for op, times in actor.operation_counts.items()
            },
        }


# Global access controller
_global_access_controller: Optional[AccessController] = None


def get_access_controller() -> AccessController:
    """Get global access controller instance."""
    global _global_access_controller
    if _global_access_controller is None:
        _global_access_controller = AccessController()
    return _global_access_controller


def authorize_operation(actor_id: str, operation: OperationType) -> bool:
    """Convenience function for operation authorization."""
    return get_access_controller().authorize_operation(actor_id, operation)
