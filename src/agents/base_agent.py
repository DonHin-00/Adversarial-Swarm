"""
Base Agent Class

Foundation for all AI agents in the system. Provides:
- Lifecycle management
- Resource monitoring
- Error handling
- Secure communication
- Audit logging integration
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum
import logging
from dataclasses import dataclass
from datetime import datetime

from src.security_core.audit import AuditLogger, AuditEventType


class AgentStatus(Enum):
    """Agent lifecycle states."""
    INITIALIZED = "initialized"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class AgentPriority(Enum):
    """Agent priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    resource_usage: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {
                "cpu_percent": 0.0,
                "memory_mb": 0.0,
                "gpu_percent": 0.0
            }


class BaseAgent(ABC):
    """
    Base class for all AI agents.
    
    Provides common functionality:
    - Lifecycle management (init, start, stop, pause)
    - Resource monitoring and limits
    - Error handling and recovery
    - Audit logging
    - Health checks
    
    All agents must implement:
    - execute(): Main agent logic
    - validate_input(): Input validation
    - get_capabilities(): Agent capabilities description
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        priority: AgentPriority = AgentPriority.MEDIUM,
        max_execution_time: int = 300,  # seconds
        max_memory_mb: int = 1024
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (e.g., 'network_scanner')
            priority: Agent priority level
            max_execution_time: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.priority = priority
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        
        self.status = AgentStatus.INITIALIZED
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"agent.{agent_type}.{agent_id}")
        self.audit = AuditLogger()
        
        self.logger.info(f"Agent {agent_id} initialized with priority {priority.name}")
    
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent task.
        
        Args:
            task: Task parameters and data
            
        Returns:
            Task results
            
        Raises:
            ValueError: If task validation fails
            RuntimeError: If execution fails
        """
        pass
    
    @abstractmethod
    def validate_input(self, task: Dict[str, Any]) -> bool:
        """
        Validate task input.
        
        Args:
            task: Task to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities description.
        
        Returns:
            Dictionary describing agent capabilities
        """
        pass
    
    def start(self) -> None:
        """Start the agent."""
        if self.status != AgentStatus.READY:
            self.logger.warning(f"Agent {self.agent_id} not ready, current status: {self.status}")
            return
        
        self.status = AgentStatus.RUNNING
        self.logger.info(f"Agent {self.agent_id} started")
        
        self.audit.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="agent_start",
            resource=self.agent_id,
            agent_type=self.agent_type
        )
    
    def stop(self) -> None:
        """Stop the agent."""
        self.status = AgentStatus.STOPPED
        self.logger.info(f"Agent {self.agent_id} stopped")
        
        self.audit.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="agent_stop",
            resource=self.agent_id,
            agent_type=self.agent_type,
            tasks_completed=self.metrics.tasks_completed,
            tasks_failed=self.metrics.tasks_failed
        )
    
    def pause(self) -> None:
        """Pause the agent."""
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.PAUSED
            self.logger.info(f"Agent {self.agent_id} paused")
    
    def resume(self) -> None:
        """Resume the agent."""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING
            self.logger.info(f"Agent {self.agent_id} resumed")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "healthy": self.status in [AgentStatus.READY, AgentStatus.RUNNING],
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "average_execution_time": self.metrics.average_execution_time,
            }
        }
    
    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """Update agent metrics after task execution."""
        if success:
            self.metrics.tasks_completed += 1
        else:
            self.metrics.tasks_failed += 1
        
        # Update average execution time (running average)
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        self.metrics.average_execution_time = (
            (self.metrics.average_execution_time * (total_tasks - 1) + execution_time) / total_tasks
        )
        
        self.metrics.last_execution_time = datetime.now()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self.agent_id}, status={self.status.value})"
