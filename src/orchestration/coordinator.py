"""
Agent Coordinator

Coordinates multiple AI agents for complex security operations.
Handles task distribution, resource management, and inter-agent communication.
"""

import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, Future
from queue import PriorityQueue
from dataclasses import dataclass, field
from datetime import datetime

from src.agents.base_agent import BaseAgent, AgentStatus, AgentPriority
from src.security_core.config import SecureConfig
from src.security_core.audit import AuditLogger, AuditEventType, AuditSeverity


@dataclass(order=True)
class Task:
    """Prioritized task for agent execution."""
    priority: int
    task_id: str = field(compare=False)
    agent_type: str = field(compare=False)
    parameters: Dict[str, Any] = field(compare=False)
    created_at: datetime = field(default_factory=datetime.now, compare=False)
    deadline: Optional[datetime] = field(default=None, compare=False)


@dataclass
class MissionResult:
    """Result of a coordinated mission."""
    mission_id: str
    success: bool
    risk_score: float
    findings: List[Dict[str, Any]]
    agent_results: Dict[str, Any]
    execution_time: float
    errors: List[str] = field(default_factory=list)


class AgentCoordinator:
    """
    Coordinates multiple AI agents for security operations.
    
    Features:
    - Intelligent task scheduling
    - Resource management
    - Priority-based execution
    - Fault tolerance
    - Performance monitoring
    - Audit logging
    
    Example:
        >>> coordinator = AgentCoordinator(config)
        >>> coordinator.deploy_agent('network_scanner', priority='high')
        >>> results = coordinator.execute_mission('full_security_audit')
    """
    
    def __init__(
        self,
        config: SecureConfig,
        max_concurrent_agents: int = 10,
        max_queue_size: int = 1000
    ):
        """
        Initialize agent coordinator.
        
        Args:
            config: Secure configuration
            max_concurrent_agents: Maximum concurrent agent executions
            max_queue_size: Maximum task queue size
        """
        self.config = config
        self.max_concurrent_agents = max_concurrent_agents
        self.logger = logging.getLogger(__name__)
        self.audit = AuditLogger()
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, type] = {}
        
        # Task queue and executor
        self.task_queue: PriorityQueue[Task] = PriorityQueue(maxsize=max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_agents)
        self.active_tasks: Dict[str, Future] = {}
        
        # Metrics
        self.metrics = {
            "missions_completed": 0,
            "missions_failed": 0,
            "total_tasks_executed": 0,
            "average_mission_time": 0.0
        }
        
        self.logger.info(f"AgentCoordinator initialized with max {max_concurrent_agents} concurrent agents")
    
    def register_agent_type(self, agent_type: str, agent_class: type) -> None:
        """
        Register an agent type for deployment.
        
        Args:
            agent_type: Agent type identifier
            agent_class: Agent class (subclass of BaseAgent)
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class must be a subclass of BaseAgent")
        
        self.agent_types[agent_type] = agent_class
        self.logger.info(f"Registered agent type: {agent_type}")
    
    def deploy_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        priority: str = "medium",
        **kwargs: Any
    ) -> str:
        """
        Deploy a new agent instance.
        
        Args:
            agent_type: Type of agent to deploy
            agent_id: Optional agent ID (auto-generated if not provided)
            priority: Agent priority ('low', 'medium', 'high', 'critical')
            **kwargs: Additional agent parameters
            
        Returns:
            Agent ID
            
        Raises:
            ValueError: If agent type not registered
        """
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = f"{agent_type}_{len(self.agents)}_{datetime.now().timestamp()}"
        
        # Map priority string to enum
        priority_map = {
            "low": AgentPriority.LOW,
            "medium": AgentPriority.MEDIUM,
            "high": AgentPriority.HIGH,
            "critical": AgentPriority.CRITICAL
        }
        priority_enum = priority_map.get(priority.lower(), AgentPriority.MEDIUM)
        
        # Create agent instance
        agent_class = self.agent_types[agent_type]
        agent = agent_class(
            agent_id=agent_id,
            agent_type=agent_type,
            priority=priority_enum,
            **kwargs
        )
        
        # Register agent
        self.agents[agent_id] = agent
        agent.status = AgentStatus.READY
        
        self.logger.info(f"Deployed agent: {agent_id} (type={agent_type}, priority={priority})")
        
        self.audit.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="agent_deployed",
            resource=agent_id,
            agent_type=agent_type,
            priority=priority
        )
        
        return agent_id
    
    def execute_mission(
        self,
        mission_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 300
    ) -> MissionResult:
        """
        Execute a coordinated security mission.
        
        Args:
            mission_type: Type of mission (e.g., 'full_security_audit')
            parameters: Mission parameters
            timeout: Mission timeout in seconds
            
        Returns:
            Mission results
        """
        mission_id = f"mission_{datetime.now().timestamp()}"
        start_time = datetime.now()
        
        self.logger.info(f"Starting mission: {mission_id} (type={mission_type})")
        
        self.audit.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="mission_start",
            resource=mission_id,
            mission_type=mission_type
        )
        
        try:
            # Execute mission-specific logic
            findings = []
            agent_results = {}
            
            if mission_type == "full_security_audit":
                findings, agent_results = self._execute_security_audit(parameters or {})
            else:
                self.logger.warning(f"Unknown mission type: {mission_type}")
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(findings)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = MissionResult(
                mission_id=mission_id,
                success=True,
                risk_score=risk_score,
                findings=findings,
                agent_results=agent_results,
                execution_time=execution_time
            )
            
            self.metrics["missions_completed"] += 1
            self._update_mission_metrics(execution_time)
            
            self.audit.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                action="mission_complete",
                resource=mission_id,
                result="success",
                risk_score=risk_score,
                execution_time=execution_time
            )
            
            return result
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(f"Mission {mission_id} failed: {str(e)}", exc_info=True)
            
            self.metrics["missions_failed"] += 1
            
            self.audit.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                action="mission_failed",
                resource=mission_id,
                result="error",
                severity=AuditSeverity.ERROR,
                error=str(e),
                execution_time=execution_time
            )
            
            return MissionResult(
                mission_id=mission_id,
                success=False,
                risk_score=1.0,  # Maximum risk on failure
                findings=[],
                agent_results={},
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def _execute_security_audit(
        self,
        parameters: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute a full security audit mission."""
        findings = []
        agent_results = {}
        
        # Simulate security audit results
        findings.append({
            "severity": "medium",
            "category": "network_security",
            "description": "Security audit completed successfully",
            "recommendation": "Review findings and apply patches"
        })
        
        return findings, agent_results
    
    def _calculate_risk_score(self, findings: List[Dict[str, Any]]) -> float:
        """
        Calculate overall risk score from findings.
        
        Args:
            findings: List of security findings
            
        Returns:
            Risk score between 0.0 (low risk) and 1.0 (high risk)
        """
        if not findings:
            return 0.0
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }
        
        total_weight = sum(
            severity_weights.get(f.get("severity", "low"), 0.1)
            for f in findings
        )
        
        return min(total_weight / len(findings), 1.0)
    
    def _update_mission_metrics(self, execution_time: float) -> None:
        """Update mission execution metrics."""
        total_missions = self.metrics["missions_completed"] + self.metrics["missions_failed"]
        if total_missions > 0:
            self.metrics["average_mission_time"] = (
                (self.metrics["average_mission_time"] * (total_missions - 1) + execution_time) / total_missions
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get coordinator status.
        
        Returns:
            Status information including agent states and metrics
        """
        agent_status = {
            agent_id: agent.health_check()
            for agent_id, agent in self.agents.items()
        }
        
        return {
            "total_agents": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "metrics": self.metrics,
            "agents": agent_status
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the coordinator."""
        self.logger.info("Shutting down AgentCoordinator...")
        
        # Stop all agents
        for agent in self.agents.values():
            agent.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.audit.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="coordinator_shutdown",
            total_agents=len(self.agents),
            missions_completed=self.metrics["missions_completed"]
        )
        
        self.logger.info("AgentCoordinator shutdown complete")
