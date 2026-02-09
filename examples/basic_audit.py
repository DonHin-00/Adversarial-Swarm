"""Example: Basic Security Audit"""
import logging
from src.security_core.config import SecureConfig, Environment
from src.orchestration.coordinator import AgentCoordinator
from src.agents.network_scanner import NetworkScannerAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    print("="*60)
    print("Adversarial Swarm - Basic Security Audit")
    print("="*60)
    config = SecureConfig(environment=Environment.DEVELOPMENT)
    coordinator = AgentCoordinator(config, max_concurrent_agents=5)
    coordinator.register_agent_type("network_scanner", NetworkScannerAgent)
    scanner_id = coordinator.deploy_agent("network_scanner", priority="high")
    result = coordinator.execute_mission("full_security_audit")
    print(f"\nMission: {result.mission_id}")
    print(f"Success: {result.success}")
    print(f"Risk Score: {result.risk_score:.2f}")
    coordinator.shutdown()
    print("\nExample completed!")

if __name__ == "__main__":
    main()
