import sys
import os
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# Mock dependencies before import
sys.modules["torch"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()
sys.modules["hive_zero_core.hive_mind"] = MagicMock()
sys.modules["hive_zero_core.orchestration"] = MagicMock()
sys.modules["hive_zero_core.orchestration.strategic_planner"] = MagicMock()
sys.modules["hive_zero_core.orchestration.safety_monitor"] = MagicMock()
sys.modules["hive_zero_core.utils.logging_config"] = MagicMock()

# Setup mocks
mock_hive = MagicMock()
mock_hive.experts = [1, 2, 3]
sys.modules["hive_zero_core.hive_mind"].HiveMind.return_value = mock_hive

# Import app
sys.path.append(os.getcwd())
from hive_zero_core.api.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "active_experts" in data

def test_dashboard():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "<html" in response.text
    assert "HIVE-ZERO Dashboard" in response.text
    assert "cytoscape" in response.text # Verify new graph tab script

def test_control_endpoints():
    # Initial state should be active (not paused)
    status_resp = client.get("/health")
    assert status_resp.json()["paused"] == False

    # Pause
    response = client.post("/control/pause")
    assert response.status_code == 200
    assert response.json()["status"] == "paused"

    # Check paused state
    status_resp = client.get("/health")
    assert status_resp.json()["paused"] == True

    # Resume
    response = client.post("/control/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

    # Check resumed state
    status_resp = client.get("/health")
    assert status_resp.json()["paused"] == False

def test_dry_run():
    payload = {
        "logs": [{"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "port": 80, "proto": 6}],
        "top_k": 3,
        "dry_run": True
    }
    response = client.post("/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "dry_run"

def test_cors():
    response = client.options("/health", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET"
    })
    assert response.status_code == 200
    # Starlette/FastAPI mirrors the origin if allowed, effectively acting as wildcard but stricter header
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

def test_execute_error_handling():
    """Test that exceptions in /execute don't leak details"""
    # Setup mock to raise an exception during encoding
    from hive_zero_core.api.server import hive
    original_update = hive.log_encoder.update
    hive.log_encoder.update = MagicMock(side_effect=RuntimeError("Internal error with sensitive data"))
    
    payload = {
        "logs": [{"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "port": 80, "proto": 6}],
        "top_k": 3,
        "dry_run": False
    }
    response = client.post("/execute", json=payload)
    
    # Restore original
    hive.log_encoder.update = original_update
    
    # Should return 500 with generic error
    assert response.status_code == 500
    data = response.json()
    # Custom exception handler wraps the response in {"error": {"code": ..., "message": ...}}
    assert "error" in data
    assert data["error"]["code"] == 500
    assert data["error"]["message"] == "Internal server error"
    # Ensure sensitive data is not leaked
    assert "sensitive data" not in data["error"]["message"].lower()
    assert "RuntimeError" not in str(data)

