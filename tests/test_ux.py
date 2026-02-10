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
API_KEY_HEADER = {"X-API-Key": "hive-zero-admin"}

def test_health_check():
    # Health check is public but rate limited
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "active_experts" in data

def test_status_auth():
    # Status requires auth
    response = client.get("/status")
    assert response.status_code == 403 # No key

    response = client.get("/status", headers=API_KEY_HEADER)
    assert response.status_code == 200

def test_dashboard():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "<html" in response.text
    assert "HIVE-ZERO Dashboard" in response.text
    assert "showAuthModal" in response.text # Verify Auth UI logic

def test_control_endpoints_auth():
    # Pause
    response = client.post("/control/pause")
    assert response.status_code == 403

    response = client.post("/control/pause", headers=API_KEY_HEADER)
    assert response.status_code == 200
    assert response.json()["status"] == "paused"

    # Resume
    response = client.post("/control/resume", headers=API_KEY_HEADER)
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_dry_run_auth():
    payload = {
        "logs": [{"src_ip": "1.1.1.1", "dst_ip": "2.2.2.2", "port": 80, "proto": 6}],
        "top_k": 3,
        "dry_run": True
    }
    response = client.post("/execute", json=payload)
    assert response.status_code == 403

    response = client.post("/execute", json=payload, headers=API_KEY_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "dry_run"

def test_cors():
    response = client.options("/health", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET"
    })
    assert response.status_code == 200
    # Starlette/FastAPI mirrors the origin if allowed
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
