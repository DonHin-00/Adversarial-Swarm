from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, Request, Security, Depends
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import torch
import uvicorn
import asyncio
import json
import os
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.orchestration.strategic_planner import StrategicPlanner
from hive_zero_core.orchestration.safety_monitor import SafetyMonitor
from hive_zero_core.utils.logging_config import setup_logger

logger = setup_logger("API")

API_KEY = os.getenv("HIVE_ZERO_API_KEY", "hive-zero-admin")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Rate Limiter Setup
limiter = Limiter(key_func=get_remote_address)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Could not validate credentials"
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n" + "="*50)
    print(" 🎨 HIVE-ZERO API Initialized")
    print(" 📊 Dashboard: http://localhost:8000/dashboard")
    print(" 📚 API Docs:  http://localhost:8000/docs")
    print(f" 🔑 API Key:   [REDACTED]")
    print("="*50 + "\n")
    logger.info("System startup complete.")
    yield
    # Shutdown
    logger.info("System shutdown initiated.")

app = FastAPI(
    title="HIVE-ZERO C2 Interface",
    description="Command & Control Interface for the HIVE-ZERO Adversarial Swarm.",
    version="1.1.0",
    lifespan=lifespan
)

# Attach Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for cross-origin requests
# Strictly configurable via env var, default to wildcard for dev convenience
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hive = HiveMind(observation_dim=64)
planner = StrategicPlanner(observation_dim=64)
monitor = SafetyMonitor()

is_paused = False
active_websockets: List[WebSocket] = []

class LogEntry(BaseModel):
    src_ip: str = Field(..., description="Source IP address", examples=["192.168.1.5"])
    dst_ip: str = Field(..., description="Destination IP address", examples=["10.0.0.1"])
    port: int = Field(..., description="Target port number", examples=[80])
    proto: int = Field(..., description="Protocol ID (e.g., 6 for TCP)", examples=[6])
    src_port: Optional[int] = Field(0, description="Source port number")

class CommandRequest(BaseModel):
    logs: List[LogEntry] = Field(..., description="List of network logs to analyze")
    top_k: int = Field(3, description="Number of experts to activate", ge=1, le=10)
    dry_run: bool = Field(False, description="If True, simulate execution without taking action")

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
    )

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/dashboard")

@app.get("/health", tags=["System"], summary="Get System Health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Returns the current operational status of the HiveMind."""
    return {
        "status": "operational",
        "version": "1.1.0",
        "active_experts": len(hive.experts),
        "paused": is_paused,
        "auth_required": True
    }

@app.get("/status", tags=["System"], summary="Get Status")
@limiter.limit("60/minute")
async def status_check(request: Request, api_key: str = Depends(get_api_key)):
    """Returns status specifically for monitoring tools. Requires Auth."""
    return {
        "paused": is_paused,
        "connections": len(active_websockets)
    }

@app.post("/control/pause", tags=["Control"], summary="Pause Operations")
async def pause_system(api_key: str = Depends(get_api_key)):
    """Pauses all swarm execution. Requires Auth."""
    global is_paused
    is_paused = True
    logger.warning("System PAUSED by operator.")
    await broadcast_status({"type": "alert", "message": "System PAUSED by operator"})
    return {"status": "paused"}

@app.post("/control/resume", tags=["Control"], summary="Resume Operations")
async def resume_system(api_key: str = Depends(get_api_key)):
    """Resumes swarm execution. Requires Auth."""
    global is_paused
    is_paused = False
    logger.info("System RESUMED by operator.")
    await broadcast_status({"type": "info", "message": "System RESUMED by operator"})
    return {"status": "running"}

@app.get("/dashboard", tags=["UI"], summary="Web Dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serves the HIVE-ZERO Web Dashboard."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse(content="<h1>Dashboard Template Not Found</h1>", status_code=500)
    return HTMLResponse(content=template_path.read_text())

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    # WebSockets don't use standard HTTP headers easily, but we can check query params or initial message
    # For now, let's keep it open but log connections heavily
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        active_websockets.remove(websocket)

async def broadcast_status(data: Dict):
    for ws in active_websockets:
        try:
            await ws.send_json(data)
        except:
            pass

@app.get("/graph/viz", tags=["Visualization"], summary="Get Graph Data")
@limiter.limit("30/minute")
async def get_graph_viz(request: Request, api_key: str = Depends(get_api_key)):
    """Returns the current knowledge graph for Cytoscape.js visualization. Requires Auth."""
    if hasattr(hive, 'last_data'):
        return hive.log_encoder.to_cytoscape_json(hive.last_data)
    # Return dummy data for viz testing if no real data
    return {
        "nodes": [
            {"data": {"id": "core", "label": "HiveMind Core", "type": "core"}},
            {"data": {"id": "net", "label": "Internet", "type": "network"}}
        ],
        "edges": [
            {"data": {"source": "core", "target": "net", "label": "scans"}}
        ]
    }

@app.post("/execute", tags=["Control"], summary="Execute Swarm Strategy")
@limiter.limit("20/minute")
async def execute_swarm(request: Request, cmd: CommandRequest, api_key: str = Depends(get_api_key)):
    """
    Analyzes logs and executes the optimal adversarial strategy. Requires Auth.
    """
    global is_paused
    if is_paused:
        return {"status": "paused"}

    if cmd.dry_run:
        logger.info("Executing DRY RUN simulation.")
        await broadcast_status({"type": "info", "message": "Dry Run: Simulation started"})
        # Simulate processing delay
        await asyncio.sleep(0.5)
        return {
            "status": "dry_run",
            "strategy": "simulation",
            "actions": {"simulated_expert": "would_execute_attack"}
        }

    raw_logs = [log.model_dump() for log in cmd.logs]

    try:
        data = hive.log_encoder.update(raw_logs)
        hive.last_data = data # Cache for viz

        device = next(hive.parameters()).device
        if 'ip' in data.node_types and hasattr(data['ip'], 'x') and data['ip'].x.size(0) > 0:
            global_state = torch.mean(data['ip'].x, dim=0, keepdim=True)
        else:
            global_state = torch.zeros(1, 64, device=device)

        strategy = planner(global_state)
        current_goal = strategy["goal_idx"].item()

        safe, reason = monitor.check_safety(global_state, torch.zeros(1, 128), 0.0)
        if not safe:
            await broadcast_status({"type": "alert", "message": f"Safety Violation: {reason}"})
            logger.warning(f"Safety violation: {reason}")
            return {"status": "blocked", "reason": reason}

        results = hive.forward(raw_logs, top_k=cmd.top_k)

        formatted_results = {}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                formatted_results[k] = v.detach().cpu().tolist()
            else:
                formatted_results[k] = str(v)

        await broadcast_status({
            "type": "execution",
            "goal": current_goal,
            "experts_active": list(formatted_results.keys())
        })

        logger.info(f"Execution complete. Strategy: {current_goal}")

        return {
            "strategy": current_goal,
            "actions": formatted_results
        }

    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
