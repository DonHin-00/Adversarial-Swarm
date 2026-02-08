from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import uvicorn
import asyncio
import json
from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.orchestration.strategic_planner import StrategicPlanner
from hive_zero_core.orchestration.safety_monitor import SafetyMonitor

app = FastAPI(title="HIVE-ZERO C2 Interface")

hive = HiveMind(observation_dim=64)
planner = StrategicPlanner(observation_dim=64)
monitor = SafetyMonitor()

is_paused = False
active_websockets: List[WebSocket] = []

class LogEntry(BaseModel):
    src_ip: str
    dst_ip: str
    port: int
    proto: int
    src_port: Optional[int] = 0

class CommandRequest(BaseModel):
    logs: List[LogEntry]
    top_k: int = 3

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
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

@app.get("/graph/viz")
def get_graph_viz():
    # Return last processed graph structure
    # We need access to the data object or store it in hive.
    # Assuming hive.log_encoder stores maps from last update.
    # But we need 'data' object.
    # Ideally store last_data in hive state.
    if hasattr(hive, 'last_data'):
        return hive.log_encoder.to_cytoscape_json(hive.last_data)
    return {"nodes": [], "edges": []}

@app.post("/execute")
async def execute_swarm(request: CommandRequest):
    global is_paused
    if is_paused:
        return {"status": "paused"}

    raw_logs = [log.model_dump() for log in request.logs]

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
            return {"status": "blocked", "reason": reason}

        results = hive.forward(raw_logs, top_k=request.top_k)

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

        return {
            "strategy": current_goal,
            "actions": formatted_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
