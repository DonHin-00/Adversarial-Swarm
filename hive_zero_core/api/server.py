from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
import uvicorn
import os
from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.orchestration.strategic_planner import StrategicPlanner

app = FastAPI(title="HIVE-ZERO C2 Interface")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    expected_api_key = os.getenv("HIVE_API_KEY")
    if not expected_api_key:
        # If no key is configured, warn but allow (or fail secure - let's fail secure)
        raise HTTPException(status_code=500, detail="Server configuration error: API Key not set")

    if api_key_header == expected_api_key:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

hive = HiveMind(observation_dim=64)
planner = StrategicPlanner(observation_dim=64)

class LogEntry(BaseModel):
    src_ip: str
    dst_ip: str
    port: int
    proto: int
    src_port: Optional[int] = 0

class CommandRequest(BaseModel):
    logs: List[LogEntry] = Field(..., max_length=1000, description="List of logs to process, max 1000")
    top_k: int = 3
    strategy_override: Optional[int] = None

@app.get("/status")
def status(api_key: str = Depends(get_api_key)):
    return {"status": "online", "experts": len(hive.experts)}

@app.post("/execute")
def execute_swarm(request: CommandRequest, api_key: str = Depends(get_api_key)):
    raw_logs = [log.model_dump() for log in request.logs]

    try:
        data = hive.log_encoder.update(raw_logs)

        if 'ip' in data.node_types and data['ip'].x.size(0) > 0:
            global_state = torch.mean(data['ip'].x, dim=0, keepdim=True)
        else:
            global_state = torch.zeros(1, 64)

        strategy = planner(global_state)
        current_goal = strategy["goal_idx"].item()

        results = hive.forward(raw_logs, top_k=request.top_k)

        formatted_results: Dict[str, Any] = {}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                formatted_results[k] = v.detach().cpu().tolist()
            else:
                formatted_results[k] = str(v)

        return {
            "strategy": current_goal,
            "actions": formatted_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    host = os.getenv("HIVE_HOST", "127.0.0.1")
    port = int(os.getenv("HIVE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
