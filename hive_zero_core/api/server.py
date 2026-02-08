from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import uvicorn
from hive_zero_core.hive_mind import HiveMind
from hive_zero_core.orchestration.strategic_planner import StrategicPlanner

app = FastAPI(title="HIVE-ZERO C2 Interface")

hive = HiveMind(observation_dim=64)
planner = StrategicPlanner(observation_dim=64)

class LogEntry(BaseModel):
    src_ip: str
    dst_ip: str
    port: int
    proto: int
    src_port: Optional[int] = 0

class CommandRequest(BaseModel):
    logs: List[LogEntry]
    top_k: int = 3
    strategy_override: Optional[int] = None

@app.get("/status")
def status():
    return {"status": "online", "experts": len(hive.experts)}

@app.post("/execute")
def execute_swarm(request: CommandRequest):
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
