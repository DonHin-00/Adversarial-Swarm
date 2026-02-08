# HIVE-ZERO API Reference

## Control Endpoints (C2)

### `POST /control/pause`
Pauses all swarm operations.
- **Response**: `{"status": "paused"}`

### `POST /control/resume`
Resumes operations.
- **Response**: `{"status": "running"}`

### `POST /control/set_goal`
Overrides strategic planning.
- **Body**: `{"goal_idx": int}` (0=Recon, 1=Infiltrate, 2=Persist, 3=Exfiltrate)

### `POST /control/approve_action`
Approves a pending high-risk action.
- **Body**: `{"action_id": str, "decision": "approve" | "deny"}`

## Execution Endpoints

### `POST /execute`
Submit network logs for analysis and action generation.
- **Body**:
  ```json
  {
    "logs": [
      {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "port": 80, "proto": 6}
    ],
    "top_k": 3
  }
  ```
- **Response**:
  ```json
  {
    "strategy": 1,
    "actions": {
      "topology": [[...]],
      "defense_score": [0.1, 0.9]
    }
  }
  ```

### `GET /graph/viz`
Returns the current knowledge graph in Cytoscape.js JSON format.

## gRPC Swarm Protocol
Defined in `protos/swarm.proto`.
- `ShareKnowledge`: Broadcast embedding updates.
- `SyncState`: Retrieve global swarm aggregation.
