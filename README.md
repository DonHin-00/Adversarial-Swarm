# HIVE-ZERO: Hierarchical Multi-Agent Reinforcement Learning System

A Hierarchical Multi-Agent Reinforcement Learning (H-MARL) system built with Python, PyTorch, Torch-Geometric, and Stable-Baselines3.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-repo/Adversarial-System-.git
    cd Adversarial-System-
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Before running the server, you must configure the environment variables for security.
Copy the example configuration:

```bash
cp .env.example .env
# Edit .env and set your HIVE_API_KEY
```

Or export them manually:

```bash
export HIVE_API_KEY="your-secret-key-123"
export HIVE_HOST="127.0.0.1" # Optional, default
export HIVE_PORT="8000"      # Optional, default
```

## Running the Server

Start the C2 API server:

```bash
python hive_zero_core/api/server.py
```

The server will start on `http://127.0.0.1:8000` (or your configured host/port).

## API Usage

All API endpoints (except documentation if enabled) require the `X-API-Key` header.

### Check Status

```bash
curl -H "X-API-Key: your-secret-key-123" http://127.0.0.1:8000/status
```

### Execute Command

```bash
curl -X POST http://127.0.0.1:8000/execute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-123" \
  -d '{
    "logs": [
      {"src_ip": "192.168.1.10", "dst_ip": "10.0.0.5", "port": 80, "proto": 6},
      {"src_ip": "192.168.1.11", "dst_ip": "10.0.0.5", "port": 443, "proto": 6}
    ],
    "top_k": 3
  }'
```
