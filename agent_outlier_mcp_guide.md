# Agent Divergence Analysis as an MCP Server Tool

## Overview

This guide explains how to set up the Agent Divergence Analysis as a local MCP (Model-Controller-Presenter) server tool that can run on your database. This setup provides a robust way to find agents that diverge significantly from population norms through a flexible API rather than requiring direct script execution.

## Why Use the Server Approach

1. **Database Integration**: Connect directly to your simulation database
2. **API Interface**: Access divergence analysis via REST endpoints or command-line
3. **Consistent Results**: Cache computed results for efficiency
4. **Flexible Deployment**: Run on the same machine as your database or separately
5. **Programmatic Access**: Integrate with dashboards, notebooks, or other tools

## Setup Instructions

### 1. Install Required Dependencies

```bash
pip install flask numpy scipy pandas faiss-cpu tabulate requests
```

### 2. Configure Files

Ensure these files are in your project directory:
- `agent_divergence_detector.py` - The core divergence detection functionality
- `agent_divergence_server.py` - The Flask server providing the API
- `agent_divergence_client.py` - The command-line client for interacting with the server

### 3. Start the Server

Connect to your database:

```bash
python agent_divergence_server.py --db-path "path/to/simulation.db" --faiss-index-path "path/to/faiss_index"
```

Options:
- `--port` - Port to run the server on (default: 5000)
- `--host` - Host address to bind to (default: 127.0.0.1)
- `--results-dir` - Directory to store results (default: "results")
- `--debug` - Run in debug mode for development

### 4. Use the Client

Once the server is running, you can use the client to find divergent agents:

```bash
python agent_divergence_client.py find-divergent --metric mahalanobis_distance --top 5
```

Or find common divergent agents across multiple metrics:

```bash
python agent_divergence_client.py find-common-divergent
```

Check details of a specific agent:

```bash
python agent_divergence_client.py get-agent 56q2nhmuN2SqH9beAEmVqo
```

## Database Requirements

The server works with SQLite databases containing:

1. `agents` table with agent metadata
2. `agent_states` table with agent state history
3. `agent_actions` table with agent actions

Additionally, a FAISS index containing agent state vectors is required:
- `<faiss_index_path>.faiss`: The actual index data
- `<faiss_index_path>.json`: Metadata including vector IDs

## API Endpoints

The server exposes these API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check server status |
| `/api/metrics` | GET | List available distance metrics |
| `/api/divergent` | GET | Find divergent agents using a specific metric |
| `/api/common-divergent` | GET | Find agents that are divergent across multiple metrics |
| `/api/agent/<agent_id>` | GET | Get detailed information about a specific agent |

## Integration Examples

### Python Integration

```python
import requests

# Find divergent agents using Mahalanobis distance
response = requests.get("http://localhost:5000/api/divergent", params={
    "metric": "mahalanobis_distance",
    "top": 5,
    "threshold": 2.0
})

data = response.json()
top_divergent = data["top_divergent_agents"]

# Process the results
for agent in top_divergent:
    print(f"Agent {agent['short_id']}: {agent['divergence_score']}")
```

### PowerShell Integration

```powershell
# Find divergent agents and format as table
$response = Invoke-RestMethod -Uri "http://localhost:5000/api/divergent?metric=mahalanobis_distance&top=5"
$response.top_divergent_agents | Format-Table short_id, divergence_score, @{N='Type';E={$_.metadata.agent_type}}
```

### Dashboard Integration

The REST API can be integrated into dashboards like Grafana, Kibana, or custom web interfaces by making HTTP requests to the endpoints and visualizing the returned JSON data.

## Scaling Considerations

For larger deployments:

1. **Database Connection Pooling**: Modify the server to use connection pooling for better performance
2. **Docker Containerization**: Package the server in a Docker container for easy deployment
3. **Load Balancing**: Set up multiple server instances behind a load balancer for higher throughput
4. **Persistent Caching**: Implement Redis or another caching system for results persistence

## Troubleshooting

Common issues:

1. **Server Won't Start**: Check that the database path is correct and that Flask is installed
2. **Slow Performance**: Large FAISS indices can be memory-intensive; consider using a machine with more RAM
3. **Database Errors**: Ensure the database schema matches the expected format
4. **"Agent Not Found"**: Confirm the agent IDs exist in both the database and FAISS index 