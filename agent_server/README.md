# LlamaIndex Agent Server

A FastAPI server with a LlamaIndex agent that uses tools to interact with the vLLM server.

## Features

- **LlamaIndex ReAct Agent** with tool support
- **3 Tools**:
  1. **Calculator Tool**: Performs mathematical calculations
  2. **Web Search Tool**: Simulated web search (for demonstration)
  3. **File Operations Tool**: Read, write, list, and check file existence
- **FastAPI Server**: RESTful API for agent interactions
- **OpenAI-Compatible**: Connects to vLLM server via OpenAI-compatible API

## Setup

1. **Install dependencies**:
```bash
cd agent_server
pip install -r requirements.txt
```

2. **Set environment variables** (optional):
```bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="Qwen2.5-32B-Instruct-AWQ"
export PORT=8001
```

3. **Start the agent server**:
```bash
python main.py
```

Or using uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

## API Endpoints

### `GET /health`
Health check endpoint. Returns server status and LLM connection info.

### `POST /chat`
Chat with the agent.

**Request:**
```json
{
  "message": "Calculate 10 * 5",
  "conversation_id": "optional-conv-id"
}
```

**Response:**
```json
{
  "response": "The result is 50",
  "conversation_id": "optional-conv-id"
}
```

### `GET /`
Root endpoint with API information.

## Testing

Run unit tests:

```bash
# Test connection to LLM server
pytest tests/test_connection.py -v

# Test agent requests
pytest tests/test_agent.py -v

# Run all tests
pytest tests/ -v
```

**Note**: Make sure both the LLM server (port 8000) and agent server (port 8001) are running before running tests.

## Architecture

```
Client → Agent Server (FastAPI) → LlamaIndex Agent → Tools
                              ↓
                         LLM Server (vLLM)
```

The agent server:
1. Receives requests from clients
2. Uses LlamaIndex agent to process requests
3. Agent can use tools (calculator, web search, file ops) when needed
4. Agent communicates with LLM server via OpenAI-compatible API
5. Returns responses to clients

## Tools

### Calculator Tool
- Performs mathematical calculations
- Supports basic operations and some math functions
- Example: "Calculate 2 + 2"

### Web Search Tool
- Simulated web search (for demonstration)
- Returns predefined results for common queries
- Example: "Search for Python programming"

### File Operations Tool
- Read, write, list files
- Check file existence
- Operates in `/tmp` directory for safety
- Example: "Read file /tmp/test.txt"
