#!/bin/bash
# Test runner script for agent server

echo "🧪 Running Agent Server Tests"
echo "=============================="
echo ""

# Check if LLM server is running
echo "📡 Checking LLM server connection..."
if curl -s http://localhost:8000/health > /dev/null 2>&1 || curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✅ LLM server is accessible"
else
    echo "⚠️  LLM server may not be running on port 8000"
    echo "   Start it with: python server.py"
fi

echo ""
echo "📡 Checking Agent server connection..."
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Agent server is running"
else
    echo "⚠️  Agent server is not running on port 8001"
    echo "   Start it with: python main.py"
    echo ""
    read -p "Start agent server now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting agent server in background..."
        python main.py &
        AGENT_PID=$!
        echo "Agent server started with PID: $AGENT_PID"
        echo "Waiting for server to be ready..."
        sleep 5
    fi
fi

echo ""
echo "🧪 Running tests..."
echo ""

# Run connection tests
echo "1️⃣  Running connection tests..."
pytest tests/test_connection.py -v

echo ""
echo "2️⃣  Running agent functionality tests..."
pytest tests/test_agent.py -v -s

echo ""
echo "✅ Tests completed!"
