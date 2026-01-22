#!/bin/bash
# Start script for agent server

echo "🚀 Starting LlamaIndex Agent Server"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the agent_server directory."
    exit 1
fi

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "📦 Activating virtual environment..."
    if [ -f "../venv/bin/activate" ]; then
        source ../venv/bin/activate
    else
        echo "❌ Error: Virtual environment not found. Please create it first."
        exit 1
    fi
fi

# Check if LLM server is running
echo "📡 Checking LLM server connection..."
if curl -s http://localhost:8000/health > /dev/null 2>&1 || curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✅ LLM server is accessible"
else
    echo "⚠️  Warning: LLM server may not be running on port 8000"
    echo "   Make sure to start it first with: python server.py"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 Starting agent server on port 8001..."
echo "   API docs will be available at: http://localhost:8001/docs"
echo "   Press Ctrl+C to stop"
echo ""

python main.py
