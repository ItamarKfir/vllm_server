"""
Example client for the Agent Server
"""
import requests
import json

AGENT_SERVER_URL = "http://localhost:8001"


def chat_with_agent(message: str, conversation_id: str = None):
    """Send a message to the agent server"""
    response = requests.post(
        f"{AGENT_SERVER_URL}/chat",
        json={
            "message": message,
            "conversation_id": conversation_id
        },
        timeout=60
    )
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


def check_health():
    """Check agent server health"""
    response = requests.get(f"{AGENT_SERVER_URL}/health", timeout=5)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Server not healthy: {response.status_code}")


if __name__ == "__main__":
    print("🤖 Agent Server Client Example")
    print("=" * 50)
    
    # Check health
    try:
        health = check_health()
        print(f"✅ Server Status: {health['status']}")
        print(f"📡 LLM Server: {health['llm_server_url']}")
        print(f"🤖 Model: {health['model']}")
        print()
    except Exception as e:
        print(f"❌ Cannot connect to agent server: {e}")
        print("   Make sure the agent server is running: python main.py")
        exit(1)
    
    # Example queries
    examples = [
        "What is 15 * 23?",
        "Calculate 100 / 4 + 10",
        "Search for information about Python programming",
        "Write a test file with content 'Hello from agent' to /tmp/test_example.txt",
    ]
    
    for i, query in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {query}")
        print("-" * 50)
        try:
            response = chat_with_agent(query)
            print(f"🤖 Agent: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()
