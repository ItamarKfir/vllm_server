#!/usr/bin/env python3
"""
Interactive chat client for the Agent Server
"""
import requests
import sys

AGENT_SERVER_URL = "http://localhost:8001"


def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{AGENT_SERVER_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                return True
        return False
    except:
        return False


def chat(message: str):
    """Send a message to the agent"""
    try:
        response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={"message": message},
            timeout=120  # Longer timeout for agent processing
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to agent server. Is it running?"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    print("=" * 60)
    print("🤖 Interactive Chat with LlamaIndex Agent")
    print("=" * 60)
    print()
    
    # Check server
    if not check_server():
        print("❌ Agent server is not running or not healthy!")
        print(f"   Please start it first: python main.py")
        print(f"   Server should be at: {AGENT_SERVER_URL}")
        sys.exit(1)
    
    print("✅ Connected to agent server")
    print("💡 Type your messages below. Type 'quit', 'exit', or 'q' to end.")
    print("💡 The agent can use tools: calculator, web search, file operations")
    print("-" * 60)
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Send to agent
            print("🤖 Agent: ", end="", flush=True)
            response = chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
