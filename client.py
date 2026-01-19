from openai import OpenAI
import sys
import time

# Connect to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

def count_tokens_approx(text):
    """Approximate token count (rough estimate: 1 token ≈ 4 characters)"""
    return len(text) // 4

def get_multiline_input(prompt="👤 You: "):
    """Read multiple lines until empty line or EOF"""
    print(prompt, end="", flush=True)
    lines = []
    
    try:
        while True:
            line = input()
            if not line.strip():  # Empty line signals end of input
                break
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        pass
    
    return "\n".join(lines)

def chat_with_model():
    """Interactive chat loop with streaming responses"""
    print("🤖 Chat with Qwen2.5-32B (type 'quit', 'exit', or 'q' to end)")
    print("💡 Tip: Paste multi-line text and press Enter twice (empty line) to submit\n")
    
    # Initialize conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    while True:
        # Get user input (supports multi-line)
        try:
            user_input = get_multiline_input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Log request info
        input_tokens = count_tokens_approx(user_input)
        total_tokens = sum(count_tokens_approx(msg.get('content', '')) for msg in messages[1:]) + input_tokens
        print(f"\n📊 Request Info: ~{input_tokens} input tokens, ~{total_tokens} total in conversation")
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Stream the response
        print("\n🤖 Assistant: ", end="", flush=True)
        start_time = time.time()
        try:
            stream = client.chat.completions.create(
                model="Qwen2.5-32B-Instruct-AWQ",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            # Collect the full response for history
            full_response = ""
            token_count = 0
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
                    token_count += count_tokens_approx(content)
            
            elapsed = time.time() - start_time
            print(f"\n\n📊 Response: ~{token_count} tokens generated in {elapsed:.2f}s , {token_count/elapsed:.2f} token/s")
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n\n❌ Error after {elapsed:.2f}s: {e}")
            print(f"   Input was ~{input_tokens} tokens")
            # Remove the user message if there was an error
            messages.pop()

if __name__ == "__main__":
    chat_with_model()