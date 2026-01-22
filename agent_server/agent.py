"""
LlamaIndex Agent setup with OpenAI-compatible LLM
"""
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole, LLMMetadata, CompletionResponse, CompletionResponseGen, ChatResponse, ChatResponseAsyncGen
from llama_index.core.llms.llm import LLM
from llama_index.core.bridge.pydantic import Field
from typing import List, Optional, Sequence, AsyncIterator, Generator
import os
import asyncio
import httpx
from openai import OpenAI as OpenAIClient, AsyncOpenAI

from tools import get_all_tools


class CustomOpenAILLM(LLM):
    """Custom OpenAI-compatible LLM that bypasses model validation"""
    
    base_url: str = Field(default="http://localhost:8000", description="Base URL for the API")
    model: str = Field(default="Qwen2.5-32B-Instruct-AWQ", description="Model name")
    api_key: str = Field(default="not-needed", description="API key")
    temperature: float = Field(default=0.7, description="Temperature")
    max_tokens: int = Field(default=2000, description="Max tokens")
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "Qwen2.5-32B-Instruct-AWQ",
        api_key: str = "not-needed",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Store the base URL without /v1 since we're using custom endpoints
        self._api_base = base_url.rstrip('/v1').rstrip('/')
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=True,
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # Not used for chat models, but required by base class
        raise NotImplementedError("Use chat() instead")
    
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        # Not used for chat models, but required by base class
        raise NotImplementedError("Use stream_chat() instead")
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        # Not used for chat models, but required by base class
        raise NotImplementedError("Use achat() instead")
    
    async def astream_complete(self, prompt: str, **kwargs) -> AsyncIterator[CompletionResponse]:
        # Not used for chat models, but required by base class
        raise NotImplementedError("Use astream_chat() instead")
    
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> Generator[CompletionResponse, None, None]:
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Call custom API endpoint
        import requests
        response = requests.post(
            f"{self._api_base}/generate/stream",
            json={"prompt": prompt},
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                yield CompletionResponse(
                    text=chunk,
                    raw=chunk,
                )
    
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Convert chat messages to a prompt string for the model"""
        # For Qwen2.5 models, use the chat template format
        # Format: <|im_start|>role\ncontent<|im_end|>\n
        prompt_parts = []
        for msg in messages:
            role = msg.role.value
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
        
        # Add assistant start token for the response
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> CompletionResponse:
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Call custom API endpoint
        import requests
        response = requests.post(
            f"{self._api_base}/generate/nonstream",
            json={"prompt": prompt},
            timeout=60
        )
        response.raise_for_status()
        
        # The response is streaming text, read it all
        content = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                content += chunk
        
        return CompletionResponse(
            text=content,
            raw=content,
        )
    
    async def achat(self, messages: Sequence[ChatMessage], **kwargs) -> CompletionResponse:
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Call custom API endpoint asynchronously
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self._api_base}/generate/nonstream",
                json={"prompt": prompt}
            ) as response:
                response.raise_for_status()
                content = ""
                async for chunk in response.aiter_text():
                    if chunk:
                        content += chunk
        
        return CompletionResponse(
            text=content,
            raw=content,
        )
    
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> ChatResponseAsyncGen:
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Create an async generator function and return it
        # This matches the pattern used by OpenAI LLM
        async def gen() -> ChatResponseAsyncGen:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._api_base}/generate/stream",
                    json={"prompt": prompt}
                ) as response:
                    response.raise_for_status()
                    content = ""
                    async for chunk in response.aiter_text():
                        if chunk:
                            delta = chunk
                            content += delta
                            # Yield ChatResponse objects
                            yield ChatResponse(
                                message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
                                delta=delta,
                                raw=chunk,
                            )
        
        # Return the async generator (not yield from it)
        return gen()


def create_llm(base_url: str = "http://localhost:8000", model: str = "Qwen2.5-32B-Instruct-AWQ"):
    """Create a custom LLM client for LlamaIndex that uses the custom API"""
    # Remove /v1 if present since we're using custom endpoints
    if base_url.endswith('/v1'):
        base_url = base_url[:-3]
    return CustomOpenAILLM(
        base_url=base_url,
        model=model,
        api_key="not-needed",
        temperature=0.7,
        max_tokens=2000,
    )


# Global variable to track tool usage during agent execution
_tool_usage_tracker = {"tools_used": []}

def create_agent(base_url: str = "http://localhost:8000", model: str = "Qwen2.5-32B-Instruct-AWQ"):
    """Create a ReAct agent with tools"""
    
    # Reset tool usage tracker
    _tool_usage_tracker["tools_used"] = []
    
    # Create LLM
    llm = create_llm(base_url=base_url, model=model)
    
    # Get all tools
    tool_instances = get_all_tools()
    
    # Convert tools to LlamaIndex FunctionTool format
    function_tools = []
    for tool in tool_instances:
        # Create a wrapper function for each tool (using lambda with default arg to capture tool)
        def create_tool_function(tool_instance):
            def tool_function(*args, **kwargs) -> str:
                # Track tool usage
                tool_name = tool_instance.name
                if tool_name not in _tool_usage_tracker["tools_used"]:
                    _tool_usage_tracker["tools_used"].append(tool_name)
                
                # Print to console for debugging (with flush)
                import sys
                print(f"\n{'='*60}", flush=True)
                print(f"[Agent] Tool '{tool_name}' CALLED!", flush=True)
                print(f"[Agent] Args: {args}", flush=True)
                print(f"[Agent] Kwargs: {kwargs}", flush=True)
                sys.stdout.flush()
                
                # Determine input - FunctionTool passes kwargs directly as the command structure
                # If kwargs are present, they ARE the command (e.g., {'action': 'search', 'filename': 'file.json'})
                # If args are present, it might be a single string argument
                input_data = None
                
                if kwargs:
                    # Kwargs might have the command nested in 'args' key
                    import json
                    # Check if there's an 'args' key (common in LlamaIndex tool calls)
                    if 'args' in kwargs:
                        # Extract the actual command from 'args'
                        args_value = kwargs['args']
                        # Handle AttributedDict or regular dict
                        if hasattr(args_value, '__dict__') or hasattr(args_value, 'keys'):
                            # Convert to regular dict if needed
                            if hasattr(args_value, 'keys'):
                                cmd_dict = dict(args_value)
                            else:
                                cmd_dict = dict(args_value) if isinstance(args_value, dict) else vars(args_value)
                        else:
                            cmd_dict = args_value
                        input_data = json.dumps(cmd_dict)
                        print(f"[Agent] Extracted command from 'args': {cmd_dict}", flush=True)
                    else:
                        # Kwargs are the command structure directly
                        input_data = json.dumps(kwargs)
                        print(f"[Agent] Using kwargs as command: {kwargs}", flush=True)
                elif args:
                    # Single positional argument
                    input_data = args[0] if isinstance(args[0], str) else str(args[0])
                    print(f"[Agent] Using args[0] as input: {input_data}", flush=True)
                else:
                    input_data = ""
                    print(f"[Agent] No args or kwargs, using empty string", flush=True)
                
                print(f"[Agent] Final input_data: {input_data}", flush=True)
                print(f"[Agent] Input type: {type(input_data)}", flush=True)
                sys.stdout.flush()
                
                # Execute the tool
                try:
                    result = tool_instance.execute(input_data)
                    print(f"[Agent] Tool '{tool_name}' returned result (length: {len(result)} chars)", flush=True)
                    print(f"[Agent] Result preview: {result[:200]}...", flush=True)
                    sys.stdout.flush()
                    return result
                except Exception as e:
                    import traceback
                    error_msg = f"Error in tool '{tool_name}': {str(e)}\n{traceback.format_exc()}"
                    print(f"[Agent] {error_msg}", flush=True)
                    sys.stdout.flush()
                    return error_msg
            return tool_function
        
        tool_func = create_tool_function(tool)
        tool_func.__name__ = tool.name  # Set function name for better debugging
        
        # Create FunctionTool
        function_tool = FunctionTool.from_defaults(
            fn=tool_func,
            name=tool.name,
            description=tool.description,
        )
        function_tools.append(function_tool)
    
    # Create ReAct agent (using constructor for LlamaIndex 0.14.x)
    agent = ReActAgent(
        tools=function_tools,
        llm=llm,
        verbose=True,
        system_prompt="You are a helpful AI assistant with access to tools. Use the tools when needed to help answer questions accurately.",
    )
    
    return agent


async def chat_with_agent(agent: ReActAgent, message: str) -> str:
    """Send a message to the agent and get response"""
    try:
        # Reset tool usage tracker for this conversation
        _tool_usage_tracker["tools_used"] = []
        
        # Track tools used during execution
        tools_used = []
        
        # In LlamaIndex 0.14.x, ReActAgent is a workflow agent
        # The run() method returns a workflow handler (WorkflowHandler)
        workflow_handler = agent.run(user_msg=message)
        
        # The WorkflowHandler is a Future-like object
        # Wait for it to complete, then get the result
        # Try multiple approaches to execute and get result
        
        # Approach 1: Check if handler has done() method and wait for it
        if hasattr(workflow_handler, 'done'):
            # Wait for the workflow to complete
            while not workflow_handler.done():
                await asyncio.sleep(0.1)
            result = workflow_handler.result()
        # Approach 2: Try to await it directly
        elif hasattr(workflow_handler, '__await__'):
            await workflow_handler
            result = workflow_handler.result() if hasattr(workflow_handler, 'result') else None
        # Approach 3: Use asyncio.gather to execute it
        else:
            try:
                # Try to gather/execute the handler
                await asyncio.gather(workflow_handler)
                result = workflow_handler.result() if hasattr(workflow_handler, 'result') else None
            except (TypeError, ValueError):
                # If that fails, try creating a task
                try:
                    task = asyncio.create_task(workflow_handler)
                    await task
                    result = workflow_handler.result() if hasattr(workflow_handler, 'result') else None
                except:
                    # Last resort: try to get result directly (might fail)
                    result = workflow_handler.result() if hasattr(workflow_handler, 'result') else workflow_handler
        
        # Extract the response from the result
        # The result structure may vary, try different access patterns
        output = None
        
        # Debug: Check what type of object we got
        result_type = type(result).__name__
        
        # Check for tool calls in the result
        # The result might be an AgentOutput with tool_calls
        if hasattr(result, 'tool_calls') and result.tool_calls:
            tools_used = [tool_call.tool_name for tool_call in result.tool_calls if hasattr(tool_call, 'tool_name')]
        
        # Try different ways to access the result
        # The result might be a StopEvent or AgentOutput
        if hasattr(result, 'result'):
            output = result.result
            # Also check for tool_calls in nested result
            if hasattr(output, 'tool_calls') and output.tool_calls and not tools_used:
                tools_used = [tool_call.tool_name for tool_call in output.tool_calls if hasattr(tool_call, 'tool_name')]
        elif hasattr(result, 'response'):
            output = result.response
        elif hasattr(result, 'message'):
            output = result.message
        elif hasattr(result, 'output'):
            output = result.output
        else:
            output = result
        
        # Extract the text content
        if output is None:
            # Try to get more info about the result structure
            result_attrs = [attr for attr in dir(result) if not attr.startswith('_')]
            return f"Error: No response from agent. Result type: {result_type}, Attributes: {result_attrs[:10]}"
        
        # Try to get text content from various possible structures
        text = None
        
        # Check if output is a message object
        if hasattr(output, 'content'):
            text = output.content
        elif hasattr(output, 'text'):
            text = output.text
        elif hasattr(output, 'response'):
            if hasattr(output.response, 'content'):
                text = output.response.content
            elif hasattr(output.response, 'text'):
                text = output.response.text
            else:
                text = str(output.response)
        elif isinstance(output, str):
            text = output
        elif hasattr(output, '__str__'):
            text = str(output)
        else:
            text = str(output)
        
        # Build the final response
        response_text = str(text) if text else f"Error: Empty response. Output type: {type(output).__name__}, Output: {output}"
        
        # Get tools used from tracker (set by tool wrapper functions)
        tools_used_from_tracker = _tool_usage_tracker["tools_used"]
        
        # Also check for tool_calls in the result object
        if not tools_used_from_tracker and tools_used:
            tools_used_from_tracker = tools_used
        
        # Append tool usage information if tools were used
        if tools_used_from_tracker:
            tools_list = ", ".join(tools_used_from_tracker)
            response_text += f"\n\n[Used tools: {tools_list}]"
        
        return response_text
        
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"
