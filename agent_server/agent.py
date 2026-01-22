"""
LlamaIndex Agent setup with OpenAI-compatible LLM
"""
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from typing import List
import os

from tools import get_all_tools


def create_llm(base_url: str = "http://localhost:8000/v1", model: str = "Qwen2.5-32B-Instruct-AWQ"):
    """Create an OpenAI-compatible LLM client for LlamaIndex"""
    return OpenAI(
        base_url=base_url,
        api_key="not-needed",  # vLLM doesn't require API key
        model=model,
        temperature=0.7,
        max_tokens=2000,
    )


def create_agent(base_url: str = "http://localhost:8000/v1", model: str = "Qwen2.5-32B-Instruct-AWQ"):
    """Create a ReAct agent with tools"""
    
    # Create LLM
    llm = create_llm(base_url=base_url, model=model)
    
    # Get all tools
    tool_instances = get_all_tools()
    
    # Convert tools to LlamaIndex FunctionTool format
    function_tools = []
    for tool in tool_instances:
        # Create a wrapper function for each tool
        def make_tool_executor(tool_instance):
            def tool_function(input_str: str) -> str:
                return tool_instance.execute(input_str)
            return tool_function
        
        tool_func = make_tool_executor(tool)
        
        # Create FunctionTool
        function_tool = FunctionTool.from_defaults(
            fn=tool_func,
            name=tool.name,
            description=tool.description,
        )
        function_tools.append(function_tool)
    
    # Create ReAct agent
    agent = ReActAgent.from_tools(
        tools=function_tools,
        llm=llm,
        verbose=True,
        system_prompt="You are a helpful AI assistant with access to tools. Use the tools when needed to help answer questions accurately.",
    )
    
    return agent


def chat_with_agent(agent: ReActAgent, message: str) -> str:
    """Send a message to the agent and get response"""
    try:
        response = agent.chat(message)
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}"
