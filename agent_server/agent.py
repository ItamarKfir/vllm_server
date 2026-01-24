from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole, LLMMetadata, CompletionResponse, ChatResponse, ChatResponseAsyncGen
from llama_index.core.llms.llm import LLM
from llama_index.core.bridge.pydantic import Field
from typing import Sequence
import asyncio
import httpx

from tools import get_function_tools


class CustomOpenAILLM(LLM):
    base_url: str = Field(default="http://localhost:8000")
    model: str = Field(default="Qwen2.5-32B-Instruct-AWQ")
    api_key: str = Field(default="not-needed")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen2.5-32B-Instruct-AWQ", api_key: str = "not-needed", temperature: float = 0.7, max_tokens: int = 2000, **kwargs):
        super().__init__(base_url=base_url, model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self._api_base = base_url.rstrip('/v1').rstrip('/')
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model, is_chat_model=True, is_function_calling_model=True)
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        raise NotImplementedError("Use astream_chat() instead")
    
    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Use astream_chat() instead")
    
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        raise NotImplementedError("Use astream_chat() instead")
    
    async def astream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Use astream_chat() instead")
    
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt_parts = []
        for msg in messages:
            role = msg.role.value
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
    
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> CompletionResponse:
        raise NotImplementedError("Use astream_chat() instead")
    
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs):
        raise NotImplementedError("Use astream_chat() instead")
    
    async def achat(self, messages: Sequence[ChatMessage], **kwargs) -> CompletionResponse:
        prompt = self._messages_to_prompt(messages)
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self._api_base}/generate/nonstream", json={"prompt": prompt}) as response:
                response.raise_for_status()
                content = ""
                async for chunk in response.aiter_text():
                    if chunk:
                        content += chunk
        return CompletionResponse(text=content, raw=content)
    
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponseAsyncGen:
        prompt = self._messages_to_prompt(messages)
        async def gen() -> ChatResponseAsyncGen:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{self._api_base}/generate/stream", json={"prompt": prompt}) as response:
                    response.raise_for_status()
                    content = ""
                    async for chunk in response.aiter_text():
                        if chunk:
                            delta = chunk
                            content += delta
                            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content), delta=delta, raw=chunk)
        return gen()


_tool_usage_tracker = {"tools_used": []}

def create_llm(base_url: str = "http://localhost:8000", model: str = "Qwen2.5-32B-Instruct-AWQ"):
    if base_url.endswith('/v1'):
        base_url = base_url[:-3]
    return CustomOpenAILLM(base_url=base_url, model=model, api_key="not-needed", temperature=0.7, max_tokens=2000)

def create_agent(base_url: str = "http://localhost:8000", model: str = "Qwen2.5-32B-Instruct-AWQ"):
    _tool_usage_tracker["tools_used"] = []
    llm = create_llm(base_url=base_url, model=model)
    function_tools = get_function_tools(_tool_usage_tracker)
    agent = ReActAgent(tools=function_tools, llm=llm, verbose=False, streaming=True, system_prompt="You are a helpful AI assistant with access to tools. Always respond in English. Use the tools when needed to help answer questions accurately. When you provide your final answer, make it clear and concise in English.")
    return agent


async def chat_with_agent_stream(agent: ReActAgent, message: str):
    try:
        _tool_usage_tracker["tools_used"] = []
        workflow_handler = agent.run(user_msg=message)
        answer_started = False
        accumulated = ""
        
        if hasattr(workflow_handler, 'stream_events'):
            async for event in workflow_handler.stream_events():
                delta = getattr(event, 'delta', None) or ""
                if delta:
                    accumulated += delta
                    if "Answer:" in accumulated and not answer_started:
                        answer_started = True
                        pos = accumulated.find("Answer:") + len("Answer:")
                        initial = accumulated[pos:].strip()
                        if initial and not any(initial.strip().startswith(m) for m in ["Thought:", "Action:", "Action Input:", "Observation:"]):
                            yield initial
                            accumulated = ""
                    elif answer_started:
                        if not any(delta.strip().startswith(m) for m in ["Thought:", "Action:", "Action Input:", "Observation:"]):
                            yield delta
        
        if hasattr(workflow_handler, 'done'):
            while not workflow_handler.done():
                await asyncio.sleep(0.1)
        elif hasattr(workflow_handler, '__await__'):
            await workflow_handler
        
        tools_used = _tool_usage_tracker["tools_used"]
        if tools_used:
            yield f"\n\n[Used tools: {', '.join(tools_used)}]"
    except Exception as e:
        import traceback
        yield f"Error: {str(e)}\n{traceback.format_exc()}"
