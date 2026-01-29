"""
OpenAI-compatible API for any OpenAI-integrated client.
Use base URL http://localhost:8000/v1 with OpenAI SDK, Open WebUI, LiteLLM, etc.
Exposes /v1/chat/completions and /v1/models. No Phoenix.
"""

import json
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from config_manager import setting
from llm_interface import LLMInterface
from metrics import start_request, record_result, finish_request
from logger import log_with_timestamp

router = APIRouter(prefix="/v1", tags=["openai"])


def _content_to_text(raw: str | list[dict] | None) -> str:
    """OpenAI content: string or list of parts e.g. [{"type":"text","text":"..."}]."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        out = []
        for p in raw:
            if isinstance(p, dict) and p.get("type") == "text":
                out.append(p.get("text") or "")
            elif isinstance(p, dict) and "text" in p:
                out.append(str(p["text"]))
        return "\n".join(out).strip()
    return str(raw)


def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert OpenAI messages to a single prompt (Qwen-style chat template)."""
    parts = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = _content_to_text(m.get("content"))
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _model_id() -> str:
    return setting.llm.llm_models[0] if setting.llm.llm_models else "local"


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] | None = ""


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "default"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionsRequest):
    """OpenAI-compatible chat completions. Use with OpenAI SDK, Open WebUI, LiteLLM, etc."""
    endpoint = "openai_stream" if req.stream else "openai_nonstream"
    prompt = _messages_to_prompt([m.model_dump() for m in req.messages])
    request_size = len(prompt.encode("utf-8"))
    start_time = start_request(endpoint, request_size)
    
    model_id = _model_id()
    log_with_timestamp(f"Request received: {endpoint} | model={req.model} | stream={req.stream} | prompt_len={len(prompt)}")

    try:
        engine = LLMInterface()

        if req.stream:
            token_count = 0
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            async def sse_stream():
                nonlocal token_count
                try:
                    async for delta in engine.stream(prompt):
                        token_count += 1
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    end_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(end_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    log_with_timestamp(f"Stream completed: {endpoint} | tokens={token_count}")
                    record_result(endpoint, "success", token_count=token_count)
                except Exception as e:
                    log_with_timestamp(f"Stream error: {endpoint} | error={str(e)}", level="ERROR")
                    record_result(endpoint, "error")
                    raise

            return StreamingResponse(
                sse_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        text = await engine.generate(prompt)
        token_approx = max(1, len(text.split()))
        log_with_timestamp(f"Generation completed: {endpoint} | tokens={token_approx} | response_len={len(text)}")
        record_result(endpoint, "success", token_count=token_approx)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": token_approx,
                "total_tokens": token_approx,
            },
        }
    except Exception as e:
        log_with_timestamp(f"Request error: {endpoint} | error={str(e)}", level="ERROR")
        record_result(endpoint, "error")
        raise
    finally:
        finish_request(endpoint, start_time)


@router.get("/models")
async def list_models():
    """List models. Uses config llm_models + common IDs (gpt-3.5-turbo, gpt-4)."""
    models = list(setting.llm.llm_models) if setting.llm.llm_models else []
    if not models:
        models = ["local"]
    ids = ["gpt-3.5-turbo", "gpt-4"] + [
        m for m in models if m not in ("gpt-3.5-turbo", "gpt-4")
    ]
    ts = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": ts, "owned_by": "local"}
            for mid in ids
        ],
    }
