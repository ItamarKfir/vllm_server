from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_interface import LLMInterface
from metrics import start_request, record_result, finish_request
from logger import log_with_timestamp

router = APIRouter()


class PromptRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    text: str


@router.post("/generate/nonstream", response_model=GenerateResponse)
async def generate_nonstream(req: PromptRequest):
    endpoint = "nonstream"
    request_size = len(req.prompt.encode("utf-8"))
    start_time = start_request(endpoint, request_size)
    log_with_timestamp(f"Request received: {endpoint} | prompt_len={len(req.prompt)}")

    try:
        engine = LLMInterface()
        text = await engine.generate(req.prompt)
        token_approx = max(1, len(text.split()))
        log_with_timestamp(f"Generation completed: {endpoint} | tokens={token_approx} | response_len={len(text)}")
        record_result(endpoint, "success", token_count=token_approx)
        return GenerateResponse(text=text)
    except Exception as e:
        log_with_timestamp(f"Request error: {endpoint} | error={str(e)}", level="ERROR")
        record_result(endpoint, "error")
        raise
    finally:
        finish_request(endpoint, start_time)


@router.post("/generate/stream")
async def generate_stream(req: PromptRequest):
    endpoint = "stream"
    request_size = len(req.prompt.encode("utf-8"))
    start_time = start_request(endpoint, request_size)
    log_with_timestamp(f"Request received: {endpoint} | prompt_len={len(req.prompt)}")

    try:
        engine = LLMInterface()
        token_count = 0

        async def event_generator():
            nonlocal token_count
            try:
                async for token in engine.stream(req.prompt):
                    token_count += 1
                    yield token
                log_with_timestamp(f"Stream completed: {endpoint} | tokens={token_count}")
                record_result(endpoint, "success", token_count=token_count)
            except Exception as e:
                log_with_timestamp(f"Stream error: {endpoint} | error={str(e)}", level="ERROR")
                record_result(endpoint, "error")
                raise

        return StreamingResponse(event_generator(), media_type="text/plain")
    except Exception as e:
        log_with_timestamp(f"Request error: {endpoint} | error={str(e)}", level="ERROR")
        record_result(endpoint, "error")
        raise
    finally:
        finish_request(endpoint, start_time)

@router.get("/health")
def health_check():
    return {"status": "ok"} 

@router.get("/get_llm_config")
def get_llm_config():
    from config_manager import setting
    return setting.llm.dict()

@router.get("/get_server_config")
def get_server_config():
    from config_manager import setting
    return setting.server.dict()

@router.post("/save_llm_config")
def save_llm_config(config: dict):
    from config_manager import setting, LLMConfig
    llm_config = LLMConfig(**config)
    setting._save_llm(llm_config)
    LLMInterface.reset()
    LLMInterface()
    return {"status": "config saved"}

@router.post("/save_server_config")
def save_server_config(config: dict):
    from config_manager import setting, ServerConfig
    server_config = ServerConfig(**config)
    setting._save_server(server_config)
    return {"status": "config saved"}