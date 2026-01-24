from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llm_interface import LLMInterface
from typing import Generator

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/generate/nonstream")
def generate_nonstream(req: PromptRequest):
    engine = LLMInterface()
    
    def token_stream() -> Generator[str, None, None]:
        for char in engine.stream(req.prompt):
            yield char

    return StreamingResponse(token_stream(), media_type="text/plain")


@router.post("/generate/stream")
async def generate_stream(req: PromptRequest):
    engine = LLMInterface()
    async def event_generator():
        async for token in engine.stream(req.prompt):
            yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
    )

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