from fastapi import FastAPI, Response
from contextlib import asynccontextmanager
from llm_interface import LLMInterface
from routes import router
from openai_routes import router as openai_router
from config_manager import setting
from metrics import get_metrics
from logger import log_with_timestamp
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_with_timestamp("Starting up LLM Engine...")
    LLMInterface()
    log_with_timestamp("LLM Engine initialized on startup.")
    yield
    log_with_timestamp("LLM Engine shut down.")

app = FastAPI(title="LLM Streaming API", lifespan=lifespan)
app.include_router(router)
app.include_router(openai_router)

@app.get("/metrics")
async def metrics():
    return Response(content=get_metrics(), media_type="text/plain")

if __name__ == "__main__":
    # Use server configuration from config manager
    uvicorn.run(
        app,
        host=setting.server.host,
        port=setting.server.port,
        reload=setting.server.reload,
        log_level=setting.server.log_level
    )