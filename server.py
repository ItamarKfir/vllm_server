from fastapi import FastAPI
from contextlib import asynccontextmanager
from llm_interface import LLMInterface
from routes import router
from config_manager import setting
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up LLM Engine...")
    LLMInterface()  # Initialize singleton
    print("LLM Engine initialized on startup.")
    yield
    print("LLM Engine shut down.")

app = FastAPI(title="LLM Streaming API", lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    # Use server configuration from config manager
    uvicorn.run(
        app,
        host=setting.server.host,
        port=setting.server.port,
        reload=setting.server.reload,
        log_level=setting.server.log_level
    )