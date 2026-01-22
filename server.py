from fastapi import FastAPI
from contextlib import asynccontextmanager
from llm_interface import LLMInterface
from routes import router
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
    uvicorn.run(app, host="0.0.0.0", port=8000)