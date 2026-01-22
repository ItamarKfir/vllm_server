"""
FastAPI server for LlamaIndex Agent
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import os

from agent import create_agent, chat_with_agent

# Global agent instance
agent_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global agent_instance
    print("🚀 Starting LlamaIndex Agent Server...")
    
    # Get LLM server URL from environment or use default
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    llm_model = os.getenv("LLM_MODEL", "Qwen2.5-32B-Instruct-AWQ")
    
    print(f"📡 Connecting to LLM server: {llm_base_url}")
    print(f"🤖 Using model: {llm_model}")
    
    try:
        agent_instance = create_agent(base_url=llm_base_url, model=llm_model)
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        raise
    
    yield
    
    print("🛑 Shutting down Agent Server...")


app = FastAPI(
    title="LlamaIndex Agent API",
    description="FastAPI server with LlamaIndex agent that uses tools to interact with LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    llm_server_url: str
    model: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    llm_model = os.getenv("LLM_MODEL", "Qwen2.5-32B-Instruct-AWQ")
    
    return HealthResponse(
        status="healthy" if agent_instance is not None else "not_initialized",
        llm_server_url=llm_base_url,
        model=llm_model,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the agent"""
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        response = chat_with_agent(agent_instance, request.message)
        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LlamaIndex Agent API",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
