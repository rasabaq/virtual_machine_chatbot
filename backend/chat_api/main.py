import os
import django
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from pathlib import Path

# Initialize Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# Import Django models AFTER setup
from web_app.models import Interaction
from core.agent import create_agent_executor, rag_system
import re

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG on startup
@app.on_event("startup")
async def startup_event():
    rag_system.initialize()

# --- API Models ---

class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest"

class ChatResponse(BaseModel):
    response: str
    thoughts: List[str] = []

# --- Helpers ---

def extract_thoughts(text: str) -> List[str]:
    # Extract content inside <think> tags if present in debug output (agent scratchpad)
    # Note: AgentExecutor usually returns 'output', intermediate steps needs configuration.
    # For now, we will just return the final answer. The prompt instructs to use <think> internally.
    # If we capture stdout or use 'return_intermediate_steps=True' we could get thoughts.
    # Given the prompt format: "Thought: <think>...</think>", it might leak into the reasoning log if we access it.
    # We'll leave thoughts empty for now unless we switch to parsing intermediate steps.
    return []

def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<final>([\s\S]*?)</final>", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

# --- Endpoints ---

agent_executors = {}

def get_executor(user_id: str):
    if user_id not in agent_executors:
        agent_executors[user_id] = create_agent_executor()
    return agent_executors[user_id]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        executor = get_executor(req.user_id)
        
        # Invoke agent (now just 2 API calls instead of 5+)
        raw_output = await executor.invoke(req.message)
        
        # Clean output (no longer needs <final> extraction)
        final_answer = raw_output.strip()
        
        # Save to Django DB (sync_to_async needed for async context)
        from asgiref.sync import sync_to_async
        try:
            await sync_to_async(Interaction.objects.create)(
                user_id=req.user_id,
                question=req.message,
                answer=final_answer
            )
        except Exception as e:
            print(f"Error saving to DB: {e}")

        return ChatResponse(response=final_answer, thoughts=[])

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR in /api/chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Virtual Machine Chatbot API"}
