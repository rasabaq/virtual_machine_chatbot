import os
import logging
import time
import django
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# Import Django models AFTER setup
from web_app.models import User, Conversation, Message, Interaction
from core.agent import create_agent_executor, rag_system
from asgiref.sync import sync_to_async
from passlib.context import CryptContext
import jwt

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize RAG on startup
@app.on_event("startup")
async def startup_event():
    logger.info("[STARTUP] Initializing RAG system...")
    rag_system.initialize()
    logger.info("[STARTUP] RAG system initialized successfully")


# --- Helper Functions ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await sync_to_async(User.objects.get)(id=user_id)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except User.DoesNotExist:
        raise HTTPException(status_code=401, detail="User not found")


# --- Pydantic Models ---

# Auth models
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user_id: int
    email: str
    name: str
    token: str


class UserResponse(BaseModel):
    user_id: int
    email: str
    name: str


# Conversation models
class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetailResponse(BaseModel):
    id: int
    title: str
    messages: List[MessageResponse]


# Chat models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: int
    message_id: int
    thoughts: List[str] = []


# --- Auth Endpoints ---

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    logger.info(f"[AUTH] Registration attempt for email: {req.email}")
    # Check if user exists
    exists = await sync_to_async(User.objects.filter(email=req.email).exists)()
    if exists:
        logger.warning(f"[AUTH] Registration failed - email already exists: {req.email}")
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    hashed_password = hash_password(req.password)
    user = await sync_to_async(User.objects.create)(
        email=req.email,
        name=req.name,
        password=hashed_password
    )

    token = create_access_token(user.id)
    logger.info(f"[AUTH] User registered successfully: {user.id} ({user.email})")
    return AuthResponse(
        user_id=user.id,
        email=user.email,
        name=user.name,
        token=token
    )


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    logger.info(f"[AUTH] Login attempt for email: {req.email}")
    try:
        user = await sync_to_async(User.objects.get)(email=req.email)
    except User.DoesNotExist:
        logger.warning(f"[AUTH] Login failed - user not found: {req.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(req.password, user.password):
        logger.warning(f"[AUTH] Login failed - invalid password for: {req.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id)
    logger.info(f"[AUTH] User logged in successfully: {user.id} ({user.email})")
    return AuthResponse(
        user_id=user.id,
        email=user.email,
        name=user.name,
        token=token
    )


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        user_id=current_user.id,
        email=current_user.email,
        name=current_user.name
    )


# --- Conversation Endpoints ---

@app.get("/api/conversations", response_model=List[ConversationResponse])
async def list_conversations(current_user: User = Depends(get_current_user)):
    conversations = await sync_to_async(list)(
        Conversation.objects.filter(user=current_user).order_by('-updated_at')
    )
    return [
        ConversationResponse(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at
        )
        for c in conversations
    ]


@app.post("/api/conversations", response_model=ConversationResponse)
async def create_conversation(
    req: ConversationCreate,
    current_user: User = Depends(get_current_user)
):
    conversation = await sync_to_async(Conversation.objects.create)(
        user=current_user,
        title=req.title
    )
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user)
):
    try:
        conversation = await sync_to_async(Conversation.objects.get)(
            id=conversation_id,
            user=current_user
        )
    except Conversation.DoesNotExist:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = await sync_to_async(list)(
        conversation.messages.all().order_by('created_at')
    )

    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at
            )
            for m in messages
        ]
    )


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user)
):
    try:
        conversation = await sync_to_async(Conversation.objects.get)(
            id=conversation_id,
            user=current_user
        )
    except Conversation.DoesNotExist:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await sync_to_async(conversation.delete)()
    return {"success": True}


# --- Chat Endpoint ---

agent_executors = {}


def get_executor(user_id: int):
    if user_id not in agent_executors:
        agent_executors[user_id] = create_agent_executor()
    return agent_executors[user_id]


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    start_time = time.time()
    logger.info(f"[CHAT] User {current_user.id} ({current_user.email}) - New request")
    logger.info(f"[CHAT] Message: {req.message[:100]}{'...' if len(req.message) > 100 else ''}")

    try:
        # Get or create conversation
        if req.conversation_id:
            logger.info(f"[CHAT] Using existing conversation: {req.conversation_id}")
            try:
                conversation = await sync_to_async(Conversation.objects.get)(
                    id=req.conversation_id,
                    user=current_user
                )
            except Conversation.DoesNotExist:
                logger.warning(f"[CHAT] Conversation {req.conversation_id} not found for user {current_user.id}")
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            # Create new conversation with title from first message
            title = req.message[:50] + "..." if len(req.message) > 50 else req.message
            conversation = await sync_to_async(Conversation.objects.create)(
                user=current_user,
                title=title
            )
            logger.info(f"[CHAT] Created new conversation: {conversation.id}")

        # Save user message
        user_message = await sync_to_async(Message.objects.create)(
            conversation=conversation,
            role='user',
            content=req.message
        )
        logger.info(f"[CHAT] Saved user message: {user_message.id}")

        # Get agent response
        logger.info(f"[CHAT] Invoking agent for user {current_user.id}")
        agent_start = time.time()
        executor = get_executor(current_user.id)
        raw_output = await executor.invoke(req.message)
        final_answer = raw_output.strip()
        agent_duration = time.time() - agent_start
        logger.info(f"[CHAT] Agent response received in {agent_duration:.2f}s - Length: {len(final_answer)} chars")

        # Save assistant message
        assistant_message = await sync_to_async(Message.objects.create)(
            conversation=conversation,
            role='assistant',
            content=final_answer
        )
        logger.info(f"[CHAT] Saved assistant message: {assistant_message.id}")

        # Update conversation timestamp
        conversation.updated_at = datetime.now()
        await sync_to_async(conversation.save)()

        total_duration = time.time() - start_time
        logger.info(f"[CHAT] Request completed in {total_duration:.2f}s - Conversation: {conversation.id}")

        return ChatResponse(
            response=final_answer,
            conversation_id=conversation.id,
            message_id=assistant_message.id,
            thoughts=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[CHAT] Critical error processing request for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"status": "ok", "service": "Virtual Machine Chatbot API"}
