
from fastapi import FastAPI, HTTPException, Depends, status, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from Phase.execute_operation import execute_site_operation
from Phase.phase_0 import execute_phase_0
from Phase.phase_1 import execute_phase_1
from Phase.phase_2 import execute_phase_2
from LLM.initialize_llm import get_gemini_client
from services.User_Service import UserManager


from db.db_services import save_conversation_to_db,get_last_conversations_from_db,get_all_session_ids,clear_conversation_from_db,get_session_intent,store_session_intent,get_last_session,get_complete_conversation_from_db,can_proceed

import logging
import json
# Import from our organized files
from services.Authentication_Service import AuthenticationManager

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class AuthRequest(BaseModel):
    refresh_token: str = Field(..., description="JWT refresh token")

class OllamaConfig(BaseModel):
    model_name: str = Field(default="llama3.1:8b", description="Ollama model name")
    temperature: float = Field(default=0.7, description="Generation temperature")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User ID")
    session_intent: Optional[str] = Field(None, description="Detected intent for the session")
    email:str=Field(...,description="email of the user")
    refresh_token: Optional[str] = Field(None, description="Refresh token for authentication")


class ChatResponse(BaseModel):
    message: str
    status: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    session_intent: Optional[str] = None

# Global variables
ollama_site_manager = None
ollama_config = OllamaConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting PulsePro Site Management API with Ollama Integration...")
    print(f"Ollama model: {ollama_config.model_name}")
    yield
    # Shutdown
    print("Shutting down PulsePro Site Management API...")

app = FastAPI(
    title="PulsePro Site Management API with Ollama",
    description="AI-powered conversational site management system using Ollama llama3.1:8b",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Global storage for chat sessions (in production, use Redis or database)
chat_sessions: Dict[str, List[Dict]] = {}



load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple session storage (keeping for backward compatibility)
chat_sessions = {}


@app.post("/chat/onboarding", response_model=ChatResponse)
async def chat_with_agent_onboarding(chat_request: ChatRequest,current_user: dict = Depends(UserManager.get_current_user)):
    """onboarding chat with agent"""

    auth = AuthenticationManager()
    print("refresh: ", chat_request.refresh_token)
    refresh= chat_request.refresh_token
    auth.set_refresh_token(refresh_token=refresh)

    print(f"Authenticated user: {current_user['email']}")
    valid_intents=["CREATE_SITE","CREATE_USER","CREATE_TEMPLATE"]

    print("inside chat onboarding")
    session_id = chat_request.session_id or f"Session_{str(uuid.uuid4())}"
    intent = get_session_intent(session_id) or "UNKNOWN_1"
    if intent not in valid_intents:
        intent="UNKNOWN_1"
    user_message = chat_request.message.strip()
    email=chat_request.email.strip()
    print(f"Session ID: {session_id}, Intent initial: {intent}")

    try:
        # # Accept refresh token via Authorization header (Bearer <token>)
        # auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
        # if auth_header and auth_header.lower().startswith("bearer "):
        #     token = auth_header.split(" ", 1)[1].strip()
        #     os.environ["refresh"] = token  # Used by AuthenticationManager

        client = get_gemini_client(temperature=0.3)
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {"conversation": []}
        
        save_conversation_to_db(session_id, "user", user_message,email=email, intent=intent)

        cancel_triggers = ["cancel", "stop", "exit", "abort", "halt", "quit", "terminate", "end"]
        if any(trigger in user_message.lower() for trigger in cancel_triggers):
            clear_conversation_from_db(session_id)
            return ChatResponse(
                message="""Operation cancelled. No action taken. Now you can start with new operation.\n 
                • **Site Creation** - Set up and configure your PulsePro site
                • **User Account Setup** - Create profiles and manage permissions  
                • **Checklist Creation** - Build industry standard checklists
                 """,
                status="cancelled",
                session_id=session_id,
                context={"phase": "cancelled"},
                data={}
            )


        if intent == 'UNKNOWN_1':
            intent = await execute_phase_0(session_id, user_message)
            if(intent in valid_intents):
                store_session_intent(session_id, intent)
            else:
                intent='UNKNOWN_1'    
            print(f"Intent after phase 0: {intent}")
        print("can proceed in main : ", can_proceed(session_id=session_id))
        execution_triggers = ["proceed", "execute", "go", "do it", "yes proceed", "execute now"]
        if user_message.lower().strip() in execution_triggers and can_proceed(session_id=session_id):
            return await execute_phase_2(session_id, intent,email=email,auth_manager=auth)
        
        print("Proceeding to Phase 1 chat...")
        return await execute_phase_1(session_id, user_message, client,intent,email=email,onboarding=True,auth=auth)
                
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            message=f"Sorry, something went wrong: {str(e)}",
            status="error",
            session_id=session_id,
            context={"error": str(e)},
            data={}
        )





@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatRequest,current_user: dict = Depends(UserManager.get_current_user)):
    """Two-phase chat agent: Phase 1 (Chat) → Phase 2 (Execute)"""


    auth = AuthenticationManager()
    print("refresh: ", chat_request.refresh_token)
    refresh= chat_request.refresh_token
    auth.set_refresh_token(refresh_token=refresh)
    print(f"Authenticated user: {current_user['email']}")


    print("inside chat")
    session_id = chat_request.session_id or str(uuid.uuid4())
    intent = get_session_intent(session_id) or "UNKNOWN"
    user_message = chat_request.message.strip()
    email=chat_request.email.strip()
    print(f"Session ID: {session_id}, Intent initial: {intent}")
    
    try:
        # # Accept refresh token via Authorization header (Bearer <token>)
        # auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
        # if auth_header and auth_header.lower().startswith("bearer "):
        #     token = auth_header.split(" ", 1)[1].strip()
        #     os.environ["refresh"] = token  # Used by AuthenticationManager

        # client = get_ollama_client()

        client = get_gemini_client()
        # model = client.GenerativeModel("gemini-2.5-pro")

        
        # Initialize session (MongoDB-based)
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {"conversation": []}
        
        # Add user message to conversation and save to MongoDB
        save_conversation_to_db(session_id, "user", user_message,email=email, intent=intent)

        # Check if user wants to execute (Phase 2)
        cancel_triggers = ["cancel", "stop", "exit", "abort", "halt", "quit", "terminate", "end"]
        if any(trigger in user_message.lower() for trigger in cancel_triggers):
            clear_conversation_from_db(session_id)
            return ChatResponse(
                message="Operation cancelled. No action taken. Now you can start with new operation",
                status="cancelled",
                session_id=session_id,
                context={"phase": "cancelled"},
                data={}
            )

        if intent == 'UNKNOWN':
            intent = await execute_phase_0(session_id, user_message)
            store_session_intent(session_id, intent)
            print(f"Intent after phase 0: {intent}")
        
        import inspect
        print(inspect.signature(execute_phase_2))

        
        execution_triggers = ["proceed", "execute", "go", "do it", "yes proceed", "execute now"]
        if user_message.lower().strip() in execution_triggers:
            return await execute_phase_2(session_id=session_id,intent=intent, email=current_user['email'],auth_manager=auth)
        
        # Phase 1: Continue conversation
        print("Proceeding to Phase 1 chat...")
        return await execute_phase_1(session_id, user_message, client,intent,email=email,onboarding=False,auth=auth)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            message=f"Sorry, something went wrong: {str(e)}",
            status="error",
            session_id=session_id,
            context={"error": str(e)},
            data={}
        )





@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get pending conversation history for a last session from MongoDB"""
    try:
        messages = get_last_conversations_from_db(session_id)
        return {
            "session_id": session_id,
            "conversation": [
                {
                    "role": msg["role"],
                    "message": msg["message"], 
                    "timestamp": msg["timestamp"].isoformat() if "timestamp" in msg else str(msg["_id"].generation_time)
                } for msg in messages
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")
    

@app.get("/chat/complete_history/{session_id}")
async def get_complete_chat_history(session_id: str):
    """Get conversation history for a session from MongoDB"""
    try:
        messages = get_complete_conversation_from_db(session_id)
        return {
            "session_id": session_id,
            "conversation": [
                {
                    "role": msg["role"],
                    "message": msg["message"],
                    # convert either timestamp or ObjectId to ISO
                    "timestamp": msg.get("timestamp").isoformat() if "timestamp" in msg else str(msg["_id"].generation_time)
                } for msg in messages
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.get("/sessions/{email}")
async def get_all_sessions(email:str):
    "Get all unique session ids"
    try:
        session_ids=get_all_session_ids(email=email)
        print("last session: ", get_last_session(email=email))
        return session_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")



@app.get("/sessions/{email}/last")
async def get_last_session_by_email(email: str):
    try:
        print("inside api")
        last_session = get_last_session(email)
        
        if last_session:
            return {"session_id": last_session}
        else:
            return {"session_id": None, "message": "No sessions found"}
            
    except Exception as e:
        logger.error(f"Error in get_last_session endpoint: {e}")
        return {"error": "Failed to fetch last session"}


@app.delete("/chat/sessions/{session_id}", response_model=StandardResponse)
async def clear_chat_session(session_id: str):
    """Clear a chat session"""

    try:
        clear_conversation_from_db(session_id)
        return StandardResponse(
            success=True,
            message="Session cleared",
            data={"session_id": session_id}
        )
    except:
        return StandardResponse(
            success=False,
            message="Session not found",
            data={"session_id": session_id}
        )




# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": "/docs"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please check logs for details"}


