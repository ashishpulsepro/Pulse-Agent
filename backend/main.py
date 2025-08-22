
from fastapi import FastAPI, HTTPException, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager
import uuid
import logging
import json

# Import from our organized files
from site_manager import SiteData, AuthenticationManager, SiteManager
from tree_system import TreeBasedSystem

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

class ChatResponse(BaseModel):
    message: str
    status: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None

# Global variables
tree_system = None
ollama_config = OllamaConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting PulsePro Site Management API with Tree System...")
    print(f"System: Tree-Based Intent Processing")
    yield
    # Shutdown
    print("Shutting down PulsePro Site Management API...")

app = FastAPI(
    title="PulsePro Site Management API with Tree System",
    description="Tree-based conversational site management system with structured intent processing: CREATE → READ → DELETE",
    version="3.0.0",
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

# Global variable to store the tree system
tree_system = None

def initialize_tree_system():
    """Initialize the Tree-based System"""
    global tree_system
    
    try:
        # Initialize authentication and site manager
        auth_manager = AuthenticationManager()
        site_manager = SiteManager(auth_manager)
        
        # Initialize tree system
        tree_system = TreeBasedSystem(site_manager)
        
        print("🌳 Tree-based System initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize Tree System: {e}")
        tree_system = None
        return False




# Test Ollama connection endpoint
@app.get("/ollama/test", response_model=StandardResponse)
async def test_ollama_connection():
    """Test Ollama connection and model availability"""
    try:
        import ollama
        
        # Try to create client with explicit host (adjust if needed)
        client = ollama.Client(host='http://localhost:11434')
        
        # Test basic connection first
        try:
            models = client.list()
            
            # Handle the models response properly
            available_models = []
            if hasattr(models, 'models'):
                # Handle case where models is an object with models attribute
                model_list = models.models
            elif isinstance(models, dict) and 'models' in models:
                # Handle case where models is a dict with models key
                model_list = models['models']
            else:
                # Handle other cases
                model_list = models if isinstance(models, list) else [models]
            
            # Extract model names
            for model in model_list:
                if hasattr(model, 'model'):
                    # Handle model objects with model attribute
                    available_models.append(model.model)
                elif hasattr(model, 'name'):
                    # Handle model objects with name attribute
                    available_models.append(model.name)
                elif isinstance(model, dict):
                    # Handle dict models
                    name = model.get('model') or model.get('name') or model.get('id')
                    if name:
                        available_models.append(name)
                else:
                    # Fallback to string representation
                    available_models.append(str(model))
        except Exception as list_error:
            return StandardResponse(
                success=False,
                message="Failed to list models from Ollama",
                data={
                    "error": str(list_error),
                    "suggestions": [
                        "Check if Ollama is running: ollama serve",
                        "Verify Ollama is accessible at http://localhost:11434",
                        "Try: curl http://localhost:11434/api/tags"
                    ]
                }
            )
        
        # Test llama3.1:8b specifically
        model_available = "llama3.1:8b" in available_models
        
        if model_available:
            try:
                # Test generation with timeout
                response = client.generate(
                    model="llama3.1:8b",
                    prompt="who was the first president of the united states?",
                    options={"num_predict": 500}
                )

                print(f"Test response: {response['response']}")
                
                return StandardResponse(
                    success=True,
                    message="Ollama connection successful",
                    data={
                        "available_models": available_models,
                        "target_model": "llama3.1:8b",
                        "model_status": "available",
                        "test_response": response['response']
                    }
                )
            except Exception as gen_error:
                return StandardResponse(
                    success=False,
                    message="Model found but generation failed",
                    data={
                        "available_models": available_models,
                        "target_model": "llama3.1:8b",
                        "model_status": "available_but_failed",
                        "error": str(gen_error),
                        "suggestion": "Model may be corrupted, try: ollama pull llama3.1:8b"
                    }
                )
        else:
            return StandardResponse(
                success=False,
                message="llama3.1:8b model not found",
                data={
                    "available_models": available_models,
                    "target_model": "llama3.1:8b",
                    "model_status": "not_found",
                    "suggestion": "Run: ollama pull llama3.1:8b"
                }
            )
            
    except ImportError:
        return StandardResponse(
            success=False,
            message="Ollama Python package not installed",
            data={
                "error": "ImportError: ollama module not found",
                "suggestion": "Install with: pip install ollama"
            }
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            message="Ollama connection failed",
            data={
                "error": str(e),
                "error_type": type(e).__name__,
                "suggestions": [
                    "Make sure Ollama is running: ollama serve",
                    "Check if port 11434 is accessible",
                    "Verify Ollama installation: ollama --version",
                    "Try manual test: curl http://localhost:11434/api/tags"
                ]
            }
        )





import ollama
import uuid
from datetime import datetime
from typing import Dict, List, Optional

# Global storage for chat sessions (in production, use Redis or database)
chat_sessions: Dict[str, List[Dict]] = {}

def get_ollama_client():
    """Get Ollama client - simple dependency"""
    try:
        client = ollama.Client(host='http://localhost:11434')
        # Quick test to ensure connection
        client.list()
        return client
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama service unavailable: {str(e)}"
        )













logging.basicConfig(
    level=logging.INFO,  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Create logger instance
logger = logging.getLogger(__name__)

# SITE_PROMPT = """You are PulsePro AI Assistant for Site Management.

# ====================CORE RULES====================
# * Handle ONLY PulsePro Site operations.
# * STRICTLY ignore chit-chat or unrelated queries. If query is unrelated, reply with: "I can only help with PulsePro Site operations."
# * Be systematic, precise, and strict.
# * NEVER assume or invent values. If a required value is missing, always explicitly ask the user for it.
# * Use the conversation history provided in the context to track collected fields. Do not forget previous user responses.

# ====================WORKFLOW====================
# 1. INTENT DETECTION
# * Detect which operation the user wants.
# * If unclear, ask the user to clarify.
# * Valid operations: CREATE_SITE, DELETE_SITE, VIEW_SITES, UPDATE_SITE

# 2. REQUIRED FIELDS COLLECTION
# * Each operation has REQUIRED_FIELDS.
# * Ask for ONLY ONE missing field at a time.
# * Do not continue until the user provides the missing field.
# * Do NOT create defaults for required fields.
# * If a value is already provided in conversation history, reuse it instead of asking again.

# REQUIRED_FIELDS:
# * CREATE_SITE: location_name (string)
# * DELETE_SITE: location_name (string)
# * VIEW_SITES: (no fields required)
# * UPDATE_SITE: location_name (string), field_to_update (string), new_value (string)

# 3. EXECUTION READINESS
# * When all required fields are collected, your response MUST BE ONLY the following format (no explanations, no extra text):

# For CREATE_SITE:
# READY_FOR_EXECUTION [location_name: VALUE]
# Should I proceed with creating this site? (yes/no)

# For DELETE_SITE:
# READY_FOR_EXECUTION [location_name: VALUE]
# Should I proceed with deleting this site? (yes/no)

# For VIEW_SITES:
# READY_FOR_EXECUTION
# Should I proceed with showing all sites? (yes/no)

# For UPDATE_SITE:
# READY_FOR_EXECUTION [location_name: VALUE1, field_to_update: VALUE2, new_value: VALUE3]
# Should I proceed with updating this site? (yes/no)

# 4. USER DECISION
# * If user replies "yes", respond ONLY with the appropriate JSON:

# For CREATE_SITE:
# { "operation": "CREATE_SITE", "data": { "location_name": "VALUE" }}

# For DELETE_SITE:
# { "operation": "DELETE_SITE", "data": { "location_name": "VALUE" }}

# For VIEW_SITES:
# { "operation": "VIEW_SITES", "data": {} }

# For UPDATE_SITE:
# { "operation": "UPDATE_SITE", "data": { "location_name": "VALUE1", "field_to_update": "VALUE2", "new_value": "VALUE3" }}

# * If user replies "no":
# Operation cancelled. No action taken.

# ====================EXAMPLES====================

# Example 1 - CREATE:
# User: "Create a site in Delhi"
# Assistant: READY_FOR_EXECUTION [location_name: Delhi]
# Should I proceed with creating this site? (yes/no)

# User: "yes"
# Assistant: { "operation": "CREATE_SITE", "data": { "location_name": "Delhi" }}

# Example 2 - DELETE:
# User: "Delete Mumbai office"
# Assistant: READY_FOR_EXECUTION [location_name: Mumbai office]
# Should I proceed with deleting this site? (yes/no)

# User: "yes"
# Assistant: { "operation": "DELETE_SITE", "data": { "location_name": "Mumbai office" }}

# Example 3 - VIEW:
# User: "Show me all sites"
# Assistant: READY_FOR_EXECUTION
# Should I proceed with showing all sites? (yes/no)

# User: "yes"
# Assistant: { "operation": "VIEW_SITES", "data": {} }

# Example 4 - UPDATE:
# User: "Update Delhi site"
# Assistant: What field do you want to update?

# User: "address"
# Assistant: What is the new address value?

# User: "New Delhi, India"
# Assistant: READY_FOR_EXECUTION [location_name: Delhi site, field_to_update: address, new_value: New Delhi, India]
# Should I proceed with updating this site? (yes/no)

# User: "yes"
# Assistant: { "operation": "UPDATE_SITE", "data": { "location_name": "Delhi site", "field_to_update": "address", "new_value": "New Delhi, India" }}

# Example 5 - CANCEL:
# User: "no"
# Assistant: Operation cancelled. No action taken.
# """

# Simple session storage
chat_sessions = {}
# MongoDB setup
from pymongo import MongoClient
from datetime import datetime
import uuid
import urllib.parse


# MongoDB connection
username = "ashish"
password = urllib.parse.quote_plus("Radhey@123")  # URL encode the password
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.3uxl669.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# MongoDB client with SSL configuration
try:
    client_mongo = MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,  # For development, remove in production
        serverSelectionTimeoutMS=5000,  # 5 second timeout
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
        maxPoolSize=10
    )
    
    # Test the connection
    client_mongo.admin.command('ping')
    db = client_mongo.Conversations
    conversations_collection = db.conversations
    logger.info("MongoDB connection successful")
    MONGODB_AVAILABLE = True
    
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    logger.info("Continuing without MongoDB - using in-memory storage")
    client_mongo = None
    db = None
    conversations_collection = None
    MONGODB_AVAILABLE = False

# Phase 1 Prompt - Intent Detection and Data Collection  
PHASE_1_PROMPT = """You are PulsePro Site Management Assistant. You handle EXACTLY 3 operations:

OPERATIONS:
1. CREATE_SITE: Create a new site (needs: location_name)
2. VIEW_SITES: Show all sites (needs: nothing)  
3. DELETE_SITE: Delete a site (needs: location_name)

STRICT RULES:
- If user asks about anything else, respond EXACTLY: "I only handle site operations: CREATE, VIEW, DELETE sites."
- Never invent, assume, or guess any values
- Ask for ONE missing piece of information at a time
- Use conversation history to avoid re-asking for already provided information

WORKFLOW:
1. Identify which operation (CREATE/VIEW/DELETE)
2. If CREATE or DELETE: get the location_name from user
3. If VIEW: no additional data needed
4. When all required data is collected, respond EXACTLY: "I have all the information needed. Type 'Proceed' to execute this operation."

EXAMPLES:
User: "Create a site"
You: "What should be the name of this site?"

User: "Mumbai Office"
You: "I have all the information needed. Type 'Proceed' to execute this operation."

User: "Delete Delhi branch" 
You: "I have all the information needed. Type 'Proceed' to execute this operation."

User: "Show sites"
You: "I have all the information needed. Type 'Proceed' to execute this operation."

User: "What's the weather?"
You: "I only handle site operations: CREATE, VIEW, DELETE sites."

CURRENT CONVERSATION HISTORY:"""

# Phase 2 Prompt - JSON Generation
PHASE_2_PROMPT = """You are a JSON generator. Extract operation details from conversation and output ONLY valid JSON.

ANALYZE the conversation and determine:
1. What operation: CREATE_SITE, VIEW_SITES, or DELETE_SITE  
2. What location_name (if CREATE_SITE or DELETE_SITE)

OUTPUT ONLY JSON:

For CREATE_SITE:
{"data": {"location_name": "EXACT_NAME_FROM_CONVERSATION"}, "operation_type": "CREATE_SITE"}

For VIEW_SITES:  
{"data": {}, "operation_type": "VIEW_SITES"}

For DELETE_SITE:
{"data": {"location_name": "EXACT_NAME_FROM_CONVERSATION"}, "operation_type": "DELETE_SITE"}

STRICT RULES:
- Output ONLY the JSON object
- No explanations, no markdown, no extra text
- Use exact location name from conversation
- Use exact operation_type names: CREATE_SITE, VIEW_SITES, DELETE_SITE

CONVERSATION:"""

# Simple session storage (keeping for backward compatibility)
chat_sessions = {}

def save_conversation_to_db(session_id: str, role: str, message: str):
    """Save message to MongoDB if available, otherwise store in memory"""
    try:
        if MONGODB_AVAILABLE and conversations_collection is not None:
            conversations_collection.insert_one({
                "session_id": session_id,
                "role": role,
                "message": message,
                "timestamp": datetime.now()
            })
        else:
            # Fallback to in-memory storage
            if session_id not in chat_sessions:
                chat_sessions[session_id] = {"conversation": []}
            chat_sessions[session_id]["conversation"].append({
                "role": role,
                "message": message,
                "timestamp": datetime.now()
            })
    except Exception as e:
        logger.error(f"Failed to save to MongoDB: {e}")
        # Fallback to in-memory storage
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {"conversation": []}
        chat_sessions[session_id]["conversation"].append({
            "role": role,
            "message": message,
            "timestamp": datetime.now()
        })

def get_conversation_from_db(session_id: str) -> list:
    """Get conversation history from MongoDB if available, otherwise from memory"""
    try:
        if MONGODB_AVAILABLE and conversations_collection is not None:
            messages = conversations_collection.find(
                {"session_id": session_id}
            ).sort("timestamp", 1)
            return list(messages)
        else:
            # Fallback to in-memory storage
            if session_id in chat_sessions:
                return chat_sessions[session_id].get("conversation", [])
            return []
    except Exception as e:
        logger.error(f"Failed to get from MongoDB: {e}")
        # Fallback to in-memory storage
        if session_id in chat_sessions:
            return chat_sessions[session_id].get("conversation", [])
        return []

def clear_conversation_from_db(session_id: str):
    """Clear conversation history from MongoDB if available, otherwise from memory"""
    try:
        if MONGODB_AVAILABLE and conversations_collection is not None:
            conversations_collection.delete_many({"session_id": session_id})
        else:
            # Fallback to in-memory storage
            if session_id in chat_sessions:
                chat_sessions[session_id]["conversation"] = []
    except Exception as e:
        logger.error(f"Failed to clear from MongoDB: {e}")
        # Fallback to in-memory storage
        if session_id in chat_sessions:
            chat_sessions[session_id]["conversation"] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatRequest):
    """Tree-based chat system: Intent → Data → Confirmation → Execution"""
    
    session_id = chat_request.session_id or str(uuid.uuid4())
    user_message = chat_request.message.strip()
    
    try:
        # Initialize tree system if not already done
        if tree_system is None:
            initialize_tree_system()
        
        if tree_system is None:
            raise Exception("Tree system not available")
        
        # Process message through tree system
        result = tree_system.process_message(user_message, session_id)
        
        return ChatResponse(
            message=result["message"],
            status=result["status"],
            session_id=result["session_id"],
            context=result.get("context", {}),
            data=result.get("result", {})
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            message=f"❌ System error: {str(e)}",
            status="error",
            session_id=session_id,
            context={"error": str(e)},
            data={}
        )

@app.get("/chat/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get information about a chat session"""
    if tree_system is None:
        raise HTTPException(status_code=503, detail="Tree system not available")
    
    session_info = tree_system.get_session_info(session_id)
    if session_info is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_info

@app.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific chat session"""
    if tree_system is None:
        raise HTTPException(status_code=503, detail="Tree system not available")
    
    cleared = tree_system.clear_session(session_id)
    return {"cleared": cleared, "session_id": session_id}

@app.delete("/chat/sessions")
async def clear_all_sessions():
    """Clear all chat sessions"""
    if tree_system is None:
        raise HTTPException(status_code=503, detail="Tree system not available")
    
    count = tree_system.clear_all_sessions()
    return {"cleared_sessions": count}

@app.get("/chat/history/{session_id}")
async def get_chat_history_endpoint(session_id: str):
    """Get conversation history for a session from MongoDB or memory"""
    try:
        messages = get_conversation_from_db(session_id)
        return {
            "session_id": session_id,
            "conversation": [
                {
                    "role": msg["role"],
                    "message": msg["message"], 
                    "timestamp": msg["timestamp"].isoformat() if hasattr(msg["timestamp"], 'isoformat') else str(msg["timestamp"])
                } for msg in messages
            ],
            "storage_type": "mongodb" if MONGODB_AVAILABLE else "memory"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")
    





















@app.delete("/chat/sessions/{session_id}", response_model=StandardResponse)
async def clear_chat_session(session_id: str):
    """Clear a chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return StandardResponse(
            success=True,
            message="Session cleared",
            data={"session_id": session_id}
        )
    else:
        return StandardResponse(
            success=False,
            message="Session not found",
            data={"session_id": session_id}
        )

@app.get("/chat/health", response_model=StandardResponse)
async def chat_health_check():
    """Check if tree-based chat system is ready"""
    try:
        # Initialize tree system if not already done
        if tree_system is None:
            initialize_tree_system()
        
        if tree_system is None:
            raise Exception("Tree system initialization failed")
        
        # Test LLM availability if applicable
        llm_status = "available" if tree_system.llm_available else "fallback_mode"
        
        return StandardResponse(
            success=True,
            message="Tree-based chat system is healthy",
            data={
                "status": "ready",
                "system_type": "tree_based",
                "llm_status": llm_status,
                "supported_intents": ["CREATE", "READ", "DELETE"],
                "active_sessions": len(tree_system.sessions)
            }
        )
        
    except Exception as e:
        return StandardResponse(
            success=False,
            message="Tree-based chat system is unhealthy",
            data={
                "status": "error",
                "error": str(e),
                "system_type": "tree_based"
            }
        )






    





# Enhanced health check
@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with Tree System status"""
    
    global tree_system
    
    # Test Tree System connection
    tree_status = "not_tested"
    if tree_system:
        tree_status = "connected"
        if tree_system.llm_available:
            tree_status = "connected_with_llm"
        else:
            tree_status = "connected_fallback_mode"
    
    health_status = {
        "status": "healthy",
        "service": "PulsePro Site Management API with Tree System",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "authentication": "healthy" if tree_system else "not_initialized",
            "tree_system": tree_status,
            "conversational_ai": "available" if tree_system else "not_available",
            "supported_operations": ["CREATE", "READ", "DELETE"] if tree_system else []
        }
    }
    
    return health_status

# ============================================
# UTILITY ENDPOINTS
# ============================================

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with Ollama model status"""
    
    global ollama_site_manager
    
    # Test Ollama connection
    ollama_status = "not_tested"
    if ollama_site_manager:
        ollama_status = "connected" if ollama_site_manager.conversational_agent.ollama_client else "fallback_mode"
    
    health_status = {
        "status": "healthy",
        "service": "PulsePro Site Management API with Ollama",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "authentication": "healthy" if ollama_site_manager else "not_initialized",
            "ollama_model": ollama_status,
            "conversational_ai": "available" if ollama_site_manager else "not_available"
        }
    }
    
    return health_status


@app.get("/status")
async def get_system_status():
    """Get detailed system status"""
    
    global tree_system, ollama_config
    
    status_info = {
        "system": {
            "type": "tree_based",
            "initialized": tree_system is not None,
            "model_name": ollama_config.model_name,
            "temperature": ollama_config.temperature
        },
        "llm": {
            "available": tree_system.llm_available if tree_system else False,
            "type": "ollama" if tree_system and tree_system.llm_available else "fallback",
            "model": ollama_config.model_name
        },
        "conversations": {
            "active_sessions": len(tree_system.sessions) if tree_system else 0,
            "supported_intents": ["CREATE", "READ", "DELETE"]
        },
        "operations": {
            "create_site": "available",
            "view_sites": "available", 
            "delete_site": "available"
        }
    }
    
    # Check Ollama status if tree system is available
    if tree_system and tree_system.llm_available:
        try:
            import ollama
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            status_info["llm"]["available_models"] = available_models
            status_info["llm"]["target_model_status"] = "available" if ollama_config.model_name in available_models else "not_found"
            
        except Exception as e:
            status_info["llm"]["error"] = str(e)
    
    return status_info

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": "/docs"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please check logs for details"}

# ============================================
# STARTUP
# ============================================

if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    print("🌳 PulsePro Site Management with Tree-Based System")
    print("=" * 60)
    print(f"🕐 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 System: Tree-Based Intent Processing")
    print(f"🎯 Supported Operations: CREATE, READ, DELETE")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"💬 Chat: http://localhost:8000/chat")
    print(f"🔍 Health: http://localhost:8000/health")
    print(f"🧪 Ollama Test: http://localhost:8000/ollama/test")
    print("=" * 60)
    
    # Check command line arguments
    host = "0.0.0.0"
    port = 8000
    reload = True
    
    if len(sys.argv) > 1:
        if "--port" in sys.argv:
            port_idx = sys.argv.index("--port") + 1
            if port_idx < len(sys.argv):
                port = int(sys.argv[port_idx])
        
        if "--host" in sys.argv:
            host_idx = sys.argv.index("--host") + 1
            if host_idx < len(sys.argv):
                host = sys.argv[host_idx]
        
        if "--no-reload" in sys.argv:
            reload = False
    
    uvicorn.run(app, host=host, port=port, reload=reload)