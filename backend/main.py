
from fastapi import FastAPI, HTTPException, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager
import uuid

# Import from our organized files
from site_manager import SiteData, AuthenticationManager

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

# Global variable to store the manager
ollama_site_manager = None

def initialize_ollama_site_manager():
    """Initialize the Ollama site manager"""
    global ollama_site_manager
    
    try:
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from site_manager import SiteManager  # Replace with actual import
        
        auth_manager = AuthenticationManager()
        ollama_site_manager = SiteManager(auth_manager)
        # Test the connection

        
        print("Ollama Site Manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize Ollama Site Manager: {e}")
        ollama_site_manager = None
        return False


def initialize_ollama_permission_manager():
    """Initialize the Ollama permission manager"""
    global ollama_permission_manager
    
    try:
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from site_manager import PermissionManager  # Replace with actual import
        
        auth_manager = AuthenticationManager()
        ollama_permission_manager = PermissionManager(auth_manager)
        # Test the connection

        
        print("Ollama Permission Manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize Ollama Permission Manager: {e}")
        ollama_permission_manager = None
        return False
    
def initialize_ollama_user_manager():
    """Initialize the Ollama user manager"""
    global ollama_user_manager
    
    try:
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from site_manager import UserManager  # Replace with actual import
        
        auth_manager = AuthenticationManager()
        ollama_user_manager = UserManager(auth_manager)
        # Test the connection

        
        print("Ollama User Manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize Ollama User Manager: {e}")
        ollama_user_manager = None
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










import logging
import json


# Setup logging
logging.basicConfig(level=logging.INFO)
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
client_mongo = MongoClient(MONGO_URI)
db = client_mongo.Conversations
conversations_collection = db.conversations

# Phase 1 Prompt - Intent Detection and Data Collection
PHASE_1_PROMPT = """You are PulsePro AI Assistant.

====================CORE RULES====================
• Handle ONLY these operations: CREATE/VIEW/DELETE sites and users, ASSIGN/UNASSIGN users to sites, VIEW permission sets
• Ignore unrelated queries. Reply: "I can only help with PulsePro operations."
• Ask for missing information. Never assume values.
• If user says cancel/stop/exit/abort/halt/quit/terminate/end, reply: "Operation cancelled. No action taken."
• When you have all required data, ask: "I have all the information needed. Type 'Proceed' to execute this operation."

====================OPERATIONS & REQUIRED DATA====================
1. CREATE_SITE: location_name
2. VIEW_SITES: no data needed  
3. DELETE_SITE: location_name
4. ASSIGN_USERS_TO_SITE: location_name, user_names (list)
5. UNASSIGN_USERS_FROM_SITE: location_name, user_names (list)
6. CREATE_USER: first_name, last_name, email, permission_set
7. DELETE_USER: full_name
8. VIEW_USERS: no data needed
9. VIEW_PERMISSION_SETS: no data needed

====================CONVERSATION EXAMPLES====================

CREATE SITE:
User: "Create a site"
Assistant: "What should be the name/location of this site?"
User: "Mumbai Office"
Assistant: "Perfect! I have all the information needed. Type 'Proceed' to execute this operation."

DELETE SITE:
User: "Delete a site"
Assistant: "Which site do you want to delete? Available sites: {all_sites_list}"
User: "Bangalore Hub"
Assistant: "Great! I have all the information needed. Type 'Proceed' to execute this operation."

VIEW SITES:
User: "Show me all sites"
Assistant: "Great! Type 'Proceed' to execute this operation."

ASSIGN USERS TO SITE:
User: "Assign users to site"
Assistant: "Which site? Available sites: {all_sites_list}"
User: "Delhi Office"
Assistant: "Which users? Available users: {all_users_list}"
User: "John, Sarah"
Assistant: "Great! I have all the information needed. Type 'Proceed' to execute this operation."

UNASSIGN USERS FROM SITE:
User: "Unassign users from site"
Assistant: "Which site? Available sites: {all_sites_list}"
User: "Delhi Office"
Assistant: "Which users? Available users: {all_users_list}"
User: "John, Sarah"
Assistant: "Great! I have all the information needed. Type 'Proceed' to execute this operation."

CREATE USER:
User: "Create a user"
Assistant: "What is the first name?"
User: "John"
Assistant: "What is the last name?"
User: "Doe"
Assistant: "What is the email address?"
User: "john@company.com"
Assistant: "Which permission set? Available sets: {all_permission_sets_list}"
User: "Field User"
Assistant: "Perfect! I have all the information needed. Type 'Proceed' to execute this operation."

DELETE USER:
User: "Delete a user"
Assistant: "Which user? Available users: {all_users_list}"
User: "John Doe"
Assistant: "Great! I have all the information needed. Type 'Proceed' to execute this operation."

VIEW USERS:
User: "Show me all users"
Assistant: "Great! Type 'Proceed' to execute this operation."

VIEW PERMISSION SETS:
User: "Show me all permission sets"
Assistant: "Great! Type 'Proceed' to execute this operation."

====================WHEN TO SHOW LISTS====================
Show sites list ONLY for: DELETE_SITE, ASSIGN_USERS_TO_SITE, UNASSIGN_USERS_FROM_SITE
Show users list ONLY for: ASSIGN_USERS_TO_SITE, UNASSIGN_USERS_FROM_SITE, DELETE_USER
Show permission sets ONLY for: CREATE_USER

Available Sites: {all_sites_list}
Available Users: {all_users_list}
Available Permission Sets: {all_permission_sets_list}
"""

# Phase 2 Prompt - JSON Generation
PHASE_2_PROMPT = """You are a JSON generator for PulsePro Site operations.

ANALYZE the conversation history and generate ONLY a JSON response in this EXACT format:

For CREATE_SITE:
{"data": {"location_name": "EXTRACTED_NAME"}, "operation_type": "CREATE_SITE"}

For VIEW_SITES:
{"data": {}, "operation_type": "VIEW_SITES"}

For DELETE_SITE:
{"data": {"location_name": "EXTRACTED_NAME"}, "operation_type": "DELETE_SITE"}

For ASSIGN_USERS_TO_SITE:
{"data": {"location_name": "EXTRACTED_NAME", "user_list": ["USER1", "USER2"]}, "operation_type": "ASSIGN_USERS_TO_SITE"}

For UNASSIGN_USERS_FROM_SITE:
{"data": {"location_name": "EXTRACTED_NAME", "user_list": ["USER1", "USER2"]}, "operation_type": "UNASSIGN_USERS_FROM_SITE"}

For CREATE_USER:
{"data": {"first_name": "FIRSTNAME", "last_name": "LASTNAME", "email": "EMAIL", "permission_set": "PERMISSIONSET"}, "operation_type": "CREATE_USER"}

For DELETE_USER:
{"data": {"full_name": "FULLNAME"}, "operation_type": "DELETE_USER"}

For VIEW_USERS:
{"data": {}, "operation_type": "VIEW_USERS"}

For VIEW_PERMISSION_SETS:
{"data": {}, "operation_type": "VIEW_PERMISSION_SETS"}

RULES:
- Extract the data from the conversation history
- Return ONLY the JSON object, nothing else
- No explanations, no text, no markdown, just pure JSON
- Use the exact operation_type names shown above
- Strictly follow the JSON format shown above , including field names and structure

"""

# Simple session storage (keeping for backward compatibility)
chat_sessions = {}

def save_conversation_to_db(session_id: str, role: str, message: str):
    """Save message to MongoDB"""
    try:
        conversations_collection.insert_one({
            "session_id": session_id,
            "role": role,
            "message": message,
            "timestamp": datetime.now()
        })
    except Exception as e:
        logger.error(f"Failed to save to MongoDB: {e}")

def get_conversation_from_db(session_id: str) -> list:
    """Get conversation history from MongoDB"""
    try:
        messages = conversations_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1)
        return list(messages)
    except Exception as e:
        logger.error(f"Failed to get from MongoDB: {e}")
        return []

def clear_conversation_from_db(session_id: str):
    """Clear conversation history from MongoDB"""
    try:
        conversations_collection.delete_many({"session_id": session_id})
    except Exception as e:
        logger.error(f"Failed to clear from MongoDB: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatRequest):
    """Two-phase chat agent: Phase 1 (Chat) → Phase 2 (Execute)"""
    
    session_id = chat_request.session_id or str(uuid.uuid4())
    user_message = chat_request.message.strip()
    
    try:
        client = get_ollama_client()
        
        # Initialize session (MongoDB-based)
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {"conversation": []}
        
        # Add user message to conversation and save to MongoDB
        save_conversation_to_db(session_id, "user", user_message)
        
        # Check if user wants to execute (Phase 2)
        cancel_triggers = ["cancel", "stop", "exit", "abort", "halt", "quit", "terminate", "end"]
        if any(trigger in user_message.lower() for trigger in cancel_triggers):
            clear_conversation_from_db(session_id)
            return ChatResponse(
                message="Operation cancelled. No action taken.",
                status="cancelled",
                session_id=session_id,
                context={"phase": "cancelled"},
                data={}
            )
        
        execution_triggers = ["proceed", "execute", "go", "do it", "yes proceed", "execute now"]
        if user_message.lower().strip() in execution_triggers:
            return await execute_phase_2(session_id, client)
        
        # Phase 1: Continue conversation
        return await execute_phase_1(session_id, user_message, client)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            message=f"Sorry, something went wrong: {str(e)}",
            status="error",
            session_id=session_id,
            context={"error": str(e)},
            data={}
        )


# Function to get formatted site list
def get_sites_list_formatted():
    """Get all sites and format them as a string"""
    initialize_ollama_site_manager()
    try:
        if ollama_site_manager:
            result = ollama_site_manager.get_all_sites()
            sites = result.get('locations', [])
            if sites:
                # Format as numbered list
                sites_list = "\n".join([f"{i+1}. {site.get('location_name', 'Unknown'),{site.get('city')}, {site.get('state')}}" 
                                      for i, site in enumerate(sites)])
                return f"Current sites:\n{sites_list}"
            else:
                return "No sites currently exist."
        else:
            return "Site information unavailable."
    except Exception as e:
        return "Unable to retrieve sites."



# Updated execute_phase_1 function
async def execute_phase_1(session_id: str, user_message: str, client) -> ChatResponse:
    """Phase 1: Normal chat - Intent detection and data collection"""
    initialize_ollama_site_manager()
    initialize_ollama_permission_manager()
    # Get conversation history from MongoDB
    db_messages = get_conversation_from_db(session_id)
    conversation_history = ""
    for msg in db_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_history += f"{role}: {msg['message']}\n"
    
    # Get current sites list
    sites_list = get_sites_list_formatted()
    print(f"Available sites: {sites_list}")

    all_user_list = ollama_site_manager.get_all_users()
    permission_set_list = ollama_permission_manager.get_all_permission_sets()
    # Format the prompt with sites list
    formatted_prompt = PHASE_1_PROMPT.format(all_sites_list=sites_list,all_users_list=all_user_list,all_permission_sets_list=permission_set_list)
    
    # Create full prompt for Phase 1
    full_prompt = f"""{formatted_prompt}

====================CONVERSATION HISTORY====================
{conversation_history}

====================CURRENT USER MESSAGE====================
{user_message}

====================YOUR RESPONSE===================="""

    # Get response from LLM
    response = client.generate(
        model="llama3.1:8b",
        prompt=full_prompt,
        options={
            "num_predict": 450,
            "temperature": 0.7
        }
    )
    
    ai_response = response['response'].strip()
    
    # Save AI response to MongoDB
    save_conversation_to_db(session_id, "assistant", ai_response)
    
    # Determine status
    if "Type 'Proceed' to execute" in ai_response:
        status = "ready_for_execution"
    elif "I can only help with PulsePro Site operations" in ai_response:
        status = "off_topic"
    else:
        status = "collecting_data"
    
    return ChatResponse(
        message=ai_response,
        status=status,
        session_id=session_id,
        context={
            "phase": 1,
            "conversation_length": len(get_conversation_from_db(session_id)),
            "available_sites": len(sites_list.split('\n')) - 1 if 'Current sites:' in sites_list else 0
        },
        data={}
    )

async def execute_phase_2(session_id: str, client) -> ChatResponse:
    """Phase 2: Generate JSON and execute operation"""
    
    try:
        # Get conversation history from MongoDB for Phase 2
        db_messages = get_conversation_from_db(session_id)
        conversation_history = ""
        for msg in db_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {msg['message']}\n"
        
        # Create Phase 2 prompt
        phase_2_prompt = f"""{PHASE_2_PROMPT}

====================CONVERSATION HISTORY====================
{conversation_history}

     JSON:"""
        
        # Get JSON response from LLM
        response = client.generate(
            model="llama3.1:8b",
            prompt=phase_2_prompt,
            options={
                "num_predict": 500,
                "temperature": 0.1  # Low temperature for precise JSON
            }
        )
        
        json_response = response['response'].strip()
        
        # Clean and parse JSON
        if json_response.startswith('```'):
            # Remove markdown formatting if present
            json_response = json_response.split('```')[1]
            if json_response.startswith('json'):
                json_response = json_response[4:]
        
        # Parse the JSON
        operation_data = json.loads(json_response)
        print("operation data : ", operation_data)
        
        # Execute the operation
        execution_result = await execute_site_operation(operation_data)
        
        # Clear conversation after execution
        clear_conversation_from_db(session_id)
        
        return ChatResponse(
            message=execution_result["message"],
            status="completed",
            session_id=session_id,
            context={
                "phase": 2,
                "operation": operation_data["operation_type"],
                "executed": True
            },
            data=execution_result["data"]
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Raw response: {json_response}")
        return ChatResponse(
            message="❌ Failed to parse operation data. Please try again.",
            status="error",
            session_id=session_id,
            context={"phase": 2, "error": "json_parse_error"},
            data={"raw_response": json_response}
        )
    
    except Exception as e:
        logger.error(f"Phase 2 execution error: {e}")
        return ChatResponse(
            message=f"❌ Failed to execute operation: {str(e)}",
            status="error",
            session_id=session_id,
            context={"phase": 2, "error": str(e)},
            data={}
        )

async def execute_site_operation(operation_data: dict) -> dict:
    """Execute the site operation based on JSON data"""
    
    try:
        initialize_ollama_site_manager()
        initialize_ollama_user_manager()
        print("in execute")
        operation_type = operation_data.get("operation_type")
        data = operation_data.get("data", {})
        print("data: ",data)
        if operation_type == "CREATE_SITE":
            location_name = data.get("location_name")
            print("in CREATE_SITE")
            result = ollama_site_manager.create_site_by_name_only(location_name)
            return {
                "success": True,
                "message": f"✅ Site '{location_name}' created successfully!",
                "data": result
            }
        
        elif operation_type == "DELETE_SITE":
            location_name = data.get("location_name")
            
            # Find and delete site
            all_sites = ollama_site_manager.get_all_sites()
            site_id = None
            for site in all_sites.get('locations', []):
                if site.get('location_name', '').lower() == location_name.lower():
                    site_id = site.get('id')
                    break
            
            if site_id:
                ollama_site_manager.delete_site(site_id)
                return {
                    "success": True,
                    "message": f"✅ Site '{location_name}' deleted successfully!",
                    "data": {"deleted": True, "site_id": site_id}
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Site '{location_name}' not found",
                    "data": {"error": "Site not found"}
                }
        
        elif operation_type == "VIEW_SITES":
            result = ollama_site_manager.get_all_sites()
            sites = result.get('locations', [])
            if sites:
                site_list = "\n".join([f"• {site.get('location_name', 'Unknown')} ,{site.get('city'), site.get('state')}" for site in sites])
                message = f"📍 Found {len(sites)} sites:\n{site_list}"
            else:
                message = "📍 No sites found"
            
            return {
                "success": True,
                "message": message,
                "data": result
            }
        
        elif operation_type == "ASSIGN_USERS_TO_SITE":
            location_name = data.get("location_name")
            user_names = data.get("user_list", [])
            
            # Find site ID
            all_sites = ollama_site_manager.get_all_sites()
            site_id = None
            for site in all_sites.get('locations', []):
                if site.get('location_name', '').lower() == location_name.lower():
                    site_id = site.get('id')
                    break

            if not site_id:
                return {
                    "success": False,
                    "message": f"❌ Site '{location_name}' not found",
                    "data": {"error": "Site not found"}
                }
            
            all_users = ollama_site_manager.get_site_users(location_id=site_id)
            user_ids= []
            for user in all_users.get('users', []) :
                if user.get('username', '').lower() in [uname.lower() for uname in user_names]:
                    user_ids.append(user.get('id'))
                    print("user ids: ",user_ids)


            if not user_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified users were found: {', '.join(user_names)}",
                    "data": {"error": "Users not found"}
                }
            
            # Assign users
            result = ollama_site_manager.assign_users_to_site(site_id, user_ids=user_ids)
            return {
                "success": True,
                "message": f"✅ Users assigned to site '{location_name}' successfully!",
                "data": result
            }
        
        elif operation_type == "UNASSIGN_USERS_FROM_SITE":
            location_name = data.get("location_name")
            user_names = data.get("user_list", [])
            
            # Find site ID
            all_sites = ollama_site_manager.get_all_sites()
            site_id = None
            for site in all_sites.get('locations', []):
                if site.get('location_name', '').lower() == location_name.lower():
                    site_id = site.get('id')
                    break

            if not site_id:
                return {
                    "success": False,
                    "message": f"❌ Site '{location_name}' not found",
                    "data": {"error": "Site not found"}
                }
            
            all_users = ollama_site_manager.get_site_users(location_id=site_id)
            user_ids= []
            for user in all_users.get('users', []) :
                if user.get('username', '').lower() in [uname.lower() for uname in user_names] and user.get('mapped'):
                    user_ids.append(user.get('id'))
                    print("user ids: ",user_ids)


            if not user_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified users were found: {', '.join(user_names)}",
                    "data": {"error": "Users not found"}
                }
            
            # Unassign users
            result = ollama_site_manager.unassign_users_from_site(mapped_location_ids=user_ids)
            return {
                "success": True,
                "message": f"✅ Users unassigned from site '{location_name}' successfully!",
                "data": result
            }
         
        elif operation_type == "CREATE_USER":
            first_name = data.get("first_name")
            last_name = data.get("last_name")
            email = data.get("email")

            permission_set_id = ollama_permission_manager.get_permission_set_id_by_name(data.get("permission_set"))
            if not permission_set_id:
                return {
                    "success": False,
                    "message": f"❌ Permission set '{data.get('permission_set')}' not found",
                    "data": {"error": "Permission set not found"}
                }
            
            result = ollama_user_manager.create_user(first_name, last_name, email, permission_set_ids=permission_set_id)
            return {
                "success": True,
                "message": f"✅ User '{first_name} {last_name}' created successfully!",
                "data": result
            }

        elif operation_type == "DELETE_USER":
            full_name = data.get("full_name")
            # first_name, last_name = full_name.split(' ', 1) if ' ' in full_name else (full_name, '')
            
            # Find user ID
            id=ollama_user_manager.get_user_id_by_name(full_name)
            if id:
                result = ollama_user_manager.delete_user(id)
                return {
                    "success": True,
                    "message": f"✅ User '{full_name}' deleted successfully!",
                    "data": result
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ User '{full_name}' not found",
                    "data": {"error": "User not found"}
                }
            
        elif operation_type == "VIEW_USERS":
            result = ollama_user_manager.get_all_users()
            users = result.get('users', [])
            if users:
                user_list = "\n".join([f"• {user.get('first_name', '')} {user.get('last_name', '')} ({user.get('email', 'No Email')})" for user in users])
                message = f"👤 Found {len(users)} users:\n{user_list}"
            else:
                message = "👤 No users found"
            
            return {
                "success": True,
                "message": message,
                "data": result
            }
        
        elif operation_type == "VIEW_PERMISSION_SETS":
            print("in permission sets")
            result = ollama_permission_manager.get_all_permission_sets()
            print("result: ",result)
            if result:
                ps_list = "\n".join([f"• {ps.get('name', 'Unknown')}" for ps in result])
                message = f"🔑 Found {len(result)} permission sets:\n{ps_list}"
            else:
                message = "🔑 No permission sets found"
            
            return {
                "success": True,
                "message": message,
                "data": result
            }

        else:
            return {
                "success": False,
                "message": f"❌ Unknown operation: {operation_type}",
                "data": {"error": "Unknown operation"}
            }
    


            
    


    except Exception as e:
        logger.error(f"Site operation error: {e}")
        return {
            "success": False,
            "message": f"❌ Operation failed: {str(e)}",
            "data": {"error": str(e)}
        }

@app.get("/chat/history/{session_id}")
async def get_chat_history_endpoint(session_id: str):
    """Get conversation history for a session from MongoDB"""
    try:
        messages = get_conversation_from_db(session_id)
        return {
            "session_id": session_id,
            "conversation": [
                {
                    "role": msg["role"],
                    "message": msg["message"], 
                    "timestamp": msg["timestamp"].isoformat()
                } for msg in messages
            ]
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
    """Check if chat system is ready"""
    try:
        client = get_ollama_client()
        
        # Quick test
        response = client.generate(
            model="llama3.1:8b",
            prompt="Say 'OK'",
            options={"num_predict": 5}
        )
        
        return StandardResponse(
            success=True,
            message="Chat system is healthy",
            data={
                "status": "ready",
                "model": "llama3.1:8b",
                "test_response": response['response'].strip()
            }
        )
        
    except Exception as e:
        return StandardResponse(
            success=False,
            message="Chat system is unhealthy",
            data={
                "status": "error",
                "error": str(e)
            }
        )






    





# Enhanced health check
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
        "components": {
            "api": "healthy",
            "authentication": "healthy" if ollama_site_manager else "not_initialized",
            "ollama_model": ollama_status,
            "conversational_ai": "available" if ollama_site_manager else "not_available"
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
    
    global ollama_site_manager, ollama_config
    
    status_info = {
        "system": {
            "initialized": ollama_site_manager is not None,
            "model_name": ollama_config.model_name,
            "temperature": ollama_config.temperature
        },
        "ollama": {
            "available": False,
            "models": [],
            "target_model_status": "unknown"
        },
        "conversations": {
            "active_sessions": 0,
            "total_conversations": 0
        }
    }
    
    # Check Ollama status
    try:
        import ollama
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        
        status_info["ollama"]["available"] = True
        status_info["ollama"]["models"] = available_models
        status_info["ollama"]["target_model_status"] = "available" if ollama_config.model_name in available_models else "not_found"
        
    except Exception as e:
        status_info["ollama"]["error"] = str(e)
    
    # Check conversation stats
    if ollama_site_manager:
        agent = ollama_site_manager.conversational_agent
        status_info["conversations"]["active_sessions"] = len(agent.conversations)
        status_info["conversations"]["total_conversations"] = sum(
            len(ctx.conversation_history) for ctx in agent.conversations.values()
        )
    
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
    
    print("🤖 PulsePro Site Management with Ollama")
    print("=" * 60)
    print(f"🕐 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Model: {ollama_config.model_name}")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"💬 Chat: http://localhost:8000/chat/interface")
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