
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
    session_intent: Optional[str] = Field(None, description="Detected intent for the session")

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

def initialize_ollama_template_manager():
    """Initialize the Ollama template manager"""
    global ollama_template_manager

    try:
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from site_manager import TemplateManager  # Replace with actual import

        auth_manager = AuthenticationManager()
        ollama_template_manager = TemplateManager(auth_manager)
        # Test the connection

        print("Ollama Template Manager initialized successfully")
        return True

    except Exception as e:
        print(f"Failed to initialize Ollama Template Manager: {e}")
        ollama_template_manager = None
        return False




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

import os
from dotenv import load_dotenv
from fastapi import HTTPException, status
import google.generativeai as genai

load_dotenv()

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI



def get_gemini_client(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    top_p: float = 0.95,
    api_key: str = None,
):
    """
    Initializes and returns a LangChain-wrapped Gemini client.

    Args:
        model (str): The Gemini model to use (e.g., "gemini-2.5-flash", "gemini-1.5-pro").
        temperature (float): Controls randomness (higher = more creative).
        max_output_tokens (int): Max tokens in output.
        top_p (float): Nucleus sampling value.
        api_key (str): Your Google API key. If None, expects GEMINI_API_KEY env variable.

    Returns:
        ChatGoogleGenerativeAI: LangChain-compatible Gemini client.
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("API key must be provided via argument or GEMINI_API_KEY env variable")

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        google_api_key=api_key,
        convert_system_message_to_human=True,  # ✅ important for tool/agent support
    )







import logging
import json


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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



# Simple session storage (keeping for backward compatibility)
chat_sessions = {}

def save_conversation_to_db(session_id: str, role: str, message: str, intent: str = None):
    """Save message to MongoDB"""
    try:
        conversations_collection.insert_one({
            "session_id": session_id,
            "role": role,
            "message": message,
            "timestamp": datetime.now(),
            "intent": intent
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

def get_session_intent(session_id):
    """Get the intent for a session from MongoDB"""
    try:
        # Find any document with the session_id that has an intent field
        session = conversations_collection.find_one(
            {"session_id": session_id, "intent": {"$exists": True}},
            sort=[("timestamp", -1)]  # Get the most recent one
        )
        print("session intent inside get_session_intent : ", session)
        return session['intent'] if session else None
    except Exception as e:
        logger.error(f"Failed to get intent from MongoDB: {e}")
        return None


def store_session_intent(session_id, session_intent):
    """Update the intent for all documents in a session"""
    try:
        # Update intent for all documents with the given session_id
        result = conversations_collection.update_many(
            {"session_id": session_id},  # Match all documents with this session_id
            {"$set": {"intent": session_intent, "timestamp": datetime.utcnow()}},
            upsert=False  # do NOT create new documents
        )

        if result.matched_count == 0:
            logger.warning(f"No documents found for session {session_id}, nothing updated")
            return False

        logger.info(f"Intent '{session_intent}' updated for {result.modified_count} documents in session {session_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to update intent in MongoDB: {e}")
        return False


def safe_extract_text(response):
    try:
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
        # fallback
        return ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool


import langchain
from langchain_core.tools import tool

def make_tools(session_id: str, intent: str | None = None):
    """Factory to create tools bound to a specific session_id and intent."""

    # @tool("detect_intent", return_direct=True)
    # async def detect_intent_tool(user_message: str) -> str:
    #     """Detects the user intent (CREATE_SITE, DELETE_USER, etc.)"""
    #     detected_intent = await execute_phase_0(session_id, user_message)
    #     store_session_intent(session_id, detected_intent)   # ✅ save new intent
    #     return detected_intent

    @tool("collect_data", return_direct=True)
    async def collect_data_tool(user_message: str) -> str:
        """Collects missing information from the user for the given intent"""
        current_intent = get_session_intent(session_id) or intent or "UNKNOWN"
        client = get_gemini_client(temperature=0.2)
        response = await execute_phase_1(session_id, user_message, client, current_intent)
        return response.message

    @tool("execute_operation", return_direct=True)
    async def execute_operation_tool(_: str = "") -> dict:
        """Converts collected data into JSON and executes the operation"""
        current_intent = get_session_intent(session_id) or intent or "UNKNOWN"
        response = await execute_phase_2(session_id, current_intent)
        return response.dict()

    return [ collect_data_tool, execute_operation_tool]











import langgraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage


        

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatRequest):
    session_id = chat_request.session_id or str(uuid.uuid4())
    user_message = chat_request.message.strip()
    intent = get_session_intent(session_id) or "UNKNOWN"

    try:
        # save user message
        save_conversation_to_db(session_id, "user", user_message, intent=intent)

        # cancel logic
        if any(trigger in user_message.lower() for trigger in ["cancel", "stop", "exit"]):
            clear_conversation_from_db(session_id)
            return ChatResponse(message="Cancelled", status="cancelled", session_id=session_id)

        # ✅ Initialize agent with session-specific tools
        tools = make_tools(session_id, intent)
        llm = get_gemini_client()

        # ✅ setup memory checkpointer
        memory = MemorySaver()
        agent_executor = create_react_agent(model=llm, tools=tools, checkpointer=memory)
        print("agent executor created")
        # ✅ include thread_id when invoking
        result: Dict[str, Any] = await agent_executor.ainvoke(
        {"messages": [HumanMessage(content=user_message)]},   # ✅ explicit HumanMessage
        config={"configurable": {"thread_id": session_id}},
        )

        print("resltu: ", result)
        ai_response = None
        if "output" in result:
            ai_response = result["output"]
        elif "messages" in result and result["messages"]:
            ai_response = result["messages"][-1].content


        save_conversation_to_db(session_id, "assistant", ai_response, intent=intent)

        return ChatResponse(
            message=ai_response,
            status="completed",
            session_id=session_id,
            context={"phase": "agent"},
            data=result,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            message=f"Sorry, something went wrong: {str(e)}",
            status="error",
            session_id=session_id,
            context={"error": str(e)},
            data={},
        )





# Function to get formatted site list


async def execute_phase_0(session_id: str, user_message: str, client=get_gemini_client(temperature=0.1)) -> str:
    """Phase 0: Optimized Intent detection - returns only the intent string"""

    # Get conversation history from MongoDB (optimized - only recent messages)
    db_messages = get_conversation_from_db(session_id)
    conversation_history = "\n".join([
    f"User: {msg['message']}"
    for msg in db_messages[-5:]  # Last 5 messages
    if msg['role'] == 'user'  # Only user messages
    ])
    
    # Valid intents list
    valid_intents = [
        "CREATE_SITE", "DELETE_SITE", "VIEW_SITES", 
        "ASSIGN_USERS_TO_SITE", "UNASSIGN_USERS_FROM_SITE", 
        "CREATE_USER", "DELETE_USER", "VIEW_USERS", 
        "VIEW_PERMISSION_SETS", "ASSIGN_PERMISSION_SET_TO_USER", 
        "UNASSIGN_PERMISSION_SET_FROM_USER","DETAIL_SPECIFIC_USER","DETAIL_SPECIFIC_SITE",
        "SHOW_ALL_TEMPLATES", "ASSIGN_TEMPLATE_TO_USER", "UNASSIGN_TEMPLATE_FROM_USER", "CREATE_TEMPLATE",
        "CREATE_A_GROUP","DELETE_A_GROUP","ADD_USER_TO_GROUP","REMOVE_USER_FROM_GROUP","VIEW_ALL_GROUPS","SHOW_USERS_ADDED_TO_GROUP","SHOW_USERS_ADDED_NOT_TO_GROUP",
        "DELETE_TEMPLATE","AUTOMATE_CUSTOMER_ACCESS_SETTING","UNKNOWN"
    ]
    
    # Improved intent detection prompt
    intent_prompt = f"""You are a PulsePro intent classifier.

ANALYZE the user message and conversation history to determine the user's intent. Focus on the most RECENT user message. that means the last message from the user in the conversation history.

USER MESSAGE: "{user_message}"

CONVERSATION HISTORY:
{conversation_history}

ONLY VALID INTENTS:
{valid_intents}


INTENTS:
- AUTOMATE_CUSTOMER_ACCESS_SETTING: "Auto assign new locations to all users" or "Switch off/autounassign/remove new locations to all users" or "Auto assign new templates to all users" or "Switch off/autounassign/remove new templates to all users"
- CREATE_TEMPLATE: create/add/make/new template/checklist/form
- SHOW_ALL_TEMPLATES: show/list/see/display all templates/checklist/forms/
- ASSIGN_TEMPLATE_TO_USER: assign/give/allot template/checklist/form to user/employee/staff/person/man or assign/give/allot user/employee/staff/person/man  to template/checklist/form
- UNASSIGN_TEMPLATE_FROM_USER: remove/take/unassign/revoke template/checklist/form from user/employee/staff/person/man or remove/take/unassign/revoke user/employee/staff/person/man from template/checklist/form
- VIEW_PERMISSION_SETS: show/list/see permissions/roles/access
- ASSIGN_PERMISSION_SET_TO_USER: give/assign permissions/roles/access to user/employee or give/assign user/employee permissions/roles/access
- UNASSIGN_PERMISSION_SET_FROM_USER: remove/take permissions/roles/access from user/employee or remove/take user/employee permissions/roles/access
- DELETE_TEMPLATE: delete/remove template/checklist/form
- CREATE_SITE: create/add/make/new/open site/office/location/branch
- DELETE_SITE: delete/remove/close site/office/location/branch
- VIEW_SITES: show/list/see/get all sites/offices/locations/branches
- ASSIGN_USERS_TO_SITE: assign/add/move user/employee to site/office/location or assign/add/move site/office/location to user/employee
- UNASSIGN_USERS_FROM_SITE: remove/take user/employee from site/office/location or remove/take site/office/location from user/employee
- CREATE_USER: create/add/make/new user/employee/account/person
- DELETE_USER: delete/remove user/employee/account/person
- VIEW_USERS: show/list/see/get all users/employees/accounts/people
- VIEW_ALL_GROUPS: state/show/list/see all the groups in the system
- CREATE_A_GROUP: create/add/make new group/cluster
- DELETE_A_GROUP: delete/remove a group
- ADD_USER_TO_GROUP: add/assign/allot user to Group/cluster
- REMOVE_USER_FROM_GROUP: remove/delete/take user from Group/cluster
- SHOW_USERS_ADDED_TO_GROUP: show/list/display users/member added/assigned to a group/cluster
- SHOW_USERS_ADDED_NOT_TO_GROUP: show/list/display users/member not added/assigned to a group/cluster
- DETAIL_SPECIFIC_USER: show/list/display  detail/more about/of a specific user/member
- DETAIL_SPECIFIC_SITE: show/list/display  detail/more about/of a specific site/location
- UNKNOWN: hello/hi/chat/help/other topics/ about the platform

RULES:
0. If message contains "auto" or "automaticcaly" + "assign" or "unassign" + "location/site" or "checklist/template" ->return :  AUTOMATE_CUSTOMER_ACCESS_SETTING
1. If message contains "permission/role/access" + "assign/give" → ASSIGN_PERMISSION_SET_TO_USER
2. If message contains "permission/role/access" + "remove/revoke" → UNASSIGN_PERMISSION_SET_FROM_USER
3. If message contains "user" + "to" + "site/office" → ASSIGN_USERS_TO_SITE
4. If message contains "show","display","list","see"+ "template","forms" → SHOW_ALL_TEMPLATES
5. If message contains "remove" + "template" → DELETE_TEMPLATE
6. Look for key action words: create, delete, view, assign, remove
7. If unsure, return UNKNOWN
CRITICAL: Return ONLY the intent name from the list of valid intents (e.g., "CREATE_SITE" or "UNKNOWN"). No explanations, no other text.
"""

    try:
        # Get response from LLM with optimized settings
        response = client.invoke([HumanMessage(content=intent_prompt)])
        
        detected_intent = (response.content or "").strip().upper()
        print(f"Detected intent: {detected_intent}")
        
        # Validate intent and return
        return detected_intent if detected_intent in valid_intents else "UNKNOWN"
        
    except Exception as e:
        print(f"Error in intent detection: {e}")
        return "UNKNOWN"



import prompt
# Updated execute_phase_1 function
async def execute_phase_1(session_id: str, user_message: str, client,intent:str) -> ChatResponse:
    """Phase 1: Normal chat - Intent detection and data collection"""
    # Get conversation history from MongoDB
    db_messages = get_conversation_from_db(session_id)
    conversation_history = ""
    print("db messages: ",db_messages)
    for msg in db_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_history += f"{role}: {msg['message']}\n"

    # Get current sites list
    # sites_list = get_sites_list_formatted()
    # print(f"Available sites: {sites_list}")

    # all_user_list = ollama_site_manager.get_all_users()
    # permission_set_list = ollama_permission_manager.get_all_permission_sets()
    print("getting prompt")
    new_prompt=prompt.get_data_collection_prompt(intent)
    # Format the prompt with sites list
    # formatted_prompt = PHASE_1_PROMPT.format(all_sites_list=sites_list,all_users_list=all_user_list,all_permission_sets_list=permission_set_list)
    
    # Create full prompt for Phase 1
    full_prompt = f"""
You are PulsePro AI Assistant.

====================CORE RULES====================
• Handle ONLY PulsePro operations as specified for this task
• Ignore unrelated queries. Reply: "I can only help with PulsePro operations."
• Ask for missing information. Never assume values.
• If user says cancel/stop/exit/abort/halt/quit/terminate/end, reply: "Operation cancelled. No action taken."
• When you have all required data, ask: "I have all the information needed. Type 'Proceed' to execute this operation."
• Must always respond in a structural and concise manner. Use bullet points or numbered lists for clarity. Highlight the main heading. Give spaces and line breaks for readability.
• Use simple, non-technical language. Avoid jargon.

====================CONVERSATION HISTORY====================
{conversation_history}

====================Follow the instructions below and keep the CONVERSATION HISTORY in mind====================

{new_prompt}

====================CURRENT USER MESSAGE====================
{user_message}
====================YOUR RESPONSE===================="""
    print("New prompt: ",full_prompt)

    print("into llm now")
    # Get response from LLM
    response =  client.invoke([HumanMessage(content=full_prompt)])
    print("response from llm: ",response)
    ai_response = (response.content or "").strip() or "UNKNOWN"

    
    # Save AI response to MongoDB
    save_conversation_to_db(session_id, "assistant", ai_response, intent=intent)
    
    # Determine status
    if "Type 'Proceed' to execute".lower() in ai_response.lower():
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
            "intent": intent},
        data={}
    )

async def execute_phase_2(session_id: str,intent:str, client=get_gemini_client(temperature=0.05)) -> ChatResponse:
    """Phase 2: Generate JSON and execute operation"""
    
    try:
        # Get conversation history from MongoDB for Phase 2
        db_messages = get_conversation_from_db(session_id)
        conversation_history = ""
        for msg in db_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {msg['message']}\n"
        
        new_prompt=prompt.get_json_response_prompt(intent)
        # Create Phase 2 prompt
        phase_2_prompt = f"""{new_prompt}

====================CONVERSATION HISTORY====================
{conversation_history}

     JSON:"""
        print("Phase 2 prompt: ",phase_2_prompt)
        
        # Get JSON response from LLM
        response = client.invoke([HumanMessage(content=phase_2_prompt)])
        
        json_response = (response.content or "").strip()
        
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
        execution_result = await execute_site_operation(operation_data,session_id)
        
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

async def execute_site_operation(operation_data: dict,session_id:str) -> dict:
    """Execute the site operation based on JSON data"""
    
    try:
        initialize_ollama_site_manager()
        initialize_ollama_user_manager()
        initialize_ollama_permission_manager()
        initialize_ollama_template_manager()
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
                "message": f"✅ Site **{location_name}** created successfully!",
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
                result=ollama_site_manager.delete_site(site_id)    

                if isinstance(result, dict) and not result.get("success", True):
                    return {
                "success": False,
                "message": f"❌ Failed to delete site **{location_name}**: {result.get('message')}",
                "data": {"error": result.get("message"), "site_id": site_id}
                }   

                return {
                    "success": True,
                    "message": f"✅ Site **{location_name}** deleted successfully!",
                    "data": {"deleted": True, "site_id": site_id}
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Site **{location_name}** not found",
                    "data": {"error": "Site not found"}
                }
        
        elif operation_type == "VIEW_SITES":
            result = ollama_site_manager.get_all_sites()
            sites = result.get('locations', [])
            if sites:
                site_list = "\n".join([f"• **{site.get('location_name', 'Unknown')}**,  **{site.get('city')}**,  **{site.get('state')}**" for site in sites])
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
                    "message": f"❌ Site **{location_name}** not found",
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
                "message": f"✅ Users assigned to site **{location_name}** successfully!",
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
                    "message": f"❌ Site **{location_name}** not found",
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
                "message": f"✅ Users unassigned from site **{location_name}** successfully!",
                "data": {"user_ids": user_ids}
            }
         
        elif operation_type == "CREATE_USER":
            first_name = data.get("first_name")
            last_name = data.get("last_name")
            email = data.get("email")

            permission_set_id = ollama_permission_manager.get_permission_set_id_by_name(data.get("permission_set"))
            if not permission_set_id:
                return {
                    "success": False,
                    "message": f"❌ Permission set **{data.get('permission_set')}** not found",
                    "data": {"error": "Permission set not found"}
                }
            
            result = ollama_user_manager.create_user(first_name, last_name, email, permission_set_ids=permission_set_id)
            return {
                "success": True,
                "message": f"✅ User **{first_name} {last_name}** created successfully!",
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
                    "message": f"✅ User **{full_name}** deleted successfully!",
                    "data": {"user_id": id}
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ User **{full_name}** not found",
                    "data": {"error": "User not found"}
                }
            
        elif operation_type == "VIEW_USERS":
            users = ollama_user_manager.get_all_users() or []   # directly get list

            if users:
                user_list = "\n".join([
                    f"• {user.get('name', '')} (ID: {user.get('id', '')})"
                    for user in users
                ])
                message = f"👤 Found {len(users)} users:\n{user_list}"
            else:
                message = "👤 No users found"

            return {
                    "success": True,
                    "message": message,
                    "data": {"users": users}   # wrap list in a dict for consistency
            }

        
        elif operation_type == "VIEW_PERMISSION_SETS":
            print("in permission sets")
            result = ollama_permission_manager.get_all_permission_sets()
            print("result: ",result)
            if result:
                ps_list = "\n".join([f"• **{ps.get('name', 'Unknown')}**" for ps in result])
                message = f"🔑 Found {len(result)} permission sets:\n{ps_list}"
            else:
                message = "🔑 No permission sets found"
            
            return {
                "success": True,
                "message": message,
                "data": {"permission_sets": result} 
            }
        
        elif operation_type in ["ASSIGN_PERMISSION_SET_TO_USER", "UNASSIGN_PERMISSION_SET_FROM_USER"]:
            full_name = data.get("full_name")
            permission_sets = data.get("permission_set", [])
            
            # Find user ID
            user_id = ollama_user_manager.get_user_by_name(full_name)
            if not user_id:
                return {
                    "success": False,
                    "message": f"❌ User **{full_name}** not found",
                    "data": {"error": "User not found"}
                }
            
            # Get permission set IDs
            permission_set_ids = []
            for ps_name in permission_sets:
                ps_id = ollama_permission_manager.get_permission_set_id_by_name(ps_name)
                if ps_id:
                    permission_set_ids.append(ps_id)
            
            if not permission_set_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified permission sets were found: {', '.join(permission_sets)}",
                    "data": {"error": "Permission sets not found"}
                }
            
            if operation_type == "ASSIGN_PERMISSION_SET_TO_USER":
                result = ollama_permission_manager.assign_permission_sets_to_user(user_id, permission_set_ids)
                action = "assigned to"
            else:
                result = ollama_permission_manager.unassign_permission_sets_from_user(user_id, permission_set_ids)
                action = "unassigned from"
            
            return {
                "success": True,
                "message": f"✅ Permission sets **{action}** user **{full_name}** successfully!",
                "data": {"permission_set_ids": permission_set_ids}
            }

        elif operation_type == "SHOW_ALL_TEMPLATES":
            templates = ollama_template_manager.get_all_templates() or []   # directly get list

            if templates:
                template_list = "\n".join([
                    f"• {template.get('name', '')} (By : {template.get('created_by', '')})"
                    for template in templates
                ])
                message = f"📄 Found {len(templates)} templates:\n{template_list}"
            else:
                message = "📄 No templates found"

            return {
                "success": True,
                "message": message,
                "data": {"templates": templates}   # wrap list in a dict for consistency
            }

        elif operation_type == "DELETE_TEMPLATE":
            template_name = data.get("template_name")

            # Find template ID
            template_id = ollama_template_manager.get_template_id_by_name(template_name)
            print("Template ID: ", template_id)
            if not template_id:
                return {
                    "success": False,
                    "message": f"❌ Template **{template_name}** not found",
                    "data": {"error": "Template not found"}
                }

            result = ollama_template_manager.delete_template(template_id)
            print("Delete Template Result: ", result)
            return {
                "success": True,
                "message": f"✅ Template **{template_name}** deleted successfully!",
                "data": {"template_id": template_id}
            }
        
        elif operation_type == "ASSIGN_TEMPLATE_TO_USER":
            full_name = data.get("user_name")
            template_names = data.get("template_name", [])

            # Find user ID
            user_id = ollama_user_manager.get_user_by_name(full_name)
            print("user ID: ", user_id)
            if not user_id:
                return {
                    "success": False,
                    "message": f"❌ User **{full_name}** not found",
                    "data": {"error": "User not found"}
                }

            # Get template IDs
            template_ids = []
            for template_name in template_names:
                template_id = ollama_template_manager.get_assign_user_template_id_by_name(template_name,user_id)
                if template_id:
                    template_ids.append(template_id)
            print("template ids: ", template_ids)
            if not template_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified templates were found: {', '.join(template_names)}",
                    "data": {"error": "Templates not found"}
                }

            result = ollama_template_manager.assign_templates_to_user(user_id, template_ids)
            return {
                "success": True,
                "message": f"✅ Templates assigned to user **{full_name}** successfully!",
                "data": {"template_ids": template_ids}
            }

        elif operation_type == "UNASSIGN_TEMPLATE_FROM_USER":
            full_name = data.get("user_name")
            template_names = data.get("template_name", [])

            # Find user ID
            user_id = ollama_user_manager.get_user_by_name(full_name)
            if not user_id:
                return {
                    "success": False,
                    "message": f"❌ User **{full_name}** not found",
                    "data": {"error": "User not found"}
                }

            # Get template IDs
            template_ids = []
            for template_name in template_names:
                template_id = ollama_template_manager.get_assign_user_template_id_by_name(template_name,user_id)
                if template_id:
                    template_ids.append(template_id)

            if not template_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified templates were found: {', '.join(template_names)}",
                    "data": {"error": "Templates not found"}
                }

            result = ollama_template_manager.unassign_templates_from_user(user_id, template_ids)
            return {
                "success": True,
                "message": f"✅ Templates unassigned from user **{full_name}** successfully!",
                "data": {"template_ids": template_ids}
            }
        
        elif operation_type == "CREATE_TEMPLATE":
            template_name = data.get("template_name")
            template_id= ollama_template_manager.get_checklist_id_by_name(template_name)

            if not template_id:
                return {
                    "success": False,
                    "message": f"❌ Template **{template_name}** does not exists. PLease select the right template name",
                    "data": {"error": "Template already exists"}
                }
            
            result=ollama_template_manager.create_checklist(template_id)
            return{
                "success": True,
                "message": f"✅ Template **{template_name}** created successfully!",
                "data": result
            }
        
        elif operation_type == "AUTOMATE_CUSTOMER_ACCESS_SETTING":
            previous_access = ollama_user_manager.get_customer_access_settings()
            print("previous access: ", previous_access)

            print("accessToAllSite",field_exists(data,"accessToAllSite"))
            print("accessToAllChecklist",field_exists(data,"accessToAllChecklist"))

            if(field_exists(data,"accessToAllSite") and field_exists(data,"accessToAllChecklist")):
                print("1st Phase")
                result=ollama_user_manager.update_customer_setting(accessToAllSite=data.get("accessToAllSite"), accessToAllChecklist=data.get("accessToAllChecklist"))
                return{
                "success": True,
                "message": f"✅ Operation on location and checklist was successful !",
                "data": result
            }
            elif(field_exists(data,"accessToAllSite")):
                print("2nd Phase")
                result= ollama_user_manager.update_customer_setting(accessToAllSite=data.get("accessToAllSite"), accessToAllChecklist=previous_access.get("accessToAllChecklist"))
                return{
                "success": True,
                "message": f"✅ Operation on location was successful !",
                "data": result
            }
            elif(field_exists(data,"accessToAllChecklist")):
                print("3rd phase")
                result= ollama_user_manager.update_customer_setting(accessToAllSite=previous_access.get("accessToAllSite"), accessToAllChecklist=data.get("accessToAllChecklist"))
                return{
                "success": True,
                "message": f"✅ Operation on checklist was successful !",
                "data": result
            }

            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to perform it, try later",
                    "data": {"error": "Some internal error"}
                }
            
        elif operation_type=="VIEW_ALL_GROUPS":
            all_groups_list= ollama_user_manager.get_all_groups()
            groups_formatted="\n".join(
                f"{idx+1}. **{group['name']}**"
                for idx,group in enumerate(all_groups_list)
                )if all_groups_list else "Not Available"

            message = f"📍 Found {len(all_groups_list)} Groups:\n{groups_formatted}"

            return{
                "success":True,
                "message": message,
                "data": {"groups_list": groups_formatted}
            }
        
        elif operation_type=="DELETE_A_GROUP":
            group_name=data.get("group_name")

            group_id=ollama_user_manager.get_group_id_by_name(group_name)

            if not group_id:
                return{
                    "success":False,
                    "message":f"❌ Group **{group_name}** not found",
                    "data":{"error":"Group not found"}
                }
            print("group_id: ", group_id)
            result=ollama_user_manager.delete_group(group_id)

            return{
                "success":True,
                "message":f"✅  Group **{group_name}** deleted successfully",
                "data":{"group_id": group_id}

            }
        
        elif operation_type=="CREATE_A_GROUP":
            group_name=data.get("group_name")
            print("group name: ",group_name)


            result=ollama_user_manager.create_a_group(group_name)
            print("result : ", result)

            return{
                "success":True,
                "message":f"✅  Group **{group_name}** created successfully",
                "data":{"group_id": group_name}

            }

        elif operation_type=="ADD_USER_TO_GROUP":
            group_name=data.get("group_name")
            user_names=data.get("user_names")

            group_id=ollama_user_manager.get_group_id_by_name(group_name)
            userIds=[]

            for user_name in user_names:
                userIds.append(ollama_user_manager.get_user_by_name(user_name))

            if not userIds:
                return{
                    "success":False,
                    "message":f"User not found",
                    "data":{"error": "user not found"}
                }
            if not group_id:
                return{
                    "success":False,
                    "message":f"Group not found",
                    "data":{"error":"group not found"}
                }

            result=ollama_user_manager.add_multiple_user_to_group(userIds=userIds,groupId=group_id)
            return{
                "success":True,
                "message":f"✅  Users **{user_names}** assigned successfully to **{group_name}**",
                "data":{"group_name": group_name}

            }
        
        elif operation_type=="REMOVE_USER_FROM_GROUP":
            group_name=data.get("group_name")
            user_names=data.get("user_names")
            print("user_names: ",user_names)

            group_id=ollama_user_manager.get_group_id_by_name(group_name)
            print("group id: " ,group_id)
            userIds=[]

            result_user_list=ollama_user_manager.get_users_added_to_group(group_id)
            print("result_user_list : ", result_user_list)
            for user in result_user_list:
                if user_names and user.get("member_name") in user_names: 
                    userIds.append(user.get("id")) 
                
            print("user ids: ",userIds )
            if not userIds:
                return{
                    "success":False,
                    "message":f"User not found",
                    "data":{"error": "user not found"}
                }
            if not group_id:
                return{
                    "success":False,
                    "message":f"Group not found",
                    "data":{"error":"group not found"}
                }

            result=ollama_user_manager.delete_multiple_group_user(group_user_ids=userIds)
            return{
                "success":True,
                "message":f"✅  Users **{user_names}** removed successfully from **{group_name}**",
                "data":{"group_name": group_name}

            }            

        elif operation_type=="SHOW_USERS_ADDED_TO_GROUP":
            group_name=data.get("group_name")
            group_id=ollama_user_manager.get_group_id_by_name(group_name)

            result=ollama_user_manager.get_users_added_to_group(groupId=group_id)

            result_formatted="\n".join(
            f"{idx+1}.{res['member_name']}"
            for idx,res in enumerate(result)
            )

            return{
                "success":True,
                "message":f"Found {len(result)} assigned users for the group {group_name} \n {result_formatted}",
                "data":{"group name":group_name }
            }

        elif operation_type=="SHOW_USERS_ADDED_NOT_TO_GROUP":
            group_name=data.get("group_name")
            group_id=ollama_user_manager.get_group_id_by_name(group_name)

            result=ollama_user_manager.get_all_members_not_added_to_group(group_id=group_id)

            result_formatted="\n".join(
            f"{idx+1}.{res['name']}"
            for idx,res in enumerate(result)
            )

            return{
                "success":True,
                "message":f"Found {len(result)} non assigned users for the group {group_name} \n {result_formatted}",
                "data":{"group name":group_name }
            }
        
        elif operation_type=="DETAIL_SPECIFIC_USER":
            all_users=ollama_user_manager.get_all_users()
            print("all_users: ", all_users)
            selected_user=data.get("user_name")
            print("data: ",data)

            if not selected_user:
                return{
                "success": False,
                "message": f"❌ Failed operation: {operation_type}, Unknown user",
                "data": {"error": "Failed operation"}
                }

            for user in all_users:
                if user.get("name").lower()==selected_user.lower():
                    email=user.get("email")
                    permission=user.get("permission_set")
                    id=user.get("id")
                    name=user.get("name")
                    

            return{
                "success":True,
                "message":f" Here is the detail of the User:\n User : **{name}** (ID: **{id}**)\n Email: **{email}**\n Permissions: **{permission}**\n ",
                "data":{"user_id": id}
            }

        elif operation_type=="DETAIL_SPECIFIC_SITE":
            all_sites=ollama_site_manager.get_all_sites().get("locations")
            print("all_sites: ", all_sites)
            selected_site=data.get("site_name")
            print("data: ",data)

            if not selected_site:
                return{
                "success": False,
                "message": f"❌ Failed operation: {operation_type}, Unknown site",
                "data": {"error": "Failed operation"}
                }
            site_name=selected_site
            id=0
            city="UNKNOWN"
            state="UNKNOWN"
            country="UNKNOWN"
            geo_fencing_enabled=False
            geo_fencing_distance=0.0
            custom_field_data=[]
            full_address="UNKNOWN"
            pincode=826001
            mobile=1234567891

            for site in all_sites:
                if site.get("location_name").lower()==selected_site.lower():
                    city=site.get("city")
                    state=site.get("state")
                    id=site.get("id")
                    site_name=site.get("location_name")
                    full_address=site.get("full_address")
                    pincode=site.get("pincode") 
                    mobile=site.get("mobile")
                    country= site.get("country")
                    geo_fencing_enabled=site.get("geo_fencing_enabled")
                    geo_fencing_distance=site.get("geo_fencing_distance")
                    custom_field_data=site.get("custom_field_meta")

                    

            return{
                "success":True,
                "message":f" Here is the detail of the SIte:\n Site : **{site_name}** (ID: **{id}**)\n Address: **{full_address}**, PIN: **{pincode}**\n City: **{city}**, State: **{state}**, Country: **{country}**\n MOBILE: **{mobile}**\n Geofencing Enabled: **{geo_fencing_enabled}**, Geofencing Distance: **{geo_fencing_distance}**\n Custom Field Data:**{custom_field_data}** ",
                "data":{"site_id": id}
            }


        else:
            clear_conversation_from_db(session_id)
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
    


def field_exists(response: dict, field: str) -> bool:
    "checks if field exist in response"
    print("inside field_exist , response:", response)
    if not isinstance(response, dict):
        return False
    return field in response



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


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": "/docs"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please check logs for details"}


