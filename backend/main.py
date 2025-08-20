"""
main.py
FastAPI Service Layer with Llama Integration
"""

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
from ai_agent import LlamaIntegratedSiteManager

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class AuthRequest(BaseModel):
    refresh_token: str = Field(..., description="JWT refresh token")

class SiteCreateRequest(BaseModel):
    location_name: str = Field(..., description="Name of the location")
    address_field1: str = Field(..., description="Primary address")
    address_field2: Optional[str] = Field("", description="Secondary address")
    country_id: int = Field(1, description="Country ID")
    state_id: int = Field(..., description="State ID")
    city_id: int = Field(..., description="City ID")
    pincode: Optional[str] = Field("", description="Postal code")
    mobile: Optional[str] = Field("", description="Mobile number")
    location_number: Optional[str] = Field("", description="Location number")
    location_code: Optional[str] = Field("", description="Location code")
    to_email: Optional[str] = Field("", description="Primary email")
    cc_email: Optional[str] = Field("", description="CC email")
    reporting_timezone: str = Field("UTC", description="Timezone for reporting")
    geo_fencing_enabled: bool = Field(False, description="Enable geo-fencing")
    geo_fencing_distance: int = Field(0, description="Geo-fencing distance")
    lat: float = Field(0.0, description="Latitude")
    lng: float = Field(0.0, description="Longitude")
    map_link: Optional[str] = Field("", description="Map link")
    has_custom_field: bool = Field(False, description="Has custom fields")
    is_schedule_active: bool = Field(False, description="Schedule active")

class SiteUpdateRequest(SiteCreateRequest):
    pass

class UserAssignmentRequest(BaseModel):
    location_id: int = Field(..., description="Location ID")
    user_ids: List[int] = Field(..., description="List of user IDs to assign")

class UserUnassignmentRequest(BaseModel):
    mapped_location_ids: List[int] = Field(..., description="List of mapped location IDs to remove")

class AIRequest(BaseModel):
    query: str = Field(..., description="Natural language query for the AI agent")

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

class ConversationHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]

class LlamaConfig(BaseModel):
    model_name: str = Field(default="meta-llama/Llama-2-8b-chat-hf", description="Llama model name")
    device: str = Field(default="auto", description="Device to run model on")
    temperature: float = Field(default=0.7, description="Generation temperature")
    max_length: int = Field(default=512, description="Maximum generation length")

# Global variables
llama_site_manager = None
llama_config = LlamaConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting PulsePro Site Management API with Llama Integration...")
    print(f"Llama model: {llama_config.model_name}")
    yield
    # Shutdown
    print("Shutting down PulsePro Site Management API...")

app = FastAPI(
    title="PulsePro Site Management API with Llama AI",
    description="AI-powered conversational site management system using Llama 8B",
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

def get_llama_site_manager():
    """Dependency to get the Llama-integrated site manager"""
    global llama_site_manager
    
    if not llama_site_manager:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="System not initialized. Please authenticate first."
        )
    return llama_site_manager

# Enhanced Authentication with Llama Integration
@app.post("/auth/initialize", response_model=StandardResponse)
async def initialize_system_with_llama(auth_request: AuthRequest, config: LlamaConfig = Body(default=LlamaConfig())):
    """Initialize the system with refresh token and Llama model"""
    global llama_site_manager, llama_config
    
    try:
        # Update Llama configuration
        llama_config = config
        
        # Initialize authentication
        auth_manager = AuthenticationManager()
        auth_manager.set_refresh_token(auth_request.refresh_token)
        
        # Test authentication
        auth_manager.refresh_access_token()
        
        # Initialize Llama-integrated site manager
        llama_site_manager = LlamaIntegratedSiteManager(
            auth_manager, 
            model_name=llama_config.model_name
        )
        
        return StandardResponse(
            success=True,
            message="System initialized successfully with Llama AI integration",
            data={
                "llama_model": llama_config.model_name,
                "device": llama_config.device,
                "features": ["conversational_ai", "natural_language_processing", "multi_turn_conversations"]
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}"
        )

# Conversational AI Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(chat_request: ChatRequest, manager=Depends(get_llama_site_manager)):
    """Main conversational interface with Llama AI"""
    
    # Generate session ID if not provided
    session_id = chat_request.session_id or str(uuid.uuid4())
    
    try:
        response = manager.chat(
            message=chat_request.message,
            session_id=session_id,
            user_id=chat_request.user_id
        )
        
        return ChatResponse(
            message=response.get("message", ""),
            status=response.get("status", "unknown"),
            session_id=session_id,
            context=response.get("context"),
            data=response.get("data")
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )

@app.get("/chat/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str, manager=Depends(get_llama_site_manager)):
    """Get conversation history for a session"""
    
    try:
        history = manager.get_conversation_history(session_id)
        
        return ConversationHistoryResponse(
            session_id=session_id,
            history=history
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )

@app.delete("/chat/history/{session_id}", response_model=StandardResponse)
async def clear_conversation_history(session_id: str, manager=Depends(get_llama_site_manager)):
    """Clear conversation history for a session"""
    
    try:
        manager.clear_conversation(session_id)
        
        return StandardResponse(
            success=True,
            message=f"Conversation history cleared for session {session_id}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation history: {str(e)}"
        )

@app.post("/chat/sessions", response_model=StandardResponse)
async def create_new_session():
    """Create a new chat session"""
    
    session_id = str(uuid.uuid4())
    
    return StandardResponse(
        success=True,
        message="New chat session created",
        data={"session_id": session_id}
    )

# Site CRUD Operations
@app.post("/sites", response_model=StandardResponse)
async def create_site(site_request: SiteCreateRequest, manager=Depends(get_llama_site_manager)):
    """Create a new site"""
    try:
        # Convert request to SiteData
        site_data = SiteData(**site_request.dict())
        result = manager.create_site(site_data)
        
        return StandardResponse(
            success=True,
            message="Site created successfully",
            data=result
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create site: {str(e)}"
        )

@app.get("/sites", response_model=StandardResponse)
async def get_all_sites(manager=Depends(get_llama_site_manager)):
    """Get all sites"""
    try:
        result = manager.get_all_sites()
        return StandardResponse(
            success=True,
            message="Sites retrieved successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sites: {str(e)}"
        )

@app.get("/sites/{location_id}", response_model=StandardResponse)
async def get_site_by_id(location_id: int, manager=Depends(get_llama_site_manager)):
    """Get a specific site by ID"""
    try:
        result = manager.get_site_by_id(location_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Site with ID {location_id} not found"
            )
        
        return StandardResponse(
            success=True,
            message="Site retrieved successfully",
            data=result
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve site: {str(e)}"
        )

@app.put("/sites/{location_id}", response_model=StandardResponse)
async def update_site(location_id: int, site_request: SiteUpdateRequest, manager=Depends(get_llama_site_manager)):
    """Update an existing site"""
    try:
        # Convert request to SiteData
        site_data = SiteData(**site_request.dict())
        result = manager.update_site(location_id, site_data)
        
        return StandardResponse(
            success=True,
            message="Site updated successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update site: {str(e)}"
        )

@app.delete("/sites/{location_id}", response_model=StandardResponse)
async def delete_site(location_id: int, manager=Depends(get_llama_site_manager)):
    """Delete a site"""
    try:
        result = manager.delete_site(location_id)
        return StandardResponse(
            success=True,
            message="Site deleted successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete site: {str(e)}"
        )

# User Assignment Operations
@app.get("/sites/{location_id}/users", response_model=StandardResponse)
async def get_site_users(location_id: int, search_keyword: str = "", manager=Depends(get_llama_site_manager)):
    """Get all users available for assignment to a site"""
    try:
        result = manager.get_site_users(location_id, search_keyword)
        return StandardResponse(
            success=True,
            message="Users retrieved successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve users: {str(e)}"
        )

@app.post("/sites/assign-users", response_model=StandardResponse)
async def assign_users_to_site(assignment_request: UserAssignmentRequest, manager=Depends(get_llama_site_manager)):
    """Assign multiple users to a site"""
    try:
        result = manager.assign_users_to_site(
            assignment_request.location_id,
            assignment_request.user_ids
        )
        
        return StandardResponse(
            success=True,
            message="Users assigned successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign users: {str(e)}"
        )

@app.post("/sites/unassign-users", response_model=StandardResponse)
async def unassign_users_from_site(unassignment_request: UserUnassignmentRequest, manager=Depends(get_llama_site_manager)):
    """Unassign users from a site"""
    try:
        result = manager.unassign_users_from_site(unassignment_request.mapped_location_ids)
        
        return StandardResponse(
            success=True,
            message="Users unassigned successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unassign users: {str(e)}"
        )

# Legacy AI endpoint for backward compatibility
@app.post("/ai/query", response_model=StandardResponse)
async def process_ai_query_legacy(ai_request: AIRequest, manager=Depends(get_llama_site_manager)):
    """Legacy AI query endpoint (redirects to conversational interface)"""
    
    # Generate a temporary session for legacy requests
    session_id = f"legacy_{uuid.uuid4()}"
    
    try:
        response = manager.chat(
            message=ai_request.query,
            session_id=session_id,
            user_id="legacy_user"
        )
        
        return StandardResponse(
            success=response.get('status') != 'error',
            message=response.get('message', 'Query processed'),
            data=response.get('data')
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process AI query: {str(e)}"
        )

# Chat Interface for Web UI
@app.get("/chat/interface")
async def get_chat_interface():
    """Serve a simple web chat interface"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PulsePro AI Chat</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .chat-container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .chat-header { background: #2196f3; color: white; padding: 20px; border-radius: 10px 10px 0 0; }
            .chat-messages { height: 500px; overflow-y: auto; padding: 20px; border-bottom: 1px solid #eee; }
            .message { margin: 15px 0; padding: 15px; border-radius: 10px; max-width: 80%; }
            .user-message { background-color: #e3f2fd; margin-left: auto; text-align: right; }
            .ai-message { background-color: #f3e5f5; margin-right: auto; }
            .input-container { display: flex; gap: 10px; padding: 20px; }
            .message-input { flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 25px; outline: none; }
            .send-button { padding: 15px 30px; background-color: #2196f3; color: white; border: none; border-radius: 25px; cursor: pointer; }
            .send-button:hover { background-color: #1976d2; }
            .status { padding: 10px 20px; color: #666; font-style: italic; }
            .typing { color: #999; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 PulsePro AI Assistant</h1>
                <p>Conversational site management with Llama AI</p>
            </div>
            <div id="chat-messages" class="chat-messages"></div>
            <div class="input-container">
                <input type="text" id="message-input" class="message-input" placeholder="Type your message here..." />
                <button onclick="sendMessage()" class="send-button">Send</button>
            </div>
            <div id="status" class="status">Ready to chat! Try: "I want to create a new site"</div>
        </div>

        <script>
            let sessionId = null;
            
            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Display user message
                addMessage(message, 'user');
                input.value = '';
                
                // Show typing indicator
                showTyping();
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: message,
                            session_id: sessionId
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (!sessionId) {
                        sessionId = data.session_id;
                    }
                    
                    // Remove typing indicator and display AI response
                    removeTyping();
                    addMessage(data.message, 'ai');
                    
                    // Update status
                    document.getElementById('status').textContent = `Status: ${data.status}`;
                    
                } catch (error) {
                    removeTyping();
                    addMessage('Sorry, I encountered an error. Please try again.', 'ai');
                    document.getElementById('status').textContent = 'Error occurred';
                }
            }
            
            function addMessage(text, sender) {
                const messagesDiv = document.getElementById('chat-messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                messageDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'AI Assistant'}:</strong><br>${text.replace(/\\n/g, '<br>')}`;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function showTyping() {
                const messagesDiv = document.getElementById('chat-messages');
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message ai-message typing';
                typingDiv.id = 'typing-indicator';
                typingDiv.innerHTML = '<strong>AI Assistant:</strong><br>Thinking...';
                messagesDiv.appendChild(typingDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function removeTyping() {
                const typingIndicator = document.getElementById('typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }
            
            // Allow Enter key to send message
            document.getElementById('message-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Initial greeting
            addMessage('Hello! I\\'m your AI assistant for site management. I can help you create, update, delete, and search sites using natural language. What would you like to do?', 'ai');
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# Enhanced health check
@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with AI model status"""
    
    global llama_site_manager
    
    health_status = {
        "status": "healthy",
        "service": "PulsePro Site Management API with Llama AI",
        "components": {
            "api": "healthy",
            "authentication": "healthy" if llama_site_manager else "not_initialized",
            "llama_model": "loaded" if (llama_site_manager and llama_site_manager.conversational_agent.generator) else "fallback_mode",
            "conversational_ai": "available" if llama_site_manager else "not_available"
        }
    }
    
    return health_status

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PulsePro Site Management API with Llama AI",
        "version": "2.0.0",
        "description": "Conversational AI-powered site management system using Llama 8B",
        "features": [
            "Multi-turn conversations",
            "Natural language processing", 
            "Site CRUD operations",
            "User management",
            "Analytics and reporting"
        ],
        "endpoints": {
            "chat": "/chat (POST) - Main conversational interface",
            "auth": "/auth/initialize (POST)",
            "sites": "/sites (GET, POST), /sites/{id} (GET, PUT, DELETE)",
            "users": "/sites/{id}/users, /sites/assign-users, /sites/unassign-users",
            "interface": "/chat/interface (GET) - Web chat interface",
            "docs": "/docs - Interactive API documentation"
        }
    }

# Analytics endpoint
@app.get("/analytics/sites-summary", response_model=StandardResponse)
async def get_sites_summary(manager=Depends(get_llama_site_manager)):
    """Get summary analytics for sites"""
    try:
        all_sites = manager.get_all_sites()
        sites = all_sites.get('locations', [])
        
        # Calculate analytics
        total_sites = len(sites)
        sites_by_country = {}
        sites_by_state = {}
        sites_with_geo_fencing = 0
        sites_with_mobile = 0
        sites_with_email = 0
        
        for site in sites:
            # Country distribution
            country = site.get('country', 'Unknown')
            sites_by_country[country] = sites_by_country.get(country, 0) + 1
            
            # State distribution
            state = site.get('state', 'Unknown')
            sites_by_state[state] = sites_by_state.get(state, 0) + 1
            
            # Feature usage
            if site.get('geo_fencing_enabled'):
                sites_with_geo_fencing += 1
            
            if site.get('mobile'):
                sites_with_mobile += 1
            
            if site.get('to_email'):
                sites_with_email += 1
        
        summary = {
            "total_sites": total_sites,
            "distribution": {
                "by_country": sites_by_country,
                "by_state": sites_by_state
            },
            "features": {
                "geo_fencing_enabled": sites_with_geo_fencing,
                "sites_with_mobile": sites_with_mobile,
                "sites_with_email": sites_with_email,
                "completion_rate": {
                    "mobile": round((sites_with_mobile / total_sites) * 100, 2) if total_sites > 0 else 0,
                    "email": round((sites_with_email / total_sites) * 100, 2) if total_sites > 0 else 0,
                    "geo_fencing": round((sites_with_geo_fencing / total_sites) * 100, 2) if total_sites > 0 else 0
                }
            }
        }
        
        return StandardResponse(
            success=True,
            message="Sites summary generated successfully",
            data=summary
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )

# AI capabilities endpoint
@app.get("/ai/capabilities", response_model=StandardResponse)
async def get_ai_capabilities():
    """Get AI agent capabilities and supported operations"""
    
    capabilities = {
        "conversational_features": [
            "Multi-turn conversations",
            "Intent recognition",
            "Information gathering",
            "Confirmation dialogs",
            "Natural language understanding"
        ],
        "supported_operations": [
            "create_site",
            "update_site", 
            "delete_site",
            "show_sites",
            "search_sites",
            "assign_users",
            "unassign_users",
            "get_analytics"
        ],
        "conversation_states": [
            "idle",
            "intent_recognition", 
            "gathering_info",
            "confirmation",
            "executing",
            "completed"
        ],
        "example_conversations": [
            {
                "user": "I want to create a new site",
                "ai": "I'll help you create a new site. What would you like to name this site?"
            },
            {
                "user": "Show me all sites in Mumbai", 
                "ai": "I'll search for sites in Mumbai. Let me get that information for you."
            }
        ]
    }
    
    return StandardResponse(
        success=True,
        message="AI capabilities retrieved successfully",
        data=capabilities
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    """
FastAPI Service Layer for PulsePro Site Management
RESTful API endpoints for the AI agent and direct site management
"""

from fastapi import FastAPI, HTTPException, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

# Import our custom classes (assuming they're in site_manager.py)
# from site_manager import SiteManager, AuthenticationManager, SiteData, AIAgentProcessor, initialize_system

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class AuthRequest(BaseModel):
    refresh_token: str = Field(..., description="JWT refresh token")

class SiteCreateRequest(BaseModel):
    location_name: str = Field(..., description="Name of the location")
    address_field1: str = Field(..., description="Primary address")
    address_field2: Optional[str] = Field("", description="Secondary address")
    country_id: int = Field(1, description="Country ID")
    state_id: int = Field(..., description="State ID")
    city_id: int = Field(..., description="City ID")
    pincode: Optional[str] = Field("", description="Postal code")
    mobile: Optional[str] = Field("", description="Mobile number")
    location_number: Optional[str] = Field("", description="Location number")
    location_code: Optional[str] = Field("", description="Location code")
    to_email: Optional[str] = Field("", description="Primary email")
    cc_email: Optional[str] = Field("", description="CC email")
    reporting_timezone: str = Field("UTC", description="Timezone for reporting")
    geo_fencing_enabled: bool = Field(False, description="Enable geo-fencing")
    geo_fencing_distance: int = Field(0, description="Geo-fencing distance")
    lat: float = Field(0.0, description="Latitude")
    lng: float = Field(0.0, description="Longitude")
    map_link: Optional[str] = Field("", description="Map link")
    has_custom_field: bool = Field(False, description="Has custom fields")
    is_schedule_active: bool = Field(False, description="Schedule active")

class SiteUpdateRequest(SiteCreateRequest):
    pass

class UserAssignmentRequest(BaseModel):
    location_id: int = Field(..., description="Location ID")
    user_ids: List[int] = Field(..., description="List of user IDs to assign")

class UserUnassignmentRequest(BaseModel):
    mapped_location_ids: List[int] = Field(..., description="List of mapped location IDs to remove")

class AIRequest(BaseModel):
    query: str = Field(..., description="Natural language query for the AI agent")

class SiteCreateRequest(BaseModel):
    location_name: str = Field(..., description="Name of the location")
    address_field1: str = Field(..., description="Primary address")
    address_field2: Optional[str] = Field("", description="Secondary address")
    country_id: int = Field(1, description="Country ID")
    state_id: int = Field(..., description="State ID")
    city_id: int = Field(..., description="City ID")
    pincode: Optional[str] = Field("", description="Postal code")
    mobile: Optional[str] = Field("", description="Mobile number")
    location_number: Optional[str] = Field("", description="Location number")
    location_code: Optional[str] = Field("", description="Location code")
    to_email: Optional[str] = Field("", description="Primary email")
    cc_email: Optional[str] = Field("", description="CC email")
    reporting_timezone: str = Field("UTC", description="Timezone for reporting")
    geo_fencing_enabled: bool = Field(False, description="Enable geo-fencing")
    geo_fencing_distance: int = Field(0, description="Geo-fencing distance")
    lat: float = Field(0.0, description="Latitude")
    lng: float = Field(0.0, description="Longitude")
    map_link: Optional[str] = Field("", description="Map link")
    has_custom_field: bool = Field(False, description="Has custom fields")
    is_schedule_active: bool = Field(False, description="Schedule active")

class SiteUpdateRequest(SiteCreateRequest):
    pass

class UserAssignmentRequest(BaseModel):
    location_id: int = Field(..., description="Location ID")
    user_ids: List[int] = Field(..., description="List of user IDs to assign")

class UserUnassignmentRequest(BaseModel):
    mapped_location_ids: List[int] = Field(..., description="List of mapped location IDs to remove")

class AIRequest(BaseModel):
    query: str = Field(..., description="Natural language query for the AI agent")

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Global variables to store system components
auth_manager = None
site_manager = None
ai_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting PulsePro Site Management API...")
    yield
    # Shutdown
    print("Shutting down PulsePro Site Management API...")

app = FastAPI(
    title="PulsePro Site Management API",
    description="AI-powered site management system with CRUD operations and user assignments",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_auth_manager():
    """Dependency to get authenticated site manager"""
    global auth_manager, site_manager, ai_agent
    
    if not auth_manager or not site_manager or not ai_agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="System not initialized. Please authenticate first."
        )
    return auth_manager, site_manager, ai_agent

@app.post("/auth/initialize", response_model=StandardResponse)
async def initialize_system_endpoint(auth_request: AuthRequest):
    """Initialize the system with refresh token"""
    global auth_manager, site_manager, ai_agent
    
    try:
        # Initialize system components
        ai_agent = initialize_system(auth_request.refresh_token)
        auth_manager = ai_agent.site_manager.auth_manager
        site_manager = ai_agent.site_manager
        
        # Test the authentication by refreshing token
        auth_manager.refresh_access_token()
        
        return StandardResponse(
            success=True,
            message="System initialized successfully"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unassign users: {str(e)}"
        )

# AI Agent Endpoints

@app.post("/ai/query", response_model=StandardResponse)
async def process_ai_query(ai_request: AIRequest, deps=Depends(get_auth_manager)):
    """Process natural language query through AI agent"""
    _, _, ai_agent = deps
    
    try:
        result = ai_agent.process_request(ai_request.query)
        return StandardResponse(
            success=result.get('success', True),
            message=result.get('message', 'Query processed'),
            data=result.get('data')
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process AI query: {str(e)}"
        )

@app.get("/ai/help", response_model=StandardResponse)
async def get_ai_help():
    """Get help information for AI agent commands"""
    help_info = {
        "available_commands": {
            "create": "Create a new site - 'create a site', 'add new location'",
            "update": "Update existing site - 'update site with id 123', 'edit location'",
            "delete": "Delete a site - 'delete site 123', 'remove location'",
            "show": "Show sites - 'show all sites', 'list locations', 'get all sites'",
            "assign": "Assign users - 'assign users to site', 'add users to location'",
            "unassign": "Unassign users - 'remove users from site', 'unassign users'"
        },
        "examples": [
            "show all sites",
            "create a new site in Mumbai",
            "delete site with id 123",
            "assign user 456 to site 123",
            "list all locations in India"
        ]
    }
    
    return StandardResponse(
        success=True,
        message="AI Agent help information",
        data=help_info
    )

# Utility Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PulsePro Site Management API"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PulsePro Site Management API",
        "version": "1.0.0",
        "description": "AI-powered site management system with CRUD operations and user assignments",
        "endpoints": {
            "auth": "/auth/initialize, /auth/refresh",
            "sites": "/sites (GET, POST), /sites/{id} (GET, PUT, DELETE)",
            "users": "/sites/{id}/users, /sites/assign-users, /sites/unassign-users",
            "ai": "/ai/query, /ai/help",
            "utility": "/health, /docs"
        }
    }

# Advanced Site Operations

@app.get("/sites/search", response_model=StandardResponse)
async def search_sites(
    name: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    deps=Depends(get_auth_manager)
):
    """Search sites with filters"""
    _, site_manager, _ = deps
    
    try:
        all_sites = site_manager.get_all_sites()
        sites = all_sites.get('locations', [])
        
        # Apply filters
        filtered_sites = sites
        
        if name:
            filtered_sites = [s for s in filtered_sites if name.lower() in s.get('location_name', '').lower()]
        
        if city:
            filtered_sites = [s for s in filtered_sites if city.lower() in s.get('city', '').lower()]
        
        if state:
            filtered_sites = [s for s in filtered_sites if state.lower() in s.get('state', '').lower()]
        
        if country:
            filtered_sites = [s for s in filtered_sites if country.lower() in s.get('country', '').lower()]
        
        return StandardResponse(
            success=True,
            message=f"Found {len(filtered_sites)} matching sites",
            data={
                "sites": filtered_sites,
                "total_count": len(filtered_sites),
                "filters_applied": {
                    "name": name,
                    "city": city,
                    "state": state,
                    "country": country
                }
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search sites: {str(e)}"
        )

@app.get("/analytics/sites-summary", response_model=StandardResponse)
async def get_sites_summary(deps=Depends(get_auth_manager)):
    """Get summary analytics for sites"""
    _, site_manager, _ = deps
    
    try:
        all_sites = site_manager.get_all_sites()
        sites = all_sites.get('locations', [])
        
        # Calculate analytics
        total_sites = len(sites)
        sites_by_country = {}
        sites_by_state = {}
        sites_with_geo_fencing = 0
        sites_with_mobile = 0
        sites_with_email = 0
        
        for site in sites:
            # Country distribution
            country = site.get('country', 'Unknown')
            sites_by_country[country] = sites_by_country.get(country, 0) + 1
            
            # State distribution
            state = site.get('state', 'Unknown')
            sites_by_state[state] = sites_by_state.get(state, 0) + 1
            
            # Feature usage
            if site.get('geo_fencing_enabled'):
                sites_with_geo_fencing += 1
            
            if site.get('mobile'):
                sites_with_mobile += 1
            
            if site.get('to_email'):
                sites_with_email += 1
        
        summary = {
            "total_sites": total_sites,
            "distribution": {
                "by_country": sites_by_country,
                "by_state": sites_by_state
            },
            "features": {
                "geo_fencing_enabled": sites_with_geo_fencing,
                "sites_with_mobile": sites_with_mobile,
                "sites_with_email": sites_with_email,
                "completion_rate": {
                    "mobile": round((sites_with_mobile / total_sites) * 100, 2) if total_sites > 0 else 0,
                    "email": round((sites_with_email / total_sites) * 100, 2) if total_sites > 0 else 0,
                    "geo_fencing": round((sites_with_geo_fencing / total_sites) * 100, 2) if total_sites > 0 else 0
                }
            }
        }
        
        return StandardResponse(
            success=True,
            message="Sites summary generated successfully",
            data=summary
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )

# Batch Operations

@app.post("/sites/batch-create", response_model=StandardResponse)
async def batch_create_sites(sites_data: List[SiteCreateRequest], deps=Depends(get_auth_manager)):
    """Create multiple sites in batch"""
    _, site_manager, _ = deps
    
    results = []
    errors = []
    
    for idx, site_request in enumerate(sites_data):
        try:
            site_data = SiteData(**site_request.dict())
            result = site_manager.create_site(site_data)
            results.append({
                "index": idx,
                "site_name": site_request.location_name,
                "success": True,
                "data": result
            })
        except Exception as e:
            errors.append({
                "index": idx,
                "site_name": site_request.location_name,
                "error": str(e)
            })
    
    return StandardResponse(
        success=len(errors) == 0,
        message=f"Batch operation completed. {len(results)} successful, {len(errors)} failed",
        data={
            "successful": results,
            "failed": errors,
            "summary": {
                "total_attempted": len(sites_data),
                "successful_count": len(results),
                "failed_count": len(errors)
            }
        }
    )

@app.delete("/sites/batch-delete", response_model=StandardResponse)
async def batch_delete_sites(location_ids: List[int] = Body(...), deps=Depends(get_auth_manager)):
    """Delete multiple sites in batch"""
    _, site_manager, _ = deps
    
    results = []
    errors = []
    
    for location_id in location_ids:
        try:
            result = site_manager.delete_site(location_id)
            results.append({
                "location_id": location_id,
                "success": True,
                "data": result
            })
        except Exception as e:
            errors.append({
                "location_id": location_id,
                "error": str(e)
            })
    
    return StandardResponse(
        success=len(errors) == 0,
        message=f"Batch deletion completed. {len(results)} successful, {len(errors)} failed",
        data={
            "successful": results,
            "failed": errors,
            "summary": {
                "total_attempted": len(location_ids),
                "successful_count": len(results),
                "failed_count": len(errors)
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

@app.post("/auth/refresh", response_model=StandardResponse)
async def refresh_token(deps=Depends(get_auth_manager)):
    """Refresh access token"""
    auth_manager, _, _ = deps
    
    try:
        access_token = auth_manager.refresh_access_token()
        return StandardResponse(
            success=True,
            message="Token refreshed successfully",
            data={"access_token": access_token}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {str(e)}"
        )

# Site CRUD Operations

@app.post("/sites", response_model=StandardResponse)
async def create_site(site_request: SiteCreateRequest, deps=Depends(get_auth_manager)):
    """Create a new site"""
    _, site_manager, _ = deps
    
    try:
        # Convert request to SiteData
        site_data = SiteData(**site_request.dict())
        result = site_manager.create_site(site_data)
        
        return StandardResponse(
            success=True,
            message="Site created successfully",
            data=result
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create site: {str(e)}"
        )

@app.get("/sites", response_model=StandardResponse)
async def get_all_sites(deps=Depends(get_auth_manager)):
    """Get all sites"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.get_all_sites()
        return StandardResponse(
            success=True,
            message="Sites retrieved successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sites: {str(e)}"
        )

@app.get("/sites/{location_id}", response_model=StandardResponse)
async def get_site_by_id(location_id: int, deps=Depends(get_auth_manager)):
    """Get a specific site by ID"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.get_site_by_id(location_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Site with ID {location_id} not found"
            )
        
        return StandardResponse(
            success=True,
            message="Site retrieved successfully",
            data=result
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve site: {str(e)}"
        )

@app.put("/sites/{location_id}", response_model=StandardResponse)
async def update_site(location_id: int, site_request: SiteUpdateRequest, deps=Depends(get_auth_manager)):
    """Update an existing site"""
    _, site_manager, _ = deps
    
    try:
        # Convert request to SiteData
        site_data = SiteData(**site_request.dict())
        result = site_manager.update_site(location_id, site_data)
        
        return StandardResponse(
            success=True,
            message="Site updated successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update site: {str(e)}"
        )

@app.delete("/sites/{location_id}", response_model=StandardResponse)
async def delete_site(location_id: int, deps=Depends(get_auth_manager)):
    """Delete a site"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.delete_site(location_id)
        return StandardResponse(
            success=True,
            message="Site deleted successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete site: {str(e)}"
        )

# User Assignment Operations

@app.get("/sites/{location_id}/users", response_model=StandardResponse)
async def get_site_users(location_id: int, search_keyword: str = "", deps=Depends(get_auth_manager)):
    """Get all users available for assignment to a site"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.get_site_users(location_id, search_keyword)
        return StandardResponse(
            success=True,
            message="Users retrieved successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve users: {str(e)}"
        )

@app.post("/sites/assign-users", response_model=StandardResponse)
async def assign_users_to_site(assignment_request: UserAssignmentRequest, deps=Depends(get_auth_manager)):
    """Assign multiple users to a site"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.assign_users_to_site(
            assignment_request.location_id,
            assignment_request.user_ids
        )
        
        return StandardResponse(
            success=True,
            message="Users assigned successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign users: {str(e)}"
        )

@app.post("/sites/unassign-users", response_model=StandardResponse)
async def unassign_users_from_site(unassignment_request: UserUnassignmentRequest, deps=Depends(get_auth_manager)):
    """Unassign users from a site"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.unassign_users_from_site(unassignment_request.mapped_location_ids)
        
        return StandardResponse(
            success=True,
            message="Users unassigned successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unassign users: {str(e)}"
        )