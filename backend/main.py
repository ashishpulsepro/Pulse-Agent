"""
main.py - COMPLETE VERSION
Complete FastAPI Service Layer with Ollama Integration
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
from ai_agent import OllamaIntegratedSiteManager

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

class ConversationHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]

class AIRequest(BaseModel):
    query: str = Field(..., description="Natural language query for the AI agent")

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

def get_ollama_site_manager():
    """Dependency to get the Ollama-integrated site manager"""
    global ollama_site_manager
    
    if not ollama_site_manager:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="System not initialized. Please authenticate first."
        )
    return ollama_site_manager

# ============================================
# AUTHENTICATION ENDPOINTS
# ============================================

@app.post("/auth/initialize", response_model=StandardResponse)
async def initialize_system_with_ollama(auth_request: AuthRequest, config: OllamaConfig = Body(default=OllamaConfig())):
    """Initialize the system with refresh token and Ollama model"""
    global ollama_site_manager, ollama_config
    
    try:
        # Update Ollama configuration
        ollama_config = config
        
        # Initialize authentication
        auth_manager = AuthenticationManager()
        auth_manager.set_refresh_token(auth_request.refresh_token)
        
        # Test authentication
        auth_manager.refresh_access_token()
        
        # Initialize Ollama-integrated site manager
        ollama_site_manager = OllamaIntegratedSiteManager(
            auth_manager=auth_manager, 
            model_name=ollama_config.model_name
        )
        
        # Test Ollama connection
        ollama_status = "connected" if ollama_site_manager.conversational_agent.ollama_client else "fallback_mode"
        
        return StandardResponse(
            success=True,
            message="System initialized successfully with Ollama AI integration",
            data={
                "ollama_model": ollama_config.model_name,
                "ollama_status": ollama_status,
                "features": ["conversational_ai", "natural_language_processing", "multi_turn_conversations"]
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}"
        )

@app.post("/auth/refresh", response_model=StandardResponse)
async def refresh_token(manager=Depends(get_ollama_site_manager)):
    """Refresh access token"""
    try:
        access_token = manager.auth_manager.refresh_access_token()
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

# ============================================
# OLLAMA-SPECIFIC ENDPOINTS
# ============================================
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
            print(f"Models response structure: {models}")  # Debug line
            
            # Handle different response structures
            if isinstance(models, dict) and 'models' in models:
                available_models = []
                for model in models['models']:
                    if isinstance(model, dict):
                        # Try different possible keys
                        name = model.get('name') or model.get('model') or model.get('id') or str(model)
                        available_models.append(name)
                    else:
                        available_models.append(str(model))
            else:
                # Handle case where models is a list directly
                available_models = [str(model) for model in (models if isinstance(models, list) else [models])]
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
                    prompt="Hello",
                    options={"num_predict": 5}
                )
                
                return StandardResponse(
                    success=True,
                    message="Ollama connection successful",
                    data={
                        "available_models": available_models,
                        "target_model": "llama3.1:8b",
                        "model_status": "available",
                        "test_response": response['response'].strip()
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
# ============================================
# CONVERSATIONAL AI ENDPOINTS
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ollama_ai(chat_request: ChatRequest, manager=Depends(get_ollama_site_manager)):
    """Main conversational interface with Ollama AI"""
    
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
async def get_conversation_history(session_id: str, manager=Depends(get_ollama_site_manager)):
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
async def clear_conversation_history(session_id: str, manager=Depends(get_ollama_site_manager)):
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

# ============================================
# SITE CRUD OPERATIONS
# ============================================

@app.post("/sites", response_model=StandardResponse)
async def create_site(site_request: SiteCreateRequest, manager=Depends(get_ollama_site_manager)):
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
async def get_all_sites(manager=Depends(get_ollama_site_manager)):
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
async def get_site_by_id(location_id: int, manager=Depends(get_ollama_site_manager)):
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
async def update_site(location_id: int, site_request: SiteUpdateRequest, manager=Depends(get_ollama_site_manager)):
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
async def delete_site(location_id: int, manager=Depends(get_ollama_site_manager)):
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

# ============================================
# USER ASSIGNMENT OPERATIONS
# ============================================

@app.get("/sites/{location_id}/users", response_model=StandardResponse)
async def get_site_users(location_id: int, search_keyword: str = "", manager=Depends(get_ollama_site_manager)):
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
async def assign_users_to_site(assignment_request: UserAssignmentRequest, manager=Depends(get_ollama_site_manager)):
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
async def unassign_users_from_site(unassignment_request: UserUnassignmentRequest, manager=Depends(get_ollama_site_manager)):
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

# ============================================
# ANALYTICS AND SEARCH
# ============================================

@app.get("/analytics/sites-summary", response_model=StandardResponse)
async def get_sites_summary(manager=Depends(get_ollama_site_manager)):
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

@app.get("/sites/search", response_model=StandardResponse)
async def search_sites(
    name: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    manager=Depends(get_ollama_site_manager)
):
    """Search sites with filters"""
    try:
        all_sites = manager.get_all_sites()
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

# ============================================
# BATCH OPERATIONS
# ============================================

@app.post("/sites/batch-create", response_model=StandardResponse)
async def batch_create_sites(sites_data: List[SiteCreateRequest], manager=Depends(get_ollama_site_manager)):
    """Create multiple sites in batch"""
    results = []
    errors = []
    
    for idx, site_request in enumerate(sites_data):
        try:
            site_data = SiteData(**site_request.dict())
            result = manager.create_site(site_data)
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
async def batch_delete_sites(location_ids: List[int] = Body(...), manager=Depends(get_ollama_site_manager)):
    """Delete multiple sites in batch"""
    results = []
    errors = []
    
    for location_id in location_ids:
        try:
            result = manager.delete_site(location_id)
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

# ============================================
# LEGACY AI ENDPOINTS
# ============================================

@app.post("/ai/query", response_model=StandardResponse)
async def process_ai_query_legacy(ai_request: AIRequest, manager=Depends(get_ollama_site_manager)):
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

@app.get("/ai/model-info", response_model=StandardResponse)
async def get_model_info():
    """Get information about the loaded Ollama model"""
    
    global ollama_config, ollama_site_manager
    
    if not ollama_site_manager:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="System not initialized"
        )
    
    model_info = {
        "model_name": ollama_config.model_name,
        "temperature": ollama_config.temperature,
        "status": "connected" if ollama_site_manager.conversational_agent.ollama_client else "fallback_mode",
        "capabilities": {
            "text_generation": True,
            "intent_classification": True,
            "information_extraction": True,
            "conversation_management": True
        }
    }
    
    return StandardResponse(
        success=True,
        message="Model information retrieved successfully",
        data=model_info
    )

# ============================================
# WEB INTERFACE
# ============================================

@app.get("/chat/interface")
async def get_chat_interface():
    """Serve a simple web chat interface"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PulsePro Ollama AI Chat</title>
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
            .ollama-status { background: #4caf50; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 PulsePro AI Assistant</h1>
                <p>Powered by Ollama llama3.1:8b <span class="ollama-status">Local AI</span></p>
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
               
                 """ 
    """
main_ollama.py
Updated FastAPI Service Layer with Ollama Integration
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
from ai_agent import OllamaIntegratedSiteManager  # Updated import

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

def get_ollama_site_manager():
    """Dependency to get the Ollama-integrated site manager"""
    global ollama_site_manager
    
    if not ollama_site_manager:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="System not initialized. Please authenticate first."
        )
    return ollama_site_manager

# Enhanced Authentication with Ollama Integration
@app.post("/auth/initialize", response_model=StandardResponse)
async def initialize_system_with_ollama(auth_request: AuthRequest, config: OllamaConfig = Body(default=OllamaConfig())):
    """Initialize the system with refresh token and Ollama model"""
    global ollama_site_manager, ollama_config
    
    try:
        # Update Ollama configuration
        ollama_config = config
        
        # Initialize authentication
        auth_manager = AuthenticationManager()
        auth_manager.set_refresh_token(auth_request.refresh_token)
        
        # Test authentication
        auth_manager.refresh_access_token()
        
        # Initialize Ollama-integrated site manager
        ollama_site_manager = OllamaIntegratedSiteManager(
            auth_manager=auth_manager, 
            model_name=ollama_config.model_name
        )
        
        # Test Ollama connection
        ollama_status = "connected" if ollama_site_manager.conversational_agent.ollama_client else "fallback_mode"
        
        return StandardResponse(
            success=True,
            message="System initialized successfully with Ollama AI integration",
            data={
                "ollama_model": ollama_config.model_name,
                "ollama_status": ollama_status,
                "features": ["conversational_ai", "natural_language_processing", "multi_turn_conversations"]
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}"
        )

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





@app.post("/chat", response_model=ChatResponse)
async def chat_with_ollama_ai(chat_request: ChatRequest, manager=Depends(get_ollama_site_manager)):
    """Main conversational interface with Ollama AI"""
    
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

@app.get("/chat/history/{session_id}")
async def get_conversation_history(session_id: str, manager=Depends(get_ollama_site_manager)):
    """Get conversation history for a session"""
    
    try:
        history = manager.get_conversation_history(session_id)
        
        return {
            "session_id": session_id,
            "history": history
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )

@app.delete("/chat/history/{session_id}", response_model=StandardResponse)
async def clear_conversation_history(session_id: str, manager=Depends(get_ollama_site_manager)):
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

# Chat Interface for Web UI (Updated for Ollama)
@app.get("/chat/interface")
async def get_chat_interface():
    """Serve a simple web chat interface"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PulsePro Ollama AI Chat</title>
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
            .ollama-status { background: #4caf50; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 PulsePro AI Assistant</h1>
                <p>Powered by Ollama llama3.1:8b <span class="ollama-status">Local AI</span></p>
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
                messageDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'Ollama AI'}:</strong><br>${text.replace(/\\n/g, '<br>')}`;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            function showTyping() {
                const messagesDiv = document.getElementById('chat-messages');
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message ai-message typing';
                typingDiv.id = 'typing-indicator';
                typingDiv.innerHTML = '<strong>Ollama AI:</strong><br>Thinking...';
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
            addMessage('Hello! I\\'m your Ollama-powered AI assistant for site management. I can help you create, update, delete, and search sites using natural language. What would you like to do?', 'ai');
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PulsePro Site Management API with Ollama",
        "version": "2.0.0",
        "description": "Conversational AI-powered site management using local Ollama llama3.1:8b",
        "features": [
            "Local Ollama integration",
            "Multi-turn conversations",
            "Natural language processing", 
            "Site CRUD operations",
            "User management",
            "Analytics and reporting",
            "Batch operations",
            "Search functionality"
        ],
        "endpoints": {
            "authentication": {
                "initialize": "/auth/initialize (POST)",
                "refresh": "/auth/refresh (POST)"
            },
            "chat": {
                "main_interface": "/chat (POST)",
                "history": "/chat/history/{session_id} (GET)",
                "clear_history": "/chat/history/{session_id} (DELETE)",
                "new_session": "/chat/sessions (POST)",
                "web_interface": "/chat/interface (GET)"
            },
            "sites": {
                "list": "/sites (GET)",
                "create": "/sites (POST)",
                "get_by_id": "/sites/{id} (GET)",
                "update": "/sites/{id} (PUT)",
                "delete": "/sites/{id} (DELETE)",
                "search": "/sites/search (GET)",
                "batch_create": "/sites/batch-create (POST)",
                "batch_delete": "/sites/batch-delete (DELETE)"
            },
            "users": {
                "site_users": "/sites/{id}/users (GET)",
                "assign": "/sites/assign-users (POST)",
                "unassign": "/sites/unassign-users (POST)"
            },
            "analytics": {
                "summary": "/analytics/sites-summary (GET)"
            },
            "ai": {
                "legacy_query": "/ai/query (POST)",
                "capabilities": "/ai/capabilities (GET)",
                "model_info": "/ai/model-info (GET)"
            },
            "ollama": {
                "test": "/ollama/test (GET)"
            },
            "utility": {
                "health": "/health (GET)",
                "docs": "/docs (GET)",
                "openapi": "/openapi.json (GET)"
            }
        },
        "usage_examples": {
            "initialize": {
                "method": "POST",
                "url": "/auth/initialize",
                "body": {
                    "refresh_token": "YOUR_PULSEPRO_TOKEN",
                    "model_name": "llama3.1:8b"
                }
            },
            "chat": {
                "method": "POST", 
                "url": "/chat",
                "body": {
                    "message": "I want to create a new site",
                    "session_id": "my_session"
                }
            },
            "create_site": {
                "method": "POST",
                "url": "/sites", 
                "body": {
                    "location_name": "Mumbai Office",
                    "address_field1": "Bandra West",
                    "country_id": 1,
                    "state_id": 61,
                    "city_id": 19788,
                    "reporting_timezone": "Asia/Kolkata"
                }
            }
        }
    }

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