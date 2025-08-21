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

# Global variable to store the manager
ollama_site_manager = None

def initialize_ollama_site_manager():
    """Initialize the Ollama site manager"""
    global ollama_site_manager
    
    try:
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from site_manager import SiteManager  # Replace with actual import
        
        ollama_site_manager = SiteManager(
            model="llama3.1:8b",
            host="http://localhost:11434"
        )
        
        # Test the connection
        test_response = ollama_site_manager.chat(
            message="Hello",
            session_id="test_session",
            user_id="system"
        )
        
        print("Ollama Site Manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize Ollama Site Manager: {e}")
        ollama_site_manager = None
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

# @app.post("/chat", response_model=ChatResponse)
# async def chat_with_ollama_ai(chat_request: ChatRequest):
#     """Chat directly with Ollama AI"""
    
#     # Generate session ID if not provided
#     session_id = chat_request.session_id or str(uuid.uuid4())
#     user_id = chat_request.user_id or "anonymous"
    
#     try:
#         # Get Ollama client
#         client = get_ollama_client()
        
#         # Validate message
#         if not chat_request.message or not chat_request.message.strip():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Message cannot be empty"
#             )
        
#         user_message = chat_request.message.strip()
        
#         # Get or create session history
#         if session_id not in chat_sessions:
#             chat_sessions[session_id] = []
        
#         session_history = chat_sessions[session_id]
        
#         # Build conversation context for better responses
#         conversation_context = ""
#         if session_history:
#             # Include last 5 exchanges for context
#             recent_history = session_history[-10:]  # Last 10 messages (5 exchanges)
#             for msg in recent_history:
#                 role = "Human" if msg["role"] == "user" else "Assistant"
#                 conversation_context += f"{role}: {msg['content']}\n"
        
#         # Create the prompt with context
#         if conversation_context:
#             full_prompt = f"""Previous conversation:
# {conversation_context}
# Human: {user_message}
# Assistant:"""
#         else:
#             full_prompt = f"Human: {user_message}\nAssistant:"
        
#         # Generate response from Ollama
#         response = client.generate(
#             model="llama3.1:8b",
#             prompt=full_prompt,
#             options={
#                 "num_predict": 500,  # Allow longer responses
#                 "temperature": 0.7,  # Slightly creative
#                 "top_p": 0.9,       # Good balance
#                 "stop": ["Human:", "human:", "User:", "user:"]  # Stop at next human input
#             }
#         )
        
#         ai_response = response['response'].strip()
        
#         # Store the conversation
#         timestamp = datetime.now().isoformat()
        
#         # Add user message to history
#         session_history.append({
#             "role": "user",
#             "content": user_message,
#             "timestamp": timestamp,
#             "user_id": user_id
#         })
        
#         # Add AI response to history
#         session_history.append({
#             "role": "assistant", 
#             "content": ai_response,
#             "timestamp": timestamp,
#             "model": "llama3.1:8b"
#         })
        
#         # Keep only last 50 messages per session to prevent memory issues
#         if len(session_history) > 50:
#             session_history = session_history[-50:]
#             chat_sessions[session_id] = session_history
        
#         return ChatResponse(
#             message=ai_response,
#             status="completed",
#             session_id=session_id,
#             context={
#                 "message_count": len(session_history),
#                 "model": "llama3.1:8b",
#                 "timestamp": timestamp
#             },
#             data={
#                 "user_id": user_id,
#                 "response_length": len(ai_response)
#             }
#         )
        
#     except HTTPException:
#         # Re-raise HTTP exceptions
#         raise
        
#     except Exception as e:
#         print(f"Chat error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Chat processing failed: {str(e)}"
#         )














import logging
import json


logging.basicConfig(
    level=logging.INFO,  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s"
)




@app.post("/chat", response_model=ChatResponse)
async def chat_with_ollama_ai(chat_request: ChatRequest):
    """Chat with Ollama AI using two-phase strategy for site management"""
    
    # Generate session ID if not provided
    session_id = chat_request.session_id or str(uuid.uuid4())
    user_id = chat_request.user_id or "anonymous"
    
    try:
        # Get Ollama client
        client = get_ollama_client()
        
        # Validate message
        if not chat_request.message or not chat_request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        user_message = chat_request.message.strip()
        
        # Get or create session history and state
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "messages": [],
                "phase": 1,  # Phase 1: Data Collection, Phase 2: JSON Generation
                "operation_type": None,
                "gathered_data": {},
                "ready_for_execution": False
            }
        
        session = chat_sessions[session_id]
        session_history = session["messages"]
        
        # Handle user confirmation for execution
        if session["ready_for_execution"] and user_message.lower() in ['yes', 'y', 'confirm', 'proceed']:
            # Move to Phase 2: JSON Generation
            result = await execute_phase_2_json_generation(session, client)
            return result
        
        # Handle user cancellation
        if user_message.lower() in ['no', 'cancel', 'stop', 'abort']:
            # Reset session
            session["phase"] = 1
            session["operation_type"] = None
            session["gathered_data"] = {}
            session["ready_for_execution"] = False
            
            return ChatResponse(
                message="Operation cancelled. How else can I help you with site management?",
                status="cancelled",
                session_id=session_id,
                context={"phase": 1, "operation": None},
                data={}
            )
        
        # Phase 1: Data Collection and Intent Recognition
        result = await execute_phase_1_data_collection(
            session, client, user_message, user_id, session_id
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )

async def execute_phase_1_data_collection(session, client, user_message, user_id, session_id):
    """Phase 1: Collect data and determine operation intent"""
    
    session_history = session["messages"]
    
    # Build conversation context
    conversation_context = ""
    if session_history:
        recent_history = session_history[-10:]  # Last 10 messages
        for msg in recent_history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
    
    # Create Phase 1 prompt
    full_prompt = f"""You are a PulsePro Site Management Assistant. Your job is to collect information and understand user intent for site management operations.

AVAILABLE OPERATIONS:
1. CREATE_SITE - Create new sites
2. DELETE_SITE - Delete existing sites  
3. UPDATE_SITE - Modify site information
4. SHOW_SITES - Display all sites
5. ASSIGN_USERS - Add users to sites
6. UNASSIGN_USERS - Remove users from sites

YOUR TASK: Have a natural conversation to collect ALL required information.

REQUIRED FIELDS BY OPERATION:

CREATE_SITE needs:
- location_name (site name)
- address_field1 (primary address)
- country_id (1=India, 2=USA)
- state_id (numeric state identifier)
- city_id (numeric city identifier)  
- reporting_timezone (UTC, Asia/Kolkata, etc.)

DELETE_SITE needs:
- location_name or location_id (which site to delete)

UPDATE_SITE needs:
- location_name or location_id (which site to update)
- Fields to update (any from CREATE_SITE list)

SHOW_SITES needs:
- No additional info required

ASSIGN_USERS needs:
- location_name or location_id (which site)
- user_names (which users to assign)

UNASSIGN_USERS needs:
- location_name or location_id (which site)
- user_names (which users to remove)

CONVERSATION STRATEGY:
1. Identify what operation user wants
2. Ask for missing required information ONE field at a time
3. Be conversational and helpful
4. When you have ALL required info, ask for confirmation
5. If user confirms, say: "READY_FOR_EXECUTION"

CONVERSATION RULES:
- Ask ONE question at a time
- Give examples for numeric fields
- For names (sites/users), accept natural names - system will convert to IDs
- Stay focused on site management only
- Be friendly but systematic

CONVERSATION HISTORY:
{conversation_context}

USER MESSAGE: {user_message}

Respond naturally to collect the next piece of information needed:"""

    # Generate response from Ollama
    response = client.generate(
        model="llama3.1:8b",
        prompt=full_prompt,
        options={
            "num_predict": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["Human:", "human:", "User:", "user:"]
        }
    )
    
    ai_response = response['response'].strip()
    timestamp = datetime.now().isoformat()
    
    # Store the conversation
    session_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": timestamp,
        "user_id": user_id
    })
    
    session_history.append({
        "role": "assistant", 
        "content": ai_response,
        "timestamp": timestamp,
        "model": "llama3.1:8b",
        "phase": 1
    })
    
    # Check if ready for execution
    if "READY_FOR_EXECUTION" in ai_response:
        session["ready_for_execution"] = True
        
        # Extract operation type from conversation
        conversation_text = " ".join([msg["content"] for msg in session_history])
        session["operation_type"] = extract_operation_type(conversation_text)
        
        clean_response = ai_response.replace("READY_FOR_EXECUTION", "").strip()
        clean_response += "\n\nPlease confirm: Should I proceed with this operation? (yes/no)"
        
        return ChatResponse(
            message=clean_response,
            status="confirmation_needed",
            session_id=session_id,
            context={
                "phase": 1,
                "ready_for_execution": True,
                "operation_type": session["operation_type"]
            },
            data={"awaiting_confirmation": True}
        )
    
    # Keep conversation history manageable
    if len(session_history) > 20:
        session_history = session_history[-20:]
    
    return ChatResponse(
        message=ai_response,
        status="collecting_info",
        session_id=session_id,
        context={
            "phase": 1,
            "message_count": len(session_history),
            "operation_type": session["operation_type"]
        },
        data={"user_id": user_id}
    )

async def execute_phase_2_json_generation(session, client):
    """Phase 2: Convert conversation to JSON and execute operation"""
    
    session_history = session["messages"]
    operation_type = session["operation_type"]
    
    if not operation_type:
        # Try to extract operation type again
        conversation_text = " ".join([msg["content"] for msg in session_history])
        operation_type = extract_operation_type(conversation_text)
    
    if not operation_type:
        return ChatResponse(
            message="I couldn't determine the operation type. Please start over.",
            status="error",
            session_id=session["session_id"],
            context={"phase": 1, "error": "operation_type_not_found"},
            data={}
        )
    
    # Build conversation history for analysis
    conversation_context = ""
    for msg in session_history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        conversation_context += f"{role}: {msg['content']}\n"
    
    # API specifications for JSON generation
    api_specs = {
        "CREATE_SITE": {
            "method": "POST",
            "endpoint": "/customer/add_location/",
            "required_fields": ["location_name", "address_field1", "country_id", "state_id", "city_id", "reporting_timezone"]
        },
        "DELETE_SITE": {
            "method": "DELETE", 
            "endpoint": "/customer/locations/{location_id}/",
            "required_fields": ["location_name"]
        },
        "UPDATE_SITE": {
            "method": "POST",
            "endpoint": "/customer/edit_location/save/?location_id={location_id}",
            "required_fields": ["location_name"]
        },
        "SHOW_SITES": {
            "method": "POST",
            "endpoint": "/customer/locations/",
            "required_fields": []
        },
        "ASSIGN_USERS": {
            "method": "POST",
            "endpoint": "/customer/add_location_to_multiple_user/",
            "required_fields": ["location_name", "user_names"]
        },
        "UNASSIGN_USERS": {
            "method": "POST",
            "endpoint": "/customer/delete_user_to_location_mapping/",
            "required_fields": ["location_name", "user_names"]
        }
    }
    
    # Create Phase 2 prompt for JSON generation
    json_prompt = f"""You are a JSON Generator for PulsePro Site Management API calls.

TASK: Convert the conversation history into a properly formatted JSON structure for {operation_type}.

CONVERSATION HISTORY TO ANALYZE:
{conversation_context}

OPERATION: {operation_type}
API SPEC: {json.dumps(api_specs.get(operation_type, {}), indent=2)}

INSTRUCTIONS:
1. Extract all relevant data from the conversation history
2. Create appropriate JSON structure for {operation_type}
3. Use correct data types (string, integer, boolean, array)
4. For CREATE_SITE, include ALL fields with defaults for optional ones
5. For location/user names, keep them as names (system will resolve IDs)

OUTPUT ONLY VALID JSON in this format:
{{
  "operation": "{operation_type}",
  "data": {{
    // extracted data here
  }}
}}

For CREATE_SITE example:
{{
  "operation": "CREATE_SITE",
  "data": {{
    "location_name": "extracted_name",
    "address_field1": "extracted_address",
    "country_id": 1,
    "state_id": extracted_state_id,
    "city_id": extracted_city_id,
    "reporting_timezone": "extracted_timezone",
    "address_field2": "",
    "pincode": "",
    "mobile": "",
    "location_number": "",
    "location_code": "",
    "to_email": "",
    "cc_email": "",
    "geo_fencing_enabled": false,
    "geo_fencing_distance": 0,
    "lat": 0.0,
    "lng": 0.0,
    "map_link": "",
    "city_list": [],
    "state_list": [],
    "has_custom_field": false,
    "is_schedule_active": false
  }}
}}

Generate JSON now:"""

    # Generate JSON structure from Ollama
    response = client.generate(
        model="llama3.1:8b",
        prompt=json_prompt,
        options={
            "num_predict": 400,
            "temperature": 0.3,  # Lower temperature for more structured output
            "top_p": 0.8
        }
    )
    
    json_response = response['response'].strip()
    
    # Parse JSON from response
    try:
        json_structure = extract_json_from_response(json_response)
        
        if json_structure:
            # Execute the operation
            execution_result = await execute_site_operation(json_structure)
            
            # Reset session for next operation
            session["phase"] = 1
            session["operation_type"] = None
            session["gathered_data"] = {}
            session["ready_for_execution"] = False
            
            return ChatResponse(
                message=f"✅ {operation_type} completed successfully! {execution_result.get('message', '')}",
                status="completed",
                session_id=session["session_id"],
                context={
                    "phase": 2,
                    "operation": operation_type,
                    "executed": True
                },
                data={
                    "json_structure": json_structure,
                    "execution_result": execution_result
                }
            )
        else:
            raise ValueError("Could not parse JSON from response")
            
    except Exception as e:
        logger.error(f"JSON generation failed: {e}")
        return ChatResponse(
            message=f"❌ Failed to process the operation: {str(e)}. Please try again.",
            status="error",
            session_id=session["session_id"],
            context={"phase": 2, "error": str(e)},
            data={"raw_response": json_response}
        )

def extract_operation_type(conversation_text):
    """Extract operation type from conversation"""
    text_lower = conversation_text.lower()
    
    if any(word in text_lower for word in ['create', 'add', 'new']) and 'site' in text_lower:
        return "CREATE_SITE"
    elif any(word in text_lower for word in ['delete', 'remove']) and 'site' in text_lower:
        return "DELETE_SITE"
    elif any(word in text_lower for word in ['update', 'edit', 'modify']) and 'site' in text_lower:
        return "UPDATE_SITE"
    elif any(word in text_lower for word in ['show', 'list', 'display', 'view']) and 'site' in text_lower:
        return "SHOW_SITES"
    elif any(word in text_lower for word in ['assign', 'add']) and 'user' in text_lower:
        return "ASSIGN_USERS"
    elif any(word in text_lower for word in ['unassign', 'remove']) and 'user' in text_lower:
        return "UNASSIGN_USERS"
    
    return None

def extract_json_from_response(response):
    """Extract JSON from Ollama response"""
    import re
    import json
    
    # Try to find JSON block
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # JSON code blocks
        r'\{[^}]*"operation"[^}]*\}.*?\}',  # JSON with operation key
        r'\{.*?\}',  # Any JSON-like structure
    ]
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                json_text = match.group(1) if match.groups() else match.group(0)
                # Clean up the JSON text
                json_text = json_text.strip()
                return json.loads(json_text)
            except (json.JSONDecodeError, IndexError):
                continue
    
    return None

async def execute_site_operation(json_structure):
    """Execute the actual site management operation"""
    operation = json_structure.get("operation")
    data = json_structure.get("data", {})
    
    try:
        if operation == "CREATE_SITE":
            # Use the existing site manager to create site
            site_data = SiteData(**data)
            result = ollama_site_manager.create_site(site_data)
            return {"success": True, "message": f"Site '{data.get('location_name')}' created", "data": result}
            
        elif operation == "SHOW_SITES":
            result = ollama_site_manager.get_all_sites()
            sites = result.get('locations', [])
            return {"success": True, "message": f"Found {len(sites)} sites", "data": result}
            
        elif operation == "DELETE_SITE":
            # First find the site by name to get ID
            all_sites = ollama_site_manager.get_all_sites()
            site_name = data.get("location_name", "")
            
            site_id = None
            for site in all_sites.get('locations', []):
                if site.get('location_name', '').lower() == site_name.lower():
                    site_id = site.get('id')
                    break
            
            if site_id:
                result = ollama_site_manager.delete_site(site_id)
                return {"success": True, "message": f"Site '{site_name}' deleted", "data": result}
            else:
                return {"success": False, "message": f"Site '{site_name}' not found"}
        
        # Add other operations here...
        else:
            return {"success": False, "message": f"Operation {operation} not implemented yet"}
            
    except Exception as e:
        logger.error(f"Operation execution failed: {e}")
        return {"success": False, "message": f"Execution failed: {str(e)}"}





























@app.get("/chat/sessions/{session_id}/history", response_model=StandardResponse)
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in chat_sessions:
        return StandardResponse(
            success=False,
            message="Session not found",
            data={"session_id": session_id}
        )
    
    return StandardResponse(
        success=True,
        message="Chat history retrieved",
        data={
            "session_id": session_id,
            "history": chat_sessions[session_id],
            "message_count": len(chat_sessions[session_id])
        }
    )

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