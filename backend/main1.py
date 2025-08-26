"""
main1.py
Enhanced PulsePro API with Complete Onboarding Flow
Integrates: Site Management, User Management, and Onboarding Process
"""

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
from datetime import datetime

# Import our organized components
from site_manager import SiteData, AuthenticationManager, SiteManager
from tree_system import TreeBasedSystem
from onboarding_system import OnboardingSystem, UserManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Request/Response Models
class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User ID")
    mode: Optional[str] = Field(default="normal", description="Chat mode: 'normal' or 'onboarding'")

class ChatResponse(BaseModel):
    message: str
    status: str
    session_id: str
    mode: str = "normal"
    context: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None

class OnboardingRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID for onboarding continuity")

class OnboardingResponse(BaseModel):
    message: str
    status: str
    session_id: str
    current_step: str
    progress: int
    data: Optional[Dict[str, Any]] = None

# Global variables
tree_system = None
onboarding_system = None
auth_manager = None
site_manager = None
user_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("🚀 Starting PulsePro Enhanced API with Onboarding...")
    print(f"🌳 System: Tree-Based Intent Processing + Complete Onboarding Flow")
    
    # Initialize systems
    initialize_systems()
    
    yield
    
    # Shutdown
    print("📴 Shutting down PulsePro Enhanced API...")

app = FastAPI(
    title="PulsePro Enhanced API with Onboarding",
    description="Complete site and user management system with guided onboarding flow",
    version="4.0.0",
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

def initialize_systems():
    """Initialize all systems"""
    global tree_system, onboarding_system, auth_manager, site_manager, user_manager
    
    try:
        # Initialize core components
        auth_manager = AuthenticationManager()
        site_manager = SiteManager(auth_manager)
        user_manager = UserManager(auth_manager)
        
        # Initialize tree system for normal operations
        tree_system = TreeBasedSystem(site_manager)
        
        # Initialize onboarding system
        onboarding_system = OnboardingSystem(site_manager, user_manager)
        
        print("✅ All systems initialized successfully")
        print(f"   🌳 Tree System: {'✅ LLM Available' if tree_system.llm_available else '🔄 Fallback Mode'}")
        print(f"   🎯 Onboarding System: {'✅ LLM Available' if onboarding_system.llm_available else '🔄 Fallback Mode'}")
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

# ============================================
# ONBOARDING ENDPOINTS
# ============================================

@app.post("/onboarding/start", response_model=OnboardingResponse)
async def start_onboarding(request: OnboardingRequest = None):
    """Start the onboarding process"""
    if onboarding_system is None:
        raise HTTPException(status_code=503, detail="Onboarding system not available")
    
    session_id = request.session_id if request else None
    result = onboarding_system.start_onboarding(session_id)
    
    return OnboardingResponse(
        message=result["message"],
        status=result["status"],
        session_id=result["session_id"],
        current_step=result["current_step"],
        progress=result["progress"],
        data=result.get("data", {})
    )

@app.post("/onboarding/chat", response_model=OnboardingResponse)
async def onboarding_chat(chat_request: ChatRequest):
    """Handle chat during onboarding process"""
    if onboarding_system is None:
        raise HTTPException(status_code=503, detail="Onboarding system not available")
    
    session_id = chat_request.session_id or str(uuid.uuid4())
    
    try:
        result = onboarding_system.process_onboarding_message(
            chat_request.message, 
            session_id
        )
        
        return OnboardingResponse(
            message=result["message"],
            status=result["status"],
            session_id=result["session_id"],
            current_step=result.get("current_step", "unknown"),
            progress=result.get("progress", 0),
            data=result.get("data", {})
        )
        
    except Exception as e:
        logger.error(f"Onboarding chat error: {e}")
        return OnboardingResponse(
            message=f"❌ Onboarding error: {str(e)}",
            status="error",
            session_id=session_id,
            current_step="error",
            progress=0,
            data={"error": str(e)}
        )

@app.get("/onboarding/status/{session_id}")
async def get_onboarding_status(session_id: str):
    """Get current onboarding status"""
    if onboarding_system is None:
        raise HTTPException(status_code=503, detail="Onboarding system not available")
    
    status = onboarding_system.get_onboarding_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Onboarding session not found")
    
    return status

@app.delete("/onboarding/session/{session_id}")
async def clear_onboarding_session(session_id: str):
    """Clear a specific onboarding session"""
    if onboarding_system is None:
        raise HTTPException(status_code=503, detail="Onboarding system not available")
    
    cleared = onboarding_system.clear_onboarding_session(session_id)
    return {"cleared": cleared, "session_id": session_id}

@app.delete("/onboarding/sessions")
async def clear_all_onboarding_sessions():
    """Clear all onboarding sessions"""
    if onboarding_system is None:
        raise HTTPException(status_code=503, detail="Onboarding system not available")
    
    count = onboarding_system.clear_all_onboarding_sessions()
    return {"cleared_sessions": count}

# ============================================
# ENHANCED CHAT ENDPOINT (UNIFIED)
# ============================================

@app.post("/chat", response_model=ChatResponse)
async def unified_chat(chat_request: ChatRequest):
    """Unified chat endpoint supporting both normal operations and onboarding"""
    
    session_id = chat_request.session_id or str(uuid.uuid4())
    user_message = chat_request.message.strip()
    mode = chat_request.mode or "normal"
    
    try:
        # Check if this is onboarding mode or if message indicates starting onboarding
        if (mode == "onboarding" or 
            user_message.lower() in ['start onboarding', 'onboarding', 'setup', 'get started']):
            
            if onboarding_system is None:
                raise Exception("Onboarding system not available")
            
            # Handle onboarding
            if user_message.lower() in ['start onboarding', 'onboarding', 'setup', 'get started']:
                result = onboarding_system.start_onboarding(session_id)
            else:
                result = onboarding_system.process_onboarding_message(user_message, session_id)
            
            return ChatResponse(
                message=result["message"],
                status=result["status"],
                session_id=result["session_id"],
                mode="onboarding",
                context={
                    "current_step": result.get("current_step", "unknown"),
                    "progress": result.get("progress", 0)
                },
                data=result.get("data", {})
            )
        
        else:
            # Handle normal tree-based operations
            if tree_system is None:
                raise Exception("Tree system not available")
            
            result = tree_system.process_message(user_message, session_id)
            
            return ChatResponse(
                message=result["message"],
                status=result["status"],
                session_id=result["session_id"],
                mode="normal",
                context=result.get("context", {}),
                data=result.get("result", {})
            )
        
    except Exception as e:
        logger.error(f"Unified chat error: {e}")
        return ChatResponse(
            message=f"❌ System error: {str(e)}",
            status="error",
            session_id=session_id,
            mode=mode,
            context={"error": str(e)},
            data={}
        )

# ============================================
# SITE MANAGEMENT ENDPOINTS
# ============================================

@app.get("/sites")
async def get_all_sites():
    """Get all sites"""
    if site_manager is None:
        raise HTTPException(status_code=503, detail="Site manager not available")
    
    try:
        result = site_manager.get_all_sites()
        return StandardResponse(
            success=True,
            message="Sites retrieved successfully",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sites")
async def create_site(site_data: SiteData):
    """Create a new site with full details"""
    if site_manager is None:
        raise HTTPException(status_code=503, detail="Site manager not available")
    
    try:
        result = site_manager.create_site(site_data)
        return StandardResponse(
            success=True,
            message="Site created successfully",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sites/simple")
async def create_simple_site(location_name: str = Body(..., embed=True)):
    """Create a site with just the location name"""
    if site_manager is None:
        raise HTTPException(status_code=503, detail="Site manager not available")
    
    try:
        result = site_manager.create_site_by_name_only(location_name)
        return StandardResponse(
            success=True,
            message=f"Site '{location_name}' created successfully",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sites/{site_id}")
async def delete_site(site_id: int):
    """Delete a site by ID"""
    if site_manager is None:
        raise HTTPException(status_code=503, detail="Site manager not available")
    
    try:
        result = site_manager.delete_site(site_id)
        return StandardResponse(
            success=True,
            message="Site deleted successfully",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# USER MANAGEMENT ENDPOINTS
# ============================================

@app.get("/users/permissions")
async def get_permission_bundles():
    """Get all available permission bundles"""
    if user_manager is None:
        raise HTTPException(status_code=503, detail="User manager not available")
    
    try:
        result = user_manager.get_permission_bundles()
        return StandardResponse(
            success=True,
            message="Permission bundles retrieved successfully",
            data={"permission_bundles": result}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CreateUserRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    permission_sets: List[int] = Field(default=[1107])

@app.post("/users")
async def create_user(user_request: CreateUserRequest):
    """Create a new user"""
    if user_manager is None:
        raise HTTPException(status_code=503, detail="User manager not available")
    
    try:
        from onboarding_system import UserData
        user_data = UserData(
            first_name=user_request.first_name,
            last_name=user_request.last_name,
            email=user_request.email,
            permission_sets=user_request.permission_sets
        )
        
        result = user_manager.create_user(user_data)
        return StandardResponse(
            success=True,
            message="User created successfully",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# SYSTEM STATUS ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    health_status = {
        "status": "healthy",
        "service": "PulsePro Enhanced API with Onboarding",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "components": {
            "api": "healthy",
            "authentication": "healthy" if auth_manager else "not_initialized",
            "tree_system": "connected" if tree_system else "not_available",
            "onboarding_system": "connected" if onboarding_system else "not_available",
            "site_manager": "available" if site_manager else "not_available",
            "user_manager": "available" if user_manager else "not_available"
        },
        "features": {
            "site_management": True,
            "user_management": True,
            "onboarding_flow": True,
            "conversational_ai": tree_system is not None,
            "llm_support": tree_system.llm_available if tree_system else False
        }
    }
    
    return health_status

@app.get("/status")
async def get_system_status():
    """Get detailed system status"""
    status_info = {
        "system": {
            "type": "enhanced_pulsepro",
            "initialized": all([tree_system, onboarding_system, site_manager, user_manager]),
            "components": ["tree_system", "onboarding_system", "site_manager", "user_manager"]
        },
        "tree_system": {
            "available": tree_system is not None,
            "llm_available": tree_system.llm_available if tree_system else False,
            "active_sessions": len(tree_system.sessions) if tree_system else 0,
            "supported_operations": ["CREATE", "READ", "DELETE"]
        },
        "onboarding_system": {
            "available": onboarding_system is not None,
            "llm_available": onboarding_system.llm_available if onboarding_system else False,
            "active_sessions": len(onboarding_system.sessions) if onboarding_system else 0,
            "supported_steps": ["sites", "users", "templates"]
        },
        "capabilities": {
            "site_operations": ["create", "read", "delete", "update"],
            "user_operations": ["create", "permissions"],
            "onboarding_features": ["guided_setup", "demo_site", "user_creation"],
            "conversational_features": ["intent_detection", "data_collection", "confirmation"]
        }
    }
    
    return status_info

@app.get("/sessions")
async def get_all_sessions():
    """Get information about all active sessions"""
    sessions_info = {
        "tree_sessions": len(tree_system.sessions) if tree_system else 0,
        "onboarding_sessions": len(onboarding_system.sessions) if onboarding_system else 0,
        "total_sessions": (
            (len(tree_system.sessions) if tree_system else 0) +
            (len(onboarding_system.sessions) if onboarding_system else 0)
        )
    }
    
    # Add session details if needed
    if tree_system:
        sessions_info["tree_session_ids"] = list(tree_system.sessions.keys())
    
    if onboarding_system:
        sessions_info["onboarding_session_ids"] = list(onboarding_system.sessions.keys())
    
    return sessions_info

@app.delete("/sessions")
async def clear_all_sessions():
    """Clear all sessions from both systems"""
    tree_cleared = 0
    onboarding_cleared = 0
    
    if tree_system:
        tree_cleared = tree_system.clear_all_sessions()
    
    if onboarding_system:
        onboarding_cleared = onboarding_system.clear_all_onboarding_sessions()
    
    return {
        "tree_sessions_cleared": tree_cleared,
        "onboarding_sessions_cleared": onboarding_cleared,
        "total_cleared": tree_cleared + onboarding_cleared
    }

# ============================================
# UTILITY ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PulsePro Enhanced API with Onboarding",
        "version": "4.0.0",
        "description": "Complete site and user management system with guided onboarding flow",
        "features": {
            "onboarding": "/onboarding/start",
            "chat": "/chat",
            "sites": "/sites",
            "users": "/users",
            "documentation": "/docs",
            "health": "/health"
        },
        "onboarding_flow": [
            "1. Create Sites",
            "2. Add Team Members", 
            "3. Setup Templates"
        ]
    }

@app.get("/docs-info")
async def get_docs_info():
    """Get information about available endpoints"""
    return {
        "interactive_docs": "/docs",
        "redoc_docs": "/redoc",
        "main_endpoints": {
            "onboarding": {
                "start": "POST /onboarding/start",
                "chat": "POST /onboarding/chat",
                "status": "GET /onboarding/status/{session_id}"
            },
            "chat": {
                "unified": "POST /chat (supports both normal and onboarding modes)"
            },
            "sites": {
                "list": "GET /sites",
                "create": "POST /sites",
                "create_simple": "POST /sites/simple",
                "delete": "DELETE /sites/{site_id}"
            },
            "users": {
                "create": "POST /users",
                "permissions": "GET /users/permissions"
            },
            "system": {
                "health": "GET /health",
                "status": "GET /status",
                "sessions": "GET /sessions"
            }
        }
    }

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
    
    print("🚀 PulsePro Enhanced API with Complete Onboarding")
    print("=" * 70)
    print(f"🕐 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Features: Site Management + User Management + Onboarding Flow")
    print(f"🌳 AI: Tree-Based Intent Processing + Conversational Onboarding")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🎉 Onboarding: http://localhost:8000/onboarding/start")
    print(f"💬 Chat: http://localhost:8000/chat")
    print(f"🔍 Health: http://localhost:8000/health")
    print("=" * 70)
    
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
