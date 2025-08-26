"""
Agentic AI Tool Server
FastAPI server that provides the chat interface for the AI agent
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core_agent import CoreAIAgent, AgentResponse

try:
    from site_manager import AuthenticationManager
except ImportError:
    # Fallback for development - create a mock auth manager
    class AuthenticationManager:
        def __init__(self, base_url="https://staging-api.pulsepro.ai"):
            self.base_url = base_url
            logger.warning("Using mock AuthenticationManager - some features may not work")
        
        def get_access_token(self):
            return "mock_token"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PulsePro Agentic AI Tool",
    description="AI-powered site and user management assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    status: str
    data: Optional[Dict[str, Any]] = None
    actions_taken: Optional[list] = None
    session_id: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    message: str
    agent_available: bool

# Global agent instance
agent = None

def initialize_agent():
    """Initialize the AI agent with authentication"""
    global agent
    try:
        # Initialize authentication manager
        auth_manager = AuthenticationManager()
        
        # Initialize the core agent
        agent = CoreAIAgent(auth_manager)
        logger.info("AI Agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize AI agent: {e}")
        return False

# Initialize agent on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the server starts"""
    success = initialize_agent()
    if not success:
        logger.warning("Agent initialization failed - some features may not work")

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Agentic AI Tool server is running",
        agent_available=agent is not None
    )

@app.get("/agent/health")
async def agent_health():
    """Check agent health and capabilities"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    capabilities = agent.get_agent_capabilities()
    return {
        "status": "healthy",
        "agent_initialized": True,
        "capabilities": capabilities
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for interacting with the AI agent"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="AI Agent not initialized")
        
        # Process the user request
        response = await agent.process_request(request.message, request.session_id)
        
        # Convert to response format
        return ChatResponse(
            message=response.message,
            status=response.status.value,
            data=response.data,
            actions_taken=response.actions_taken,
            session_id=response.session_id,
            timestamp=response.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Session management endpoints
@app.get("/chat/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="AI Agent not initialized")
        
        history = agent.get_session_history(session_id)
        return {"session_id": session_id, "messages": history}
        
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="AI Agent not initialized")
        
        success = agent.clear_session(session_id)
        return {"session_id": session_id, "cleared": success}
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent management endpoints
@app.get("/agent/capabilities")
async def get_agent_capabilities():
    """Get agent capabilities and available actions"""
    if not agent:
        raise HTTPException(status_code=503, detail="AI Agent not initialized")
    
    return agent.get_agent_capabilities()

@app.post("/agent/reinitialize")
async def reinitialize_agent():
    """Reinitialize the agent (useful for development)"""
    global agent
    try:
        success = initialize_agent()
        return {
            "success": success,
            "message": "Agent reinitialized successfully" if success else "Agent initialization failed"
        }
    except Exception as e:
        logger.error(f"Error reinitializing agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Development endpoints
@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to view all sessions (development only)"""
    if not agent:
        raise HTTPException(status_code=503, detail="AI Agent not initialized")
    
    try:
        stats = agent.session_manager.get_session_stats()
        active_sessions = agent.session_manager.list_active_sessions()
        
        return {
            "stats": stats,
            "active_sessions": active_sessions
        }
    except Exception as e:
        logger.error(f"Error getting debug info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/health",
            "/agent/health", 
            "/chat",
            "/agent/capabilities"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again."
    }

# Main entry point
if __name__ == "__main__":
    # Check for development mode
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    
    logger.info(f"Starting Agentic AI Tool server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
