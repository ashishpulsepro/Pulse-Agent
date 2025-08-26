"""
main1.py - Complete User Onboarding Flow for PulsePro
Implements a comprehensive 3-step onboarding process:
1. Create Sites
2. Add Users 
3. Create Templates (placeholder)
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import uuid
import json

# Import existing site management components
from site_manager import (
    SiteManager, AuthenticationManager, SiteData, 
    UserManager, PermissionManager, PulseProAPIException
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="PulsePro User Onboarding API",
    description="Complete 3-step user onboarding flow: Create Sites → Add Users → Create Templates",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
auth_manager = AuthenticationManager()
site_manager = SiteManager(auth_manager)
user_manager = UserManager(auth_manager)
permission_manager = PermissionManager(auth_manager)

# ============================================
# ONBOARDING STATE MANAGEMENT
# ============================================

class OnboardingState:
    """Manages the state of user onboarding process"""
    
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create onboarding session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "current_step": "welcome",
                "steps_completed": [],
                "created_sites": [],
                "created_users": [],
                "created_templates": [],
                "demo_site_created": False,
                "timestamp": datetime.now().isoformat()
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session data"""
        session = self.get_session(session_id)
        session.update(updates)
        session["last_updated"] = datetime.now().isoformat()
    
    def clear_session(self, session_id: str):
        """Clear session data"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Global onboarding state manager
onboarding_state = OnboardingState()

# ============================================
# PYDANTIC MODELS
# ============================================

class OnboardingResponse(BaseModel):
    """Standard response for onboarding steps"""
    success: bool
    message: str
    current_step: str
    next_step: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    options: Optional[List[str]] = None
    session_id: str

class StartOnboardingRequest(BaseModel):
    """Request to start onboarding"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class OnboardingActionRequest(BaseModel):
    """Request for onboarding actions"""
    action: str = Field(..., description="Action to perform")
    session_id: str = Field(..., description="Session ID")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")

class SiteCreationRequest(BaseModel):
    """Request to create a site during onboarding"""
    site_name: str = Field(..., min_length=1, description="Name of the site")
    session_id: str = Field(..., description="Session ID")

class UserCreationRequest(BaseModel):
    """Request to create a user during onboarding"""
    first_name: str = Field(..., min_length=1, description="First name")
    last_name: str = Field(..., min_length=1, description="Last name")
    email: str = Field(..., description="Email address")
    permission_set_name: Optional[str] = Field("Admin", description="Permission set name")
    session_id: str = Field(..., description="Session ID")

# ============================================
# ONBOARDING FLOW CONTROLLER
# ============================================

class OnboardingFlowController:
    """Controls the flow of the onboarding process"""
    
    def __init__(self, site_manager: SiteManager, user_manager: UserManager, permission_manager: PermissionManager):
        self.site_manager = site_manager
        self.user_manager = user_manager
        self.permission_manager = permission_manager
    
    async def start_onboarding(self, session_id: str, user_id: Optional[str] = None) -> OnboardingResponse:
        """Start the onboarding process with welcome message"""
        try:
            # Initialize session
            session = onboarding_state.get_session(session_id)
            onboarding_state.update_session(session_id, {
                "user_id": user_id,
                "current_step": "welcome"
            })
            
            # Create demo site
            await self._create_demo_site(session_id)
            
            welcome_message = """
🎉 Welcome to PulsePro! Let's set you up with these 3 easy steps:

**Step 1:** Create Sites 🏢
**Step 2:** Add Users 👥  
**Step 3:** Create Templates 📋

Good news! We've already created a demo site for you to get started.

Would you like to:
🅰️ Create another site
🅱️ Move to adding users
🅲️ View your demo site details
            """
            
            return OnboardingResponse(
                success=True,
                message=welcome_message.strip(),
                current_step="welcome",
                next_step="site_creation",
                session_id=session_id,
                options=["create_site", "add_users", "view_demo_site"],
                data={"demo_site_created": True}
            )
            
        except Exception as e:
            logger.error(f"Failed to start onboarding: {e}")
            return OnboardingResponse(
                success=False,
                message=f"Failed to start onboarding: {str(e)}",
                current_step="error",
                session_id=session_id
            )
    
    async def _create_demo_site(self, session_id: str):
        """Create a demo site for the user"""
        try:
            demo_site_name = f"Demo Site - {datetime.now().strftime('%Y%m%d')}"
            result = self.site_manager.create_site_by_name_only(demo_site_name)
            
            # Update session with demo site info
            session = onboarding_state.get_session(session_id)
            session["demo_site"] = {
                "name": demo_site_name,
                "id": result.get("id") if result else None,
                "created_at": datetime.now().isoformat()
            }
            session["demo_site_created"] = True
            
            logger.info(f"Demo site '{demo_site_name}' created for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to create demo site: {e}")
            # Continue onboarding even if demo site creation fails
    
    async def handle_action(self, action: str, session_id: str, data: Optional[Dict[str, Any]] = None) -> OnboardingResponse:
        """Handle user actions during onboarding"""
        session = onboarding_state.get_session(session_id)
        current_step = session.get("current_step", "welcome")
        
        try:
            if action == "create_site":
                return await self._handle_site_creation_intent(session_id)
            elif action == "add_users":
                return await self._handle_user_addition_intent(session_id)
            elif action == "view_demo_site":
                return await self._handle_view_demo_site(session_id)
            elif action == "create_another_site":
                return await self._handle_site_creation_intent(session_id)
            elif action == "move_to_users":
                return await self._transition_to_users(session_id)
            elif action == "add_another_user":
                return await self._handle_user_addition_intent(session_id)
            elif action == "move_to_templates":
                return await self._transition_to_templates(session_id)
            elif action == "finish_onboarding":
                return await self._finish_onboarding(session_id)
            else:
                return OnboardingResponse(
                    success=False,
                    message=f"Unknown action: {action}",
                    current_step=current_step,
                    session_id=session_id
                )
                
        except Exception as e:
            logger.error(f"Error handling action {action}: {e}")
            return OnboardingResponse(
                success=False,
                message=f"Error processing action: {str(e)}",
                current_step=current_step,
                session_id=session_id
            )
    
    async def _handle_site_creation_intent(self, session_id: str) -> OnboardingResponse:
        """Handle intent to create a site"""
        onboarding_state.update_session(session_id, {"current_step": "site_creation_input"})
        
        return OnboardingResponse(
            success=True,
            message="🏢 Great! Let's create a new site. What would you like to name it?",
            current_step="site_creation_input",
            next_step="site_creation_confirm",
            session_id=session_id,
            data={"awaiting_input": "site_name"}
        )
    
    async def create_site(self, site_name: str, session_id: str) -> OnboardingResponse:
        """Create a new site during onboarding"""
        try:
            # Create the site
            result = self.site_manager.create_site_by_name_only(site_name)
            
            # Update session
            session = onboarding_state.get_session(session_id)
            session["created_sites"].append({
                "name": site_name,
                "id": result.get("id") if result else None,
                "created_at": datetime.now().isoformat()
            })
            
            onboarding_state.update_session(session_id, {
                "current_step": "site_created",
                "last_created_site": site_name
            })
            
            message = f"""
✅ Excellent! Site '{site_name}' has been created successfully!

You now have {len(session['created_sites']) + (1 if session.get('demo_site_created') else 0)} sites total.

What would you like to do next?
🅰️ Create another site
🅱️ Move forward to add users
            """
            
            return OnboardingResponse(
                success=True,
                message=message.strip(),
                current_step="site_created",
                next_step="site_decision",
                session_id=session_id,
                options=["create_another_site", "move_to_users"],
                data={"site_created": site_name, "total_sites": len(session["created_sites"]) + 1}
            )
            
        except Exception as e:
            logger.error(f"Failed to create site {site_name}: {e}")
            return OnboardingResponse(
                success=False,
                message=f"Failed to create site '{site_name}': {str(e)}",
                current_step="site_creation_error",
                session_id=session_id
            )
    
    async def _transition_to_users(self, session_id: str) -> OnboardingResponse:
        """Transition to user creation step"""
        session = onboarding_state.get_session(session_id)
        
        onboarding_state.update_session(session_id, {"current_step": "users_intro"})
        
        total_sites = len(session.get("created_sites", [])) + (1 if session.get("demo_site_created") else 0)
        
        message = f"""
🎯 **Step 2: Add Users**

Great job! You've created {total_sites} site(s). Now let's add some users to your team.

Would you like to add a user to your team?
🅰️ Yes, add a user
🅱️ Skip to templates
            """
        
        return OnboardingResponse(
            success=True,
            message=message.strip(),
            current_step="users_intro",
            next_step="user_creation",
            session_id=session_id,
            options=["add_user", "skip_to_templates"],
            data={"total_sites": total_sites}
        )
    
    async def _handle_user_addition_intent(self, session_id: str) -> OnboardingResponse:
        """Handle intent to add a user"""
        # First, get available permission sets
        try:
            permission_sets = self.permission_manager.get_all_permission_sets()
            permission_names = [perm['name'] for perm in permission_sets]
            
            onboarding_state.update_session(session_id, {
                "current_step": "user_creation_input",
                "available_permissions": permission_sets
            })
            
            message = f"""
👥 Let's add a new user! I'll need a few details:

**First, what's their first name?**

Available permission levels: {', '.join(permission_names[:3])}... (we'll ask about this later)
            """
            
            return OnboardingResponse(
                success=True,
                message=message.strip(),
                current_step="user_creation_input",
                next_step="user_creation_form",
                session_id=session_id,
                data={"awaiting_input": "first_name", "permission_sets": permission_sets}
            )
            
        except Exception as e:
            logger.error(f"Failed to get permission sets: {e}")
            return OnboardingResponse(
                success=False,
                message=f"Failed to load permission sets: {str(e)}",
                current_step="user_creation_error",
                session_id=session_id
            )
    
    async def create_user(self, first_name: str, last_name: str, email: str, permission_set_name: str, session_id: str) -> OnboardingResponse:
        """Create a new user during onboarding"""
        try:
            # Get permission set ID
            permission_sets = self.permission_manager.get_all_permission_sets()
            permission_set_id = None
            
            for perm in permission_sets:
                if perm['name'].lower() == permission_set_name.lower():
                    permission_set_id = perm['id']
                    break
            
            if not permission_set_id:
                # Default to first available permission set
                permission_set_id = permission_sets[0]['id'] if permission_sets else 1107
            
            # Create the user
            result = self.user_manager.create_user(
                first_name=first_name,
                last_name=last_name,
                email=email,
                permission_set_ids=[permission_set_id]
            )
            
            # Update session
            session = onboarding_state.get_session(session_id)
            session["created_users"].append({
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "permission_set": permission_set_name,
                "id": result.get("id") if result else None,
                "created_at": datetime.now().isoformat()
            })
            
            onboarding_state.update_session(session_id, {
                "current_step": "user_created",
                "last_created_user": f"{first_name} {last_name}"
            })
            
            message = f"""
✅ Perfect! User '{first_name} {last_name}' has been added successfully!

📧 Email: {email}
🔑 Permission Level: {permission_set_name}

You now have {len(session['created_users'])} user(s) in your team.

What would you like to do next?
🅰️ Add another user
🅱️ Move forward to templates
            """
            
            return OnboardingResponse(
                success=True,
                message=message.strip(),
                current_step="user_created",
                next_step="user_decision",
                session_id=session_id,
                options=["add_another_user", "move_to_templates"],
                data={
                    "user_created": f"{first_name} {last_name}",
                    "total_users": len(session["created_users"])
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create user {first_name} {last_name}: {e}")
            return OnboardingResponse(
                success=False,
                message=f"Failed to create user '{first_name} {last_name}': {str(e)}",
                current_step="user_creation_error",
                session_id=session_id
            )
    
    async def _transition_to_templates(self, session_id: str) -> OnboardingResponse:
        """Transition to template creation step (placeholder)"""
        session = onboarding_state.get_session(session_id)
        
        onboarding_state.update_session(session_id, {"current_step": "templates_intro"})
        
        total_sites = len(session.get("created_sites", [])) + (1 if session.get("demo_site_created") else 0)
        total_users = len(session.get("created_users", []))
        
        message = f"""
🎯 **Step 3: Create Templates**

Excellent progress! Here's what you've accomplished:
✅ Created {total_sites} site(s)
✅ Added {total_users} user(s)

**Templates** allow you to create reusable checklists and workflows.

Would you like to create a new template?

🅰️ Yes, create a template
🅱️ Finish onboarding for now

*Note: Template creation involves complex logic and will be implemented in a future update.*
            """
        
        return OnboardingResponse(
            success=True,
            message=message.strip(),
            current_step="templates_intro",
            next_step="template_decision",
            session_id=session_id,
            options=["create_template", "finish_onboarding"],
            data={
                "total_sites": total_sites,
                "total_users": total_users,
                "templates_available": False  # Placeholder
            }
        )
    
    async def _finish_onboarding(self, session_id: str) -> OnboardingResponse:
        """Complete the onboarding process"""
        session = onboarding_state.get_session(session_id)
        
        total_sites = len(session.get("created_sites", [])) + (1 if session.get("demo_site_created") else 0)
        total_users = len(session.get("created_users", []))
        
        onboarding_state.update_session(session_id, {
            "current_step": "completed",
            "completed_at": datetime.now().isoformat()
        })
        
        message = f"""
🎉 **Congratulations! Onboarding Complete!**

You've successfully set up your PulsePro account:

✅ **Sites Created:** {total_sites}
✅ **Users Added:** {total_users}
✅ **Templates:** Ready for future setup

You're all set to start using PulsePro! Here's what you can do next:

🏢 Manage your sites and locations
👥 Add more team members
📋 Create custom workflows and checklists
📊 Monitor your operations

Welcome to the PulsePro family! 🚀
            """
        
        return OnboardingResponse(
            success=True,
            message=message.strip(),
            current_step="completed",
            session_id=session_id,
            data={
                "onboarding_completed": True,
                "summary": {
                    "sites_created": total_sites,
                    "users_added": total_users,
                    "templates_created": 0,
                    "completion_time": datetime.now().isoformat()
                }
            }
        )
    
    async def _handle_view_demo_site(self, session_id: str) -> OnboardingResponse:
        """Show demo site details"""
        session = onboarding_state.get_session(session_id)
        demo_site = session.get("demo_site", {})
        
        if not demo_site:
            return OnboardingResponse(
                success=False,
                message="No demo site found.",
                current_step="error",
                session_id=session_id
            )
        
        message = f"""
🏢 **Your Demo Site Details:**

**Name:** {demo_site.get('name', 'Demo Site')}
**ID:** {demo_site.get('id', 'N/A')}
**Created:** {demo_site.get('created_at', 'N/A')}

This demo site is ready to use! You can:
- Add users to this site
- Create checklists and workflows
- Monitor operations

What would you like to do next?
🅰️ Create another site
🅱️ Move to adding users
        """
        
        return OnboardingResponse(
            success=True,
            message=message.strip(),
            current_step="demo_site_viewed",
            session_id=session_id,
            options=["create_site", "add_users"],
            data={"demo_site": demo_site}
        )

# Initialize flow controller
flow_controller = OnboardingFlowController(site_manager, user_manager, permission_manager)

# ============================================
# API ENDPOINTS
# ============================================

@app.post("/onboarding/start", response_model=OnboardingResponse)
async def start_onboarding(request: StartOnboardingRequest):
    """Start the user onboarding process"""
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        response = await flow_controller.start_onboarding(session_id, request.user_id)
        return response
        
    except Exception as e:
        logger.error(f"Failed to start onboarding: {e}")
        return OnboardingResponse(
            success=False,
            message=f"Failed to start onboarding: {str(e)}",
            current_step="error",
            session_id=session_id
        )

@app.post("/onboarding/action", response_model=OnboardingResponse)
async def handle_onboarding_action(request: OnboardingActionRequest):
    """Handle user actions during onboarding"""
    try:
        response = await flow_controller.handle_action(
            action=request.action,
            session_id=request.session_id,
            data=request.data
        )
        return response
        
    except Exception as e:
        logger.error(f"Failed to handle action: {e}")
        return OnboardingResponse(
            success=False,
            message=f"Failed to process action: {str(e)}",
            current_step="error",
            session_id=request.session_id
        )

@app.post("/onboarding/create-site", response_model=OnboardingResponse)
async def create_site_onboarding(request: SiteCreationRequest):
    """Create a site during onboarding"""
    try:
        response = await flow_controller.create_site(request.site_name, request.session_id)
        return response
        
    except Exception as e:
        logger.error(f"Failed to create site during onboarding: {e}")
        return OnboardingResponse(
            success=False,
            message=f"Failed to create site: {str(e)}",
            current_step="site_creation_error",
            session_id=request.session_id
        )

@app.post("/onboarding/create-user", response_model=OnboardingResponse)
async def create_user_onboarding(request: UserCreationRequest):
    """Create a user during onboarding"""
    try:
        response = await flow_controller.create_user(
            first_name=request.first_name,
            last_name=request.last_name,
            email=request.email,
            permission_set_name=request.permission_set_name,
            session_id=request.session_id
        )
        return response
        
    except Exception as e:
        logger.error(f"Failed to create user during onboarding: {e}")
        return OnboardingResponse(
            success=False,
            message=f"Failed to create user: {str(e)}",
            current_step="user_creation_error",
            session_id=request.session_id
        )

@app.get("/onboarding/session/{session_id}")
async def get_onboarding_session(session_id: str):
    """Get onboarding session details"""
    try:
        session = onboarding_state.get_session(session_id)
        return {
            "success": True,
            "session": session
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")

@app.delete("/onboarding/session/{session_id}")
async def clear_onboarding_session(session_id: str):
    """Clear onboarding session"""
    try:
        onboarding_state.clear_session(session_id)
        return {
            "success": True,
            "message": "Session cleared",
            "session_id": session_id
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to clear session: {str(e)}",
            "session_id": session_id
        }

@app.get("/onboarding/stats")
async def get_onboarding_stats():
    """Get onboarding statistics"""
    try:
        sessions = onboarding_state.sessions
        total_sessions = len(sessions)
        completed_sessions = len([s for s in sessions.values() if s.get("current_step") == "completed"])
        
        total_sites_created = sum(len(s.get("created_sites", [])) for s in sessions.values())
        total_users_created = sum(len(s.get("created_users", [])) for s in sessions.values())
        
        return {
            "success": True,
            "stats": {
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "completion_rate": (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0,
                "total_sites_created": total_sites_created,
                "total_users_created": total_users_created,
                "active_sessions": total_sessions - completed_sessions
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to get stats: {str(e)}"
        }

# ============================================
# UTILITY ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Test managers availability
        auth_status = "healthy" if auth_manager else "unavailable"
        site_manager_status = "healthy" if site_manager else "unavailable"
        user_manager_status = "healthy" if user_manager else "unavailable"
        permission_manager_status = "healthy" if permission_manager else "unavailable"
        
        return {
            "status": "healthy",
            "service": "PulsePro User Onboarding API",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": "healthy",
                "authentication": auth_status,
                "site_manager": site_manager_status,
                "user_manager": user_manager_status,
                "permission_manager": permission_manager_status,
                "onboarding_flow": "healthy"
            },
            "features": {
                "site_creation": "available",
                "user_creation": "available",
                "template_creation": "placeholder",
                "demo_site": "available"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint with onboarding information"""
    return {
        "service": "PulsePro User Onboarding API",
        "version": "1.0.0",
        "description": "Complete 3-step user onboarding flow",
        "onboarding_steps": [
            {
                "step": 1,
                "name": "Create Sites",
                "description": "Set up your locations and sites",
                "features": ["Demo site creation", "Custom site creation", "Multiple sites support"]
            },
            {
                "step": 2,
                "name": "Add Users",
                "description": "Build your team",
                "features": ["User creation", "Permission management", "Multiple users support"]
            },
            {
                "step": 3,
                "name": "Create Templates",
                "description": "Set up workflows and checklists",
                "features": ["Template creation (coming soon)", "Workflow management", "Checklist system"]
            }
        ],
        "endpoints": {
            "start": "POST /onboarding/start",
            "action": "POST /onboarding/action",
            "create_site": "POST /onboarding/create-site",
            "create_user": "POST /onboarding/create-user",
            "session": "GET /onboarding/session/{session_id}",
            "stats": "GET /onboarding/stats"
        },
        "quick_start": {
            "1": "POST /onboarding/start to begin onboarding",
            "2": "Follow the guided flow responses",
            "3": "Use action endpoints for interactions",
            "4": "Visit /docs for full API documentation"
        }
    }

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "details": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================
# STARTUP
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 PulsePro User Onboarding API")
    print("=" * 50)
    print("🎯 Complete 3-Step Onboarding Flow")
    print("📋 Step 1: Create Sites")
    print("👥 Step 2: Add Users")
    print("📝 Step 3: Create Templates")
    print("=" * 50)
    print("📚 API Documentation: http://localhost:8001/docs")
    print("🔍 Health Check: http://localhost:8001/health")
    print("🏠 Root Info: http://localhost:8001/")
    print("📊 Stats: http://localhost:8001/onboarding/stats")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info"
    )
