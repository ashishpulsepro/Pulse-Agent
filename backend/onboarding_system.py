"""
onboarding_system.py
Complete Onboarding Flow System for PulsePro
Handles: Sites, Users, and Checklists/Templates
"""

import json
import logging
import requests
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

# Import existing components
from site_manager import SiteManager, AuthenticationManager

load_dotenv()
logger = logging.getLogger(__name__)

class OnboardingStep(Enum):
    """Onboarding process steps"""
    GREETING = "greeting"
    DEMO_SITE_NOTIFICATION = "demo_site_notification" 
    SITE_CREATION = "site_creation"
    SITE_CREATION_LOOP = "site_creation_loop"
    USER_CREATION = "user_creation"
    USER_CREATION_LOOP = "user_creation_loop"
    TEMPLATE_CREATION = "template_creation"
    COMPLETED = "completed"

class OnboardingState(Enum):
    """States within each step"""
    WAITING_INPUT = "waiting_input"
    COLLECTING_DATA = "collecting_data"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    MOVING_FORWARD = "moving_forward"

@dataclass
class UserData:
    """Data structure for user creation"""
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    permission_sets: List[int] = field(default_factory=list)

@dataclass
class OnboardingContext:
    """Context for onboarding process"""
    session_id: str
    current_step: OnboardingStep = OnboardingStep.GREETING
    current_state: OnboardingState = OnboardingState.WAITING_INPUT
    demo_site_created: bool = False
    sites_created: List[str] = field(default_factory=list)
    users_created: List[Dict] = field(default_factory=list)
    current_site_name: str = ""
    current_user_data: UserData = field(default_factory=UserData)
    permission_bundles: List[Dict] = field(default_factory=list)
    
    def reset_current_site(self):
        self.current_site_name = ""
    
    def reset_current_user(self):
        self.current_user_data = UserData()

class UserManager:
    """Manages user-related operations"""
    
    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
        self.base_url = auth_manager.base_url
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication"""
        access_token = self.auth_manager.get_access_token()
        return {
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'Origin': 'https://staging.pulsepro.ai',
            'Referer': 'https://staging.pulsepro.ai/',
        }
    
    def get_permission_bundles(self) -> List[Dict]:
        """Get all available permission bundles"""
        url = f"{self.base_url}/customer/get_all_permission_bundles/"
        headers = self._get_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get('permission_bundles', [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get permission bundles: {e}")
            # Return default permission sets if API fails
            return [
                {"id": 1107, "name": "Default User", "description": "Basic user permissions"}
            ]
    
    def create_user(self, user_data: UserData) -> Dict[str, Any]:
        """Create a new team member"""
        url = f"{self.base_url}/customer/add_team_member/"
        headers = self._get_headers()
        
        payload = {
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "email": user_data.email,
            "permissionSets": user_data.permission_sets
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"User '{user_data.first_name} {user_data.last_name}' created successfully")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create user: {e}")
            try:
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")
            except:
                pass
            raise Exception(f"User creation failed: {e}")

class OnboardingSystem:
    """Main onboarding system orchestrator"""
    
    def __init__(self, site_manager: SiteManager, user_manager: UserManager):
        self.site_manager = site_manager
        self.user_manager = user_manager
        self.sessions: Dict[str, OnboardingContext] = {}
        
        # Initialize LLM if available
        try:
            import ollama
            self.ollama_client = ollama.Client(host='http://localhost:11434')
            self.ollama_client.list()
            self.llm_available = True
            logger.info("✅ LLM (Ollama) connection successful for onboarding")
        except Exception as e:
            logger.warning(f"LLM not available for onboarding: {e}")
            self.ollama_client = None
            self.llm_available = False

    def start_onboarding(self, session_id: str = None) -> Dict[str, Any]:
        """Start the onboarding process"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create new onboarding context
        context = OnboardingContext(session_id=session_id)
        self.sessions[session_id] = context
        
        # Return greeting message
        greeting_message = """🎉 **Welcome to PulsePro!**

I'm your personal assistant and I'm excited to help you get set up! Let's configure your workspace together with these 3 easy steps:

**Step 1:** Create Sites 🏢
**Step 2:** Add Team Members 👥  
**Step 3:** Setup Templates 📋

I'll guide you through each step personally. Ready to begin your PulsePro journey?"""

        # Use LLM for more personalized greeting if available
        if self.llm_available:
            try:
                enhanced_greeting = self._use_llm_for_response("User starting onboarding", context, greeting_message)
                greeting_message = enhanced_greeting
            except:
                pass

        # Move to demo site notification
        context.current_step = OnboardingStep.DEMO_SITE_NOTIFICATION
        
        return {
            "message": greeting_message,
            "status": "onboarding_started",
            "session_id": session_id,
            "current_step": context.current_step.value,
            "progress": 0
        }

    def process_onboarding_message(self, message: str, session_id: str) -> Dict[str, Any]:
        """Process user message during onboarding"""
        if session_id not in self.sessions:
            return self.start_onboarding(session_id)
        
        context = self.sessions[session_id]
        
        try:
            # Route based on current step
            if context.current_step == OnboardingStep.DEMO_SITE_NOTIFICATION:
                return self._handle_demo_site_notification(message, context)
            elif context.current_step == OnboardingStep.SITE_CREATION:
                return self._handle_site_creation(message, context)
            elif context.current_step == OnboardingStep.SITE_CREATION_LOOP:
                return self._handle_site_creation_loop(message, context)
            elif context.current_step == OnboardingStep.USER_CREATION:
                return self._handle_user_creation(message, context)
            elif context.current_step == OnboardingStep.USER_CREATION_LOOP:
                return self._handle_user_creation_loop(message, context)
            elif context.current_step == OnboardingStep.TEMPLATE_CREATION:
                return self._handle_template_creation(message, context)
            elif context.current_step == OnboardingStep.COMPLETED:
                return self._handle_completed(message, context)
            else:
                return self.start_onboarding(session_id)
                
        except Exception as e:
            logger.error(f"Error in onboarding process: {e}")
            return self._create_error_response(str(e), session_id, context)

    def _handle_demo_site_notification(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Handle demo site notification - mention that demo site is already created by default"""
        if not context.demo_site_created:
            # Just mention that demo site is already available, don't actually create one
            context.demo_site_created = True
            
            demo_message = """✅ **Good news!** A demo site has already been created for you by default to help you get started.

**Step 1: Site Creation** 🏢

Would you like to create additional sites for your business locations?

Type:
- **"Yes"** to create a new site
- **"Skip"** to move to adding team members"""

            context.current_step = OnboardingStep.SITE_CREATION
            
            return {
                "message": demo_message,
                "status": "demo_site_available",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 20
            }
        else:
            # Demo site already acknowledged, move to site creation
            context.current_step = OnboardingStep.SITE_CREATION
            return self._handle_site_creation(message, context)

    def _handle_site_creation(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Handle site creation step"""
        message_lower = message.strip().lower()
        
        if message_lower in ['yes', 'y', 'create', 'ok']:
            context.current_state = OnboardingState.COLLECTING_DATA
            
            response_message = "Great! What would you like to name your new site?"
            enhanced_message = self._generate_intelligent_response(
                "User wants to create a site", 
                context, 
                response_message
            )
            
            return {
                "message": enhanced_message,
                "status": "collecting_site_name",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 25
            }
            
        elif message_lower in ['skip', 'no', 'n', 'next']:
            # Move to user creation
            context.current_step = OnboardingStep.USER_CREATION
            return self._start_user_creation_step(context)
            
        elif context.current_state == OnboardingState.COLLECTING_DATA:
            # User provided site name
            site_name = message.strip()
            context.current_site_name = site_name
            context.current_state = OnboardingState.CONFIRMING
            
            return {
                "message": f"Perfect! I'll create a site called \"{site_name}\".\n\nShould I proceed? (yes/no)",
                "status": "confirming_site_creation",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 30
            }
            
        elif context.current_state == OnboardingState.CONFIRMING:
            if message_lower in ['yes', 'y', 'proceed', 'ok']:
                # Create the site
                try:
                    created_site_name = context.current_site_name  # Store before reset
                    result = self.site_manager.create_site_by_name_only(context.current_site_name)
                    context.sites_created.append(context.current_site_name)
                    
                    # Move to site creation loop
                    context.current_step = OnboardingStep.SITE_CREATION_LOOP
                    context.reset_current_site()
                    context.current_state = OnboardingState.WAITING_INPUT
                    
                    success_message = f"✅ **Site \"{created_site_name}\" created successfully!**\n\nWould you like to create another site?\n\nType:\n- **\"Yes\"** to create another site\n- **\"Next\"** to move to adding team members"
                    
                    # Use LLM for more natural response
                    enhanced_message = self._generate_intelligent_response(
                        f"Site {created_site_name} was created successfully", 
                        context, 
                        success_message
                    )
                    
                    return {
                        "message": enhanced_message,
                        "status": "site_created",
                        "session_id": context.session_id,
                        "current_step": context.current_step.value,
                        "progress": 35,
                        "data": {"sites_created": context.sites_created}
                    }
                    
                except Exception as e:
                    return {
                        "message": f"❌ Failed to create site: {str(e)}\n\nWould you like to try again with a different name?",
                        "status": "site_creation_failed",
                        "session_id": context.session_id,
                        "current_step": context.current_step.value,
                        "progress": 30
                    }
                    
            elif message_lower in ['no', 'n', 'cancel']:
                context.reset_current_site()
                context.current_state = OnboardingState.WAITING_INPUT
                return {
                    "message": "No problem! Would you like to create a site with a different name?\n\nType:\n- **\"Yes\"** to try again\n- **\"Skip\"** to move to the next step",
                    "status": "site_creation_cancelled",
                    "session_id": context.session_id,
                    "current_step": context.current_step.value,
                    "progress": 25
                }
        
        else:
            return {
                "message": "Would you like to create a new site?\n\nType:\n- **\"Yes\"** to create a new site\n- **\"Skip\"** to move to the next step",
                "status": "awaiting_site_decision",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 20
            }

    def _handle_site_creation_loop(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Handle the site creation loop"""
        message_lower = message.strip().lower()
        
        if message_lower in ['yes', 'y', 'create', 'another']:
            context.current_state = OnboardingState.COLLECTING_DATA
            return {
                "message": "What would you like to name this new site?",
                "status": "collecting_site_name",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 35
            }
            
        elif message_lower in ['next', 'no', 'n', 'move on', 'continue']:
            # Move to user creation
            context.current_step = OnboardingStep.USER_CREATION
            return self._start_user_creation_step(context)
            
        elif context.current_state == OnboardingState.COLLECTING_DATA:
            # User provided site name
            site_name = message.strip()
            context.current_site_name = site_name
            context.current_state = OnboardingState.CONFIRMING
            
            return {
                "message": f"I'll create a site called \"{site_name}\".\n\nShould I proceed? (yes/no)",
                "status": "confirming_site_creation",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 40
            }
            
        elif context.current_state == OnboardingState.CONFIRMING:
            if message_lower in ['yes', 'y', 'proceed', 'ok']:
                # Create the site
                try:
                    created_site_name = context.current_site_name  # Store before reset
                    result = self.site_manager.create_site_by_name_only(context.current_site_name)
                    context.sites_created.append(context.current_site_name)
                    
                    context.reset_current_site()
                    context.current_state = OnboardingState.WAITING_INPUT
                    
                    success_message = f"✅ **Site \"{created_site_name}\" created successfully!**\n\nWould you like to create another site?\n\nType:\n- **\"Yes\"** to create another site\n- **\"Next\"** to move to adding team members"
                    
                    # Use LLM for more natural response
                    enhanced_message = self._generate_intelligent_response(
                        f"Another site {created_site_name} was created", 
                        context, 
                        success_message
                    )
                    
                    return {
                        "message": enhanced_message,
                        "status": "site_created",
                        "session_id": context.session_id,
                        "current_step": context.current_step.value,
                        "progress": 45,
                        "data": {"sites_created": context.sites_created}
                    }
                    
                except Exception as e:
                    return {
                        "message": f"❌ Failed to create site: {str(e)}\n\nWould you like to try again?",
                        "status": "site_creation_failed",
                        "session_id": context.session_id,
                        "current_step": context.current_step.value,
                        "progress": 40
                    }
                    
            elif message_lower in ['no', 'n', 'cancel']:
                context.reset_current_site()
                context.current_state = OnboardingState.WAITING_INPUT
                return {
                    "message": "Would you like to create another site?\n\nType:\n- **\"Yes\"** to create another site\n- **\"Next\"** to move to adding team members",
                    "status": "site_creation_cancelled",
                    "session_id": context.session_id,
                    "current_step": context.current_step.value,
                    "progress": 35
                }

    def _start_user_creation_step(self, context: OnboardingContext) -> Dict[str, Any]:
        """Start the user creation step"""
        # Get permission bundles if not already loaded
        if not context.permission_bundles:
            try:
                context.permission_bundles = self.user_manager.get_permission_bundles()
            except Exception as e:
                logger.error(f"Failed to load permission bundles: {e}")
                # Set default permission bundle
                context.permission_bundles = [
                    {"id": 1107, "name": "Default User", "description": "Basic user permissions"}
                ]
        
        sites_summary = ""
        if context.sites_created:
            sites_summary = f"\n\n**Sites Created:** {len(context.sites_created)}\n" + \
                          "\n".join([f"• {site}" for site in context.sites_created])
        
        message = f"""🎉 **Great job!** {sites_summary}

**Step 2: Add Team Members** 👥

Would you like to add a team member to your workspace?

Type:
- **\"Yes\"** to add a team member
- **\"Skip\"** to move to templates"""

        context.current_state = OnboardingState.WAITING_INPUT
        
        return {
            "message": message,
            "status": "user_creation_started",
            "session_id": context.session_id,
            "current_step": context.current_step.value,
            "progress": 50
        }

    def _handle_user_creation(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Handle user creation step"""
        message_lower = message.strip().lower()
        
        if message_lower in ['yes', 'y', 'add', 'create']:
            context.current_state = OnboardingState.COLLECTING_DATA
            context.reset_current_user()
            
            return {
                "message": "Perfect! Let's add a team member.\n\nWhat's their **first name**?",
                "status": "collecting_first_name",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 55
            }
            
        elif message_lower in ['skip', 'no', 'n', 'next']:
            # Move to template creation
            context.current_step = OnboardingStep.TEMPLATE_CREATION
            return self._start_template_creation_step(context)
            
        elif context.current_state == OnboardingState.COLLECTING_DATA:
            return self._collect_user_data(message, context)
            
        elif context.current_state == OnboardingState.CONFIRMING:
            return self._confirm_user_creation(message, context)
        
        else:
            return {
                "message": "Would you like to add a team member?\n\nType:\n- **\"Yes\"** to add a team member\n- **\"Skip\"** to move to templates",
                "status": "awaiting_user_decision",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 50
            }

    def _collect_user_data(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Collect user data step by step"""
        user_data = context.current_user_data
        
        if not user_data.first_name:
            user_data.first_name = message.strip()
            
            response_message = f"Great! First name: **{user_data.first_name}**\n\nWhat's their **last name**?"
            enhanced_message = self._generate_intelligent_response(
                f"User provided first name: {user_data.first_name}", 
                context, 
                response_message
            )
            
            return {
                "message": enhanced_message,
                "status": "collecting_last_name",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 60
            }
            
        elif not user_data.last_name:
            user_data.last_name = message.strip()
            
            response_message = f"Perfect! Name: **{user_data.first_name} {user_data.last_name}**\n\nWhat's their **email address**?"
            enhanced_message = self._generate_intelligent_response(
                f"User provided last name: {user_data.last_name}", 
                context, 
                response_message
            )
            
            return {
                "message": enhanced_message,
                "status": "collecting_email",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 65
            }
            
        elif not user_data.email:
            email = message.strip()
            # Basic email validation
            if '@' not in email or '.' not in email:
                return {
                    "message": "Please provide a valid email address:",
                    "status": "invalid_email",
                    "session_id": context.session_id,
                    "current_step": context.current_step.value,
                    "progress": 65
                }
            
            user_data.email = email
            
            # Set default permission set
            if context.permission_bundles:
                user_data.permission_sets = [context.permission_bundles[0]["id"]]
            else:
                user_data.permission_sets = [1107]  # Default
            
            # Move to confirmation
            context.current_state = OnboardingState.CONFIRMING
            
            permission_name = context.permission_bundles[0]["name"] if context.permission_bundles else "Default User"
            
            return {
                "message": f"""**User Details:**
• Name: {user_data.first_name} {user_data.last_name}
• Email: {user_data.email}
• Permissions: {permission_name}

Should I create this user? (yes/no)""",
                "status": "confirming_user_creation",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 70
            }

    def _confirm_user_creation(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Confirm and create user"""
        message_lower = message.strip().lower()
        
        if message_lower in ['yes', 'y', 'proceed', 'create']:
            try:
                # Create the user
                created_user_name = f"{context.current_user_data.first_name} {context.current_user_data.last_name}"  # Store before reset
                result = self.user_manager.create_user(context.current_user_data)
                
                user_info = {
                    "name": f"{context.current_user_data.first_name} {context.current_user_data.last_name}",
                    "email": context.current_user_data.email
                }
                context.users_created.append(user_info)
                
                # Move to user creation loop
                context.current_step = OnboardingStep.USER_CREATION_LOOP
                context.reset_current_user()
                context.current_state = OnboardingState.WAITING_INPUT
                
                success_message = f"✅ **User created successfully!**\n\nWould you like to add another team member?\n\nType:\n- **\"Yes\"** to add another user\n- **\"Next\"** to move to templates"
                
                # Use LLM for more natural response
                enhanced_message = self._generate_intelligent_response(
                    f"User {created_user_name} was created", 
                    context, 
                    success_message
                )
                
                return {
                    "message": enhanced_message,
                    "status": "user_created",
                    "session_id": context.session_id,
                    "current_step": context.current_step.value,
                    "progress": 75,
                    "data": {"users_created": context.users_created}
                }
                
            except Exception as e:
                return {
                    "message": f"❌ Failed to create user: {str(e)}\n\nWould you like to try again?",
                    "status": "user_creation_failed",
                    "session_id": context.session_id,
                    "current_step": context.current_step.value,
                    "progress": 70
                }
                
        elif message_lower in ['no', 'n', 'cancel']:
            context.reset_current_user()
            context.current_state = OnboardingState.WAITING_INPUT
            return {
                "message": "Would you like to add a team member?\n\nType:\n- **\"Yes\"** to add a team member\n- **\"Skip\"** to move to templates",
                "status": "user_creation_cancelled",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 55
            }

    def _handle_user_creation_loop(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Handle the user creation loop"""
        message_lower = message.strip().lower()
        
        if message_lower in ['yes', 'y', 'add', 'another']:
            context.current_state = OnboardingState.COLLECTING_DATA
            context.reset_current_user()
            
            return {
                "message": "Perfect! Let's add another team member.\n\nWhat's their **first name**?",
                "status": "collecting_first_name",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 75
            }
            
        elif message_lower in ['next', 'no', 'n', 'continue', 'templates']:
            # Move to template creation
            context.current_step = OnboardingStep.TEMPLATE_CREATION
            return self._start_template_creation_step(context)
            
        elif context.current_state == OnboardingState.COLLECTING_DATA:
            return self._collect_user_data(message, context)
            
        elif context.current_state == OnboardingState.CONFIRMING:
            return self._confirm_user_creation(message, context)

    def _start_template_creation_step(self, context: OnboardingContext) -> Dict[str, Any]:
        """Start the template creation step"""
        sites_summary = ""
        if context.sites_created:
            sites_summary = f"\n**Sites Created:** {len(context.sites_created)}"
        
        users_summary = ""
        if context.users_created:
            users_summary = f"\n**Users Created:** {len(context.users_created)}"
        
        message = f"""🎉 **Excellent progress!**{sites_summary}{users_summary}

**Step 3: Templates & Checklists** 📋

Templates help you create standardized checklists for your team.

Would you like to create a new template?

Type:
- **\"Yes\"** to create a template
- **\"Finish\"** to complete onboarding"""

        context.current_state = OnboardingState.WAITING_INPUT
        
        return {
            "message": message,
            "status": "template_creation_started",
            "session_id": context.session_id,
            "current_step": context.current_step.value,
            "progress": 80
        }

    def _handle_template_creation(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Handle template creation step"""
        message_lower = message.strip().lower()
        
        if message_lower in ['yes', 'y', 'create']:
            # For now, just acknowledge and move to completion
            # Template creation logic would be implemented here
            context.current_step = OnboardingStep.COMPLETED
            
            return {
                "message": "Template creation is a powerful feature that will be available soon!\n\nFor now, let's complete your onboarding.",
                "status": "template_creation_placeholder",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 90
            }
            
        elif message_lower in ['finish', 'complete', 'no', 'n', 'skip']:
            context.current_step = OnboardingStep.COMPLETED
            return self._complete_onboarding(context)
        
        else:
            return {
                "message": "Would you like to create a template?\n\nType:\n- **\"Yes\"** to create a template\n- **\"Finish\"** to complete onboarding",
                "status": "awaiting_template_decision",
                "session_id": context.session_id,
                "current_step": context.current_step.value,
                "progress": 80
            }

    def _handle_completed(self, message: str, context: OnboardingContext) -> Dict[str, Any]:
        """Handle completed onboarding"""
        return self._complete_onboarding(context)

    def _complete_onboarding(self, context: OnboardingContext) -> Dict[str, Any]:
        """Complete the onboarding process"""
        sites_summary = ""
        if context.sites_created:
            sites_summary = f"\n\n**🏢 Sites Created ({len(context.sites_created)}):**\n" + \
                          "\n".join([f"• {site}" for site in context.sites_created])
        
        users_summary = ""
        if context.users_created:
            users_summary = f"\n\n**👥 Team Members Added ({len(context.users_created)}):**\n" + \
                          "\n".join([f"• {user['name']} ({user['email']})" for user in context.users_created])
        
        completion_message = f"""🎉 **Congratulations! Onboarding Complete!**

You've successfully set up your PulsePro workspace:{sites_summary}{users_summary}

**What's Next:**
• Explore your sites and assign team members
• Create checklists and templates (coming soon!)
• Start managing your operations with PulsePro

**Need Help?**
Just ask me about creating, viewing, or deleting sites anytime!

Welcome to PulsePro! 🚀"""

        # Clean up session
        if context.session_id in self.sessions:
            del self.sessions[context.session_id]
        
        return {
            "message": completion_message,
            "status": "onboarding_completed",
            "session_id": context.session_id,
            "current_step": "completed",
            "progress": 100,
            "summary": {
                "sites_created": context.sites_created,
                "users_created": context.users_created,
                "total_sites": len(context.sites_created),
                "total_users": len(context.users_created)
            }
        }

    def _use_llm_for_response(self, user_message: str, context: OnboardingContext, fallback_response: str) -> str:
        """Use LLM to generate more natural responses during onboarding"""
        if not self.llm_available:
            return fallback_response
        
        try:
            # Create context for LLM
            step_info = {
                "current_step": context.current_step.value,
                "sites_created": len(context.sites_created),
                "users_created": len(context.users_created),
                "demo_site_available": context.demo_site_created
            }
            
            prompt = f"""You are a helpful onboarding assistant for PulsePro, a business management platform. 
You are currently helping a user through their setup process.

Current context:
- Step: {context.current_step.value}
- Sites created: {len(context.sites_created)}
- Users created: {len(context.users_created)}
- User just said: "{user_message}"

Please provide a friendly, helpful response that:
1. Acknowledges what the user said
2. Guides them to the next step in onboarding
3. Keeps the tone professional but warm
4. Is concise but informative

Fallback response if you're unsure: {fallback_response}

Your response:"""

            response = self.ollama_client.generate(
                model='llama3.2:latest',
                prompt=prompt
            )
            
            llm_response = response['response'].strip()
            
            # Use LLM response if it's reasonable, otherwise fallback
            if len(llm_response) > 10 and len(llm_response) < 500:
                return llm_response
            else:
                return fallback_response
                
        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}")
            return fallback_response

    def _generate_intelligent_response(self, user_message: str, context: OnboardingContext, base_message: str) -> str:
        """Generate more intelligent responses based on user input"""
        if not self.llm_available:
            return base_message
        
        try:
            # Analyze user intent and generate appropriate response
            prompt = f"""You are an AI assistant helping with PulsePro onboarding. The user said: "{user_message}"

Current onboarding step: {context.current_step.value}
Sites created so far: {len(context.sites_created)}
Users created so far: {len(context.users_created)}

Generate a helpful, conversational response that:
- Acknowledges their input naturally
- Guides them through the next step
- Maintains a friendly, professional tone
- Is concise but informative

Base response to enhance: {base_message}

Enhanced response:"""

            response = self.ollama_client.generate(
                model='llama3.2:latest',
                prompt=prompt
            )
            
            enhanced_response = response['response'].strip()
            
            # Validate response quality
            if 20 <= len(enhanced_response) <= 300 and not enhanced_response.lower().startswith('i cannot') and not enhanced_response.lower().startswith('sorry'):
                return enhanced_response
            else:
                return base_message
                
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return base_message

    def _create_error_response(self, error_message: str, session_id: str, context: OnboardingContext) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "message": f"❌ **Error:** {error_message}\n\nWould you like to restart the onboarding process?",
            "status": "onboarding_error",
            "session_id": session_id,
            "current_step": context.current_step.value if context else "unknown",
            "error": error_message
        }

    def get_onboarding_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current onboarding status"""
        if session_id not in self.sessions:
            return None
        
        context = self.sessions[session_id]
        return {
            "session_id": session_id,
            "current_step": context.current_step.value,
            "current_state": context.current_state.value,
            "sites_created": context.sites_created,
            "users_created": context.users_created,
            "progress": self._calculate_progress(context)
        }

    def _calculate_progress(self, context: OnboardingContext) -> int:
        """Calculate onboarding progress percentage"""
        step_progress = {
            OnboardingStep.GREETING: 0,
            OnboardingStep.DEMO_SITE_NOTIFICATION: 10,
            OnboardingStep.SITE_CREATION: 20,
            OnboardingStep.SITE_CREATION_LOOP: 40,
            OnboardingStep.USER_CREATION: 50,
            OnboardingStep.USER_CREATION_LOOP: 70,
            OnboardingStep.TEMPLATE_CREATION: 80,
            OnboardingStep.COMPLETED: 100
        }
        
        base_progress = step_progress.get(context.current_step, 0)
        
        # Add bonus for items created
        bonus_progress = min(len(context.sites_created) * 5 + len(context.users_created) * 5, 20)
        
        return min(base_progress + bonus_progress, 100)

    def clear_onboarding_session(self, session_id: str) -> bool:
        """Clear a specific onboarding session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def clear_all_onboarding_sessions(self) -> int:
        """Clear all onboarding sessions"""
        count = len(self.sessions)
        self.sessions.clear()
        return count

# Test function
def test_onboarding_system():
    """Test the onboarding system"""
    try:
        # Initialize components
        auth_manager = AuthenticationManager()
        site_manager = SiteManager(auth_manager)
        user_manager = UserManager(auth_manager)
        onboarding_system = OnboardingSystem(site_manager, user_manager)
        
        print("🎉 Onboarding System initialized successfully!")
        
        # Test starting onboarding
        result = onboarding_system.start_onboarding()
        print(f"Start result: {result['message']}")
        
        return onboarding_system
        
    except Exception as e:
        print(f"❌ Onboarding system initialization failed: {e}")
        return None

if __name__ == "__main__":
    test_onboarding_system()
