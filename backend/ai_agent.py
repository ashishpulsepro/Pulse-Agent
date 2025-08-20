"""
AI_agent_ollama.py
Ollama-based Conversational AI Agent for PulsePro Site Management
Uses local Ollama with llama3.1:8b model
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import logging
from datetime import datetime

# Import from our site_manager.py
from site_manager import SiteManager, SiteData, AuthenticationManager

# For Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("Ollama not available. Install with: pip install ollama")
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """States of conversation flow"""
    IDLE = "idle"
    INTENT_RECOGNITION = "intent_recognition"
    GATHERING_INFO = "gathering_info"
    CONFIRMATION = "confirmation"
    EXECUTING = "executing"
    COMPLETED = "completed"

class Intent(Enum):
    """User intents"""
    CREATE_SITE = "create_site"
    UPDATE_SITE = "update_site"
    DELETE_SITE = "delete_site"
    SHOW_SITES = "show_sites"
    SEARCH_SITES = "search_sites"
    ASSIGN_USERS = "assign_users"
    UNASSIGN_USERS = "unassign_users"
    GET_ANALYTICS = "get_analytics"
    HELP = "help"
    UNKNOWN = "unknown"

@dataclass
class ConversationContext:
    """Maintains conversation state and gathered information"""
    state: ConversationState = ConversationState.IDLE
    intent: Optional[Intent] = None
    gathered_data: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    current_field: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class OllamaConversationalAgent:
    """Conversational AI agent using Ollama with Llama3.1:8b"""
    
    def __init__(self, model_name: str = "llama3.1:8b", site_manager: SiteManager = None):
        self.model_name = model_name
        self.site_manager = site_manager
        self.conversations: Dict[str, ConversationContext] = {}
        self.ollama_client = None
        
        # Initialize Ollama client if available
        if OLLAMA_AVAILABLE:
            self._initialize_ollama()
        else:
            logger.warning("Ollama not available, using fallback responses")
        
        # Define field requirements for different operations
        self.field_requirements = {
            Intent.CREATE_SITE: {
                'required': ['location_name', 'address_field1', 'country_id', 'state_id', 'city_id', 'reporting_timezone'],
                'optional': ['address_field2', 'pincode', 'mobile', 'location_number', 'location_code', 
                           'to_email', 'cc_email', 'geo_fencing_enabled', 'geo_fencing_distance', 
                           'lat', 'lng', 'map_link', 'has_custom_field', 'is_schedule_active']
            },
            Intent.UPDATE_SITE: {
                'required': ['location_id'],
                'optional': ['location_name', 'address_field1', 'address_field2', 'pincode', 'mobile', 
                           'location_number', 'location_code', 'to_email', 'cc_email', 'geo_fencing_enabled']
            },
            Intent.DELETE_SITE: {
                'required': ['location_id'],
                'optional': []
            },
            Intent.ASSIGN_USERS: {
                'required': ['location_id', 'user_ids'],
                'optional': []
            },
            Intent.UNASSIGN_USERS: {
                'required': ['mapped_location_ids'],
                'optional': []
            },
            Intent.SEARCH_SITES: {
                'required': [],
                'optional': ['name', 'city', 'state', 'country']
            }
        }
        
        # Field descriptions for better user understanding
        self.field_descriptions = {
            'location_name': 'Name of the site/location (e.g., "Mumbai Office", "Delhi Store")',
            'address_field1': 'Primary address (e.g., "123 Main Street", "Bandra West")',
            'address_field2': 'Secondary address (optional - building, suite, etc.)',
            'country_id': 'Country ID (1 for India, 2 for USA)',
            'state_id': 'State ID (numeric identifier for the state)',
            'city_id': 'City ID (numeric identifier for the city)',
            'pincode': 'Postal/ZIP code',
            'mobile': 'Mobile/phone number',
            'location_number': 'Internal location number/identifier',
            'location_code': 'Location code for internal reference',
            'to_email': 'Primary email address for this location',
            'cc_email': 'CC email address',
            'reporting_timezone': 'Timezone for reporting (e.g., "UTC", "Asia/Kolkata")',
            'geo_fencing_enabled': 'Enable geo-fencing? (yes/no)',
            'geo_fencing_distance': 'Geo-fencing distance in meters',
            'location_id': 'ID of the location to modify',
            'user_ids': 'List of user IDs to assign (comma-separated)',
            'mapped_location_ids': 'List of mapped location IDs to unassign'
        }

    def _initialize_ollama(self):
        """Initialize the Ollama client"""
        try:
            logger.info(f"Initializing Ollama with model: {self.model_name}")
            
            # Test connection to Ollama
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model_name in available_models:
                logger.info(f"✅ Found model {self.model_name} in Ollama")
                self.ollama_client = ollama
                
                # Test the model with a simple prompt
                test_response = self.ollama_client.generate(
                    model=self.model_name,
                    prompt="Hello",
                    options={"num_predict": 5}
                )
                
                logger.info("✅ Ollama model test successful")
                
            else:
                logger.error(f"❌ Model {self.model_name} not found in Ollama")
                logger.info(f"Available models: {available_models}")
                logger.info(f"Run: ollama pull {self.model_name}")
                self.ollama_client = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            self.ollama_client = None

    def get_conversation_context(self, session_id: str) -> ConversationContext:
        """Get or create conversation context for a session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(session_id=session_id)
        return self.conversations[session_id]

    def process_message(self, message: str, session_id: str, user_id: str = None) -> Dict[str, Any]:
        """Process user message and return AI response"""
        context = self.get_conversation_context(session_id)
        context.user_id = user_id
        
        # Add user message to history
        context.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # Process based on current state
            if context.state == ConversationState.IDLE:
                response = self._handle_intent_recognition(message, context)
            elif context.state == ConversationState.GATHERING_INFO:
                response = self._handle_info_gathering(message, context)
            elif context.state == ConversationState.CONFIRMATION:
                response = self._handle_confirmation(message, context)
            else:
                response = self._generate_error_response("I'm not sure how to help with that.")
            
            # Add AI response to history
            context.conversation_history.append({
                "role": "assistant",
                "content": response["message"],
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._generate_error_response("I encountered an error. Please try again.")

    def _handle_intent_recognition(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Recognize user intent from the message"""
        intent = self._classify_intent(message)
        context.intent = intent
        
        if intent == Intent.UNKNOWN:
            return self._generate_help_response()
        
        if intent == Intent.HELP:
            return self._generate_help_response()
        
        if intent == Intent.SHOW_SITES:
            return self._execute_show_sites(context)
        
        if intent == Intent.GET_ANALYTICS:
            return self._execute_analytics(context)
        
        # For operations requiring information gathering
        if intent in [Intent.CREATE_SITE, Intent.UPDATE_SITE, Intent.DELETE_SITE, 
                     Intent.ASSIGN_USERS, Intent.UNASSIGN_USERS, Intent.SEARCH_SITES]:
            return self._start_info_gathering(intent, context, message)
        
        return self._generate_error_response("I didn't understand that request.")

    def _classify_intent(self, message: str) -> Intent:
        """Classify user intent using Ollama or fallback"""
        if self.ollama_client:
            return self._classify_intent_with_ollama(message)
        else:
            return self._classify_intent_fallback(message)

    def _classify_intent_with_ollama(self, message: str) -> Intent:
        """Classify intent using Ollama"""
        prompt = f"""You are an AI assistant for site management. Classify the user's intent from their message.

Available intents:
- create_site: User wants to create a new site/location
- update_site: User wants to update an existing site
- delete_site: User wants to delete a site
- show_sites: User wants to view/list sites
- search_sites: User wants to search for specific sites
- assign_users: User wants to assign users to a site
- unassign_users: User wants to remove users from a site
- get_analytics: User wants analytics or reports
- help: User needs help

User message: "{message}"

Respond with only the intent name (e.g., "create_site"):"""
        
        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 10,
                    "stop": ["\n", ".", ","]
                }
            )
            
            intent_text = response['response'].strip().lower()
            
            # Map response to intent
            intent_mapping = {
                'create_site': Intent.CREATE_SITE,
                'update_site': Intent.UPDATE_SITE,
                'delete_site': Intent.DELETE_SITE,
                'show_sites': Intent.SHOW_SITES,
                'search_sites': Intent.SEARCH_SITES,
                'assign_users': Intent.ASSIGN_USERS,
                'unassign_users': Intent.UNASSIGN_USERS,
                'get_analytics': Intent.GET_ANALYTICS,
                'help': Intent.HELP
            }
            
            for intent_name, intent_enum in intent_mapping.items():
                if intent_name in intent_text:
                    return intent_enum
            
            return Intent.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error in Ollama intent classification: {e}")
            return self._classify_intent_fallback(message)

    def _classify_intent_fallback(self, message: str) -> Intent:
        """Fallback keyword-based intent classification"""
        message_lower = message.lower()
        
        # Keywords for different intents
        create_keywords = ['create', 'add', 'new', 'make', 'build', 'establish']
        update_keywords = ['update', 'edit', 'modify', 'change', 'alter']
        delete_keywords = ['delete', 'remove', 'destroy', 'eliminate']
        show_keywords = ['show', 'list', 'display', 'view', 'get all', 'see']
        search_keywords = ['search', 'find', 'look for', 'filter']
        assign_keywords = ['assign', 'add user', 'attach user']
        unassign_keywords = ['unassign', 'remove user', 'detach user']
        analytics_keywords = ['analytics', 'summary', 'statistics', 'report']
        help_keywords = ['help', 'how', 'what can', 'commands']
        
        if any(keyword in message_lower for keyword in create_keywords) and 'site' in message_lower:
            return Intent.CREATE_SITE
        elif any(keyword in message_lower for keyword in update_keywords) and 'site' in message_lower:
            return Intent.UPDATE_SITE
        elif any(keyword in message_lower for keyword in delete_keywords) and 'site' in message_lower:
            return Intent.DELETE_SITE
        elif any(keyword in message_lower for keyword in show_keywords) and 'site' in message_lower:
            return Intent.SHOW_SITES
        elif any(keyword in message_lower for keyword in search_keywords) and 'site' in message_lower:
            return Intent.SEARCH_SITES
        elif any(keyword in message_lower for keyword in assign_keywords):
            return Intent.ASSIGN_USERS
        elif any(keyword in message_lower for keyword in unassign_keywords):
            return Intent.UNASSIGN_USERS
        elif any(keyword in message_lower for keyword in analytics_keywords):
            return Intent.GET_ANALYTICS
        elif any(keyword in message_lower for keyword in help_keywords):
            return Intent.HELP
        else:
            return Intent.UNKNOWN

    def _start_info_gathering(self, intent: Intent, context: ConversationContext, initial_message: str) -> Dict[str, Any]:
        """Start the information gathering process"""
        context.state = ConversationState.GATHERING_INFO
        context.required_fields = self.field_requirements[intent]['required'].copy()
        context.optional_fields = self.field_requirements[intent]['optional'].copy()
        
        # Try to extract information from initial message
        extracted_info = self._extract_info_from_message(initial_message, context)
        
        # Update gathered data
        for key, value in extracted_info.items():
            if key in context.required_fields or key in context.optional_fields:
                context.gathered_data[key] = value
                if key in context.required_fields:
                    context.required_fields.remove(key)
                if key in context.optional_fields:
                    context.optional_fields.remove(key)
        
        # Ask for next required field
        return self._ask_for_next_field(context)

    def _handle_info_gathering(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle information gathering phase"""
        # Check for cancellation
        if message.lower() in ['cancel', 'stop', 'quit', 'exit']:
            context.state = ConversationState.IDLE
            return {
                "message": "Operation cancelled. How else can I help you?",
                "status": "cancelled",
                "context": self._serialize_context(context)
            }
        
        # Check for skip requests
        if message.lower() in ['skip', 'no', 'not needed', 'optional']:
            if context.current_field and context.current_field in context.optional_fields:
                context.optional_fields.remove(context.current_field)
                return self._ask_for_next_field(context)
            else:
                return {
                    "message": "This field is required. Please provide a value or type 'cancel' to stop.",
                    "status": "info_needed",
                    "context": self._serialize_context(context)
                }
        
        # Extract information from user response
        if context.current_field:
            value = self._parse_field_value(message, context.current_field)
            context.gathered_data[context.current_field] = value
            
            # Remove from required fields
            if context.current_field in context.required_fields:
                context.required_fields.remove(context.current_field)
            if context.current_field in context.optional_fields:
                context.optional_fields.remove(context.current_field)
        
        # Ask for next field or proceed to confirmation
        return self._ask_for_next_field(context)

    def _ask_for_next_field(self, context: ConversationContext) -> Dict[str, Any]:
        """Ask for the next required field or move to confirmation"""
        # Check if all required fields are gathered
        if not context.required_fields:
            return self._move_to_confirmation(context)
        
        # Get next required field
        next_field = context.required_fields[0]
        context.current_field = next_field
        
        # Generate question for the field
        question = self._generate_field_question(next_field, context)
        
        return {
            "message": question,
            "status": "info_needed",
            "context": self._serialize_context(context),
            "current_field": next_field,
            "remaining_required": len(context.required_fields),
            "remaining_optional": len(context.optional_fields)
        }

    def _generate_field_question(self, field: str, context: ConversationContext) -> str:
        """Generate a natural question for a specific field"""
        if self.ollama_client:
            return self._generate_field_question_with_ollama(field, context)
        else:
            return self._generate_field_question_fallback(field)

    def _generate_field_question_with_ollama(self, field: str, context: ConversationContext) -> str:
        """Generate question using Ollama"""
        description = self.field_descriptions.get(field, field)
        intent_name = context.intent.value if context.intent else "operation"
        
        prompt = f"""Generate a friendly, natural question to ask a user for the field "{field}" when doing {intent_name}.

Field description: {description}

Requirements:
- Ask in a conversational, friendly tone
- Keep it short and clear
- End with a question mark
- Don't include technical jargon

Question:"""
        
        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 30
                }
            )
            
            question = response['response'].strip()
            
            # Clean up the response
            question = question.split('\n')[0].strip()
            if not question.endswith('?'):
                question += '?'
            
            return question
            
        except Exception as e:
            logger.error(f"Error generating field question with Ollama: {e}")
            return self._generate_field_question_fallback(field)

    def _generate_field_question_fallback(self, field: str) -> str:
        """Fallback method for generating field questions"""
        questions = {
            'location_name': "What would you like to name this site?",
            'address_field1': "What's the primary address for this location?",
            'address_field2': "Any additional address details? (You can skip this if not needed)",
            'country_id': "What's the country ID? (1 for India, 2 for USA)",
            'state_id': "What's the state ID? (numeric identifier)",
            'city_id': "What's the city ID? (numeric identifier)",
            'pincode': "What's the postal/ZIP code?",
            'mobile': "What's the mobile/phone number for this location?",
            'location_number': "Any internal location number/identifier?",
            'location_code': "Any location code for internal reference?",
            'to_email': "What's the primary email address for this location?",
            'cc_email': "Any CC email address?",
            'reporting_timezone': "What timezone should be used for reporting? (e.g., UTC, Asia/Kolkata)",
            'geo_fencing_enabled': "Should geo-fencing be enabled? (yes/no)",
            'geo_fencing_distance': "What should be the geo-fencing distance in meters?",
            'location_id': "What's the ID of the location you want to modify?",
            'user_ids': "Please provide the user IDs to assign (comma-separated)",
            'mapped_location_ids': "Please provide the mapped location IDs to unassign (comma-separated)"
        }
        
        return questions.get(field, f"Please provide the {field}:")

    def _extract_info_from_message(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Extract information from user message"""
        # Simple pattern matching for common fields
        extracted = {}
        message_lower = message.lower()
        
        # Extract email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        if emails and 'to_email' in (context.required_fields + context.optional_fields):
            extracted['to_email'] = emails[0]
        
        # Extract phone numbers
        phone_pattern = r'[\+]?[1-9]?[0-9]{7,15}'
        phones = re.findall(phone_pattern, message)
        if phones and 'mobile' in (context.required_fields + context.optional_fields):
            extracted['mobile'] = phones[0]
        
        # Extract numbers for IDs
        if any('id' in field for field in context.required_fields):
            numbers = re.findall(r'\b\d+\b', message)
            if numbers:
                if 'location_id' in context.required_fields:
                    extracted['location_id'] = int(numbers[0])
                elif 'country_id' in context.required_fields:
                    extracted['country_id'] = int(numbers[0])
        
        return extracted

    def _parse_field_value(self, message: str, field: str) -> Any:
        """Parse user input for a specific field"""
        message = message.strip()
        
        # Boolean fields
        if field in ['geo_fencing_enabled', 'has_custom_field', 'is_schedule_active']:
            return message.lower() in ['yes', 'y', 'true', '1', 'enable', 'enabled']
        
        # Numeric fields
        if field in ['country_id', 'state_id', 'city_id', 'geo_fencing_distance', 'location_id']:
            try:
                return int(message)
            except ValueError:
                # Try to extract first number from message
                numbers = re.findall(r'\d+', message)
                return int(numbers[0]) if numbers else 0
        
        # Float fields
        if field in ['lat', 'lng']:
            try:
                return float(message)
            except ValueError:
                return 0.0
        
        # List fields
        if field in ['user_ids', 'mapped_location_ids']:
            # Parse comma-separated values
            if ',' in message:
                return [int(x.strip()) for x in message.split(',') if x.strip().isdigit()]
            else:
                return [int(message)] if message.isdigit() else []
        
        # String fields (default)
        return message

    def _move_to_confirmation(self, context: ConversationContext) -> Dict[str, Any]:
        """Move to confirmation phase"""
        context.state = ConversationState.CONFIRMATION
        
        # Generate confirmation message
        confirmation_msg = self._generate_confirmation_message(context)
        
        return {
            "message": confirmation_msg,
            "status": "confirmation_needed",
            "context": self._serialize_context(context),
            "gathered_data": context.gathered_data
        }

    def _generate_confirmation_message(self, context: ConversationContext) -> str:
        """Generate confirmation message"""
        intent_name = context.intent.value.replace('_', ' ').title() if context.intent else "Operation"
        data_summary = self._format_data_summary(context.gathered_data)
        
        return f"""Please confirm this {intent_name}:

{data_summary}

Is this correct? (yes/no):"""

    def _format_data_summary(self, data: Dict[str, Any]) -> str:
        """Format gathered data for display"""
        if not data:
            return "No data collected"
        
        summary_lines = []
        for key, value in data.items():
            if value:  # Only show non-empty values
                formatted_key = key.replace('_', ' ').title()
                summary_lines.append(f"• {formatted_key}: {value}")
        
        return '\n'.join(summary_lines)

    def _handle_confirmation(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """Handle confirmation phase"""
        message_lower = message.lower().strip()
        
        if message_lower in ['yes', 'y', 'confirm', 'ok', 'proceed']:
            # Execute the operation
            return self._execute_operation(context)
        elif message_lower in ['no', 'n', 'cancel', 'stop']:
            # Cancel operation
            context.state = ConversationState.IDLE
            return {
                "message": "Operation cancelled. How else can I help you?",
                "status": "cancelled",
                "context": self._serialize_context(context)
            }
        else:
            return {
                "message": "Please confirm with 'yes' or 'no':",
                "status": "confirmation_needed",
                "context": self._serialize_context(context)
            }

    def _execute_operation(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute the confirmed operation"""
        context.state = ConversationState.EXECUTING
        
        try:
            if context.intent == Intent.CREATE_SITE:
                result = self._execute_create_site(context)
            elif context.intent == Intent.UPDATE_SITE:
                result = self._execute_update_site(context)
            elif context.intent == Intent.DELETE_SITE:
                result = self._execute_delete_site(context)
            elif context.intent == Intent.ASSIGN_USERS:
                result = self._execute_assign_users(context)
            elif context.intent == Intent.UNASSIGN_USERS:
                result = self._execute_unassign_users(context)
            elif context.intent == Intent.SEARCH_SITES:
                result = self._execute_search_sites(context)
            else:
                result = self._generate_error_response("Unsupported operation")
            
            # Reset context
            self._reset_context(context)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing operation: {e}")
            self._reset_context(context)
            return self._generate_error_response(f"Operation failed: {str(e)}")

    def _reset_context(self, context: ConversationContext):
        """Reset conversation context to idle state"""
        context.state = ConversationState.IDLE
        context.intent = None
        context.gathered_data = {}
        context.required_fields = []
        context.optional_fields = []
        context.current_field = None

    def _execute_create_site(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute site creation"""
        if not self.site_manager:
            return self._generate_error_response("Site manager not available")
        
        try:
            # Create SiteData object from gathered information
            site_data = SiteData(**context.gathered_data)
            result = self.site_manager.create_site(site_data)
            
            return {
                "message": f"✅ Site '{context.gathered_data.get('location_name')}' created successfully!",
                "status": "completed",
                "data": result,
                "context": self._serialize_context(context)
            }
        except Exception as e:
            return self._generate_error_response(f"Failed to create site: {str(e)}")

    def _execute_update_site(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute site update"""
        if not self.site_manager:
            return self._generate_error_response("Site manager not available")
        
        try:
            location_id = context.gathered_data.pop('location_id')
            site_data = SiteData(**context.gathered_data)
            result = self.site_manager.update_site(location_id, site_data)
            
            return {
                "message": f"✅ Site {location_id} updated successfully!",
                "status": "completed",
                "data": result,
                "context": self._serialize_context(context)
            }
        except Exception as e:
            return self._generate_error_response(f"Failed to update site: {str(e)}")

    def _execute_delete_site(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute site deletion"""
        if not self.site_manager:
            return self._generate_error_response("Site manager not available")
        
        try:
            location_id = context.gathered_data.get('location_id')
            result = self.site_manager.delete_site(location_id)
            
            return {
                "message": f"✅ Site {location_id} deleted successfully!",
                "status": "completed",
                "data": result,
                "context": self._serialize_context(context)
            }
        except Exception as e:
            return self._generate_error_response(f"Failed to delete site: {str(e)}")

    def _execute_assign_users(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute user assignment"""
        if not self.site_manager:
            return self._generate_error_response("Site manager not available")
        
        try:
            location_id = context.gathered_data.get('location_id')
            user_ids = context.gathered_data.get('user_ids', [])
            result = self.site_manager.assign_users_to_site(location_id, user_ids)
            
            return {
                "message": f"✅ {len(user_ids)} users assigned to site {location_id} successfully!",
                "status": "completed",
                "data": result,
                "context": self._serialize_context(context)
            }
        except Exception as e:
            return self._generate_error_response(f"Failed to assign users: {str(e)}")

    def _execute_unassign_users(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute user unassignment"""
        if not self.site_manager:
            return self._generate_error_response("Site manager not available")
        
        try:
            mapped_location_ids = context.gathered_data.get('mapped_location_ids', [])
            result = self.site_manager.unassign_users_from_site(mapped_location_ids)
            
            return {
                "message": f"✅ {len(mapped_location_ids)} user mappings removed successfully!",
                "status": "completed",
                "data": result,
                "context": self._serialize_context(context)
            }
        except Exception as e:
            return self._generate_error_response(f"Failed to unassign users: {str(e)}")

    def _execute_search_sites(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute site search"""
        if not self.site_manager:
            return self._generate_error_response("Site manager not available")
        
        try:
            # Get all sites and filter based on gathered criteria
            all_sites = self.site_manager.get_all_sites()
            sites = all_sites.get('locations', [])
            
            # Apply filters
            filtered_sites = sites
            
            search_criteria = context.gathered_data
            if search_criteria.get('name'):
                filtered_sites = [s for s in filtered_sites 
                                if search_criteria['name'].lower() in s.get('location_name', '').lower()]
            
            if search_criteria.get('city'):
                filtered_sites = [s for s in filtered_sites 
                                if search_criteria['city'].lower() in s.get('city', '').lower()]
            
            if search_criteria.get('state'):
                filtered_sites = [s for s in filtered_sites 
                                if search_criteria['state'].lower() in s.get('state', '').lower()]
            
            if search_criteria.get('country'):
                filtered_sites = [s for s in filtered_sites 
                                if search_criteria['country'].lower() in s.get('country', '').lower()]
            
            return {
                "message": f"🔍 Found {len(filtered_sites)} sites matching your criteria:",
                "status": "completed",
                "data": {
                    "sites": filtered_sites,
                    "total_count": len(filtered_sites),
                    "search_criteria": search_criteria
                },
                "context": self._serialize_context(context)
            }
        except Exception as e:
            return self._generate_error_response(f"Failed to search sites: {str(e)}")

    def _execute_show_sites(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute show all sites"""
        if not self.site_manager:
            return self._generate_error_response("Site manager not available")
        
        try:
            result = self.site_manager.get_all_sites()
            sites = result.get('locations', [])
            
            return {
                "message": f"📍 Found {len(sites)} sites in total:",
                "status": "completed",
                "data": result,
                "context": self._serialize_context(context)
            }
        except Exception as e:
            return self._generate_error_response(f"Failed to retrieve sites: {str(e)}")

    def _execute_analytics(self, context: ConversationContext) -> Dict[str, Any]:
        """Execute analytics request"""
        if not self.site_manager:
            return self._generate_error_response("Site manager not available")
        
        try:
            # Get all sites for analytics
            all_sites = self.site_manager.get_all_sites()
            sites = all_sites.get('locations', [])
            
            # Calculate analytics
            analytics = self._calculate_analytics(sites)
            
            return {
                "message": "📊 Here's your site analytics summary:",
                "status": "completed",
                "data": analytics,
                "context": self._serialize_context(context)
            }
        except Exception as e:
            return self._generate_error_response(f"Failed to generate analytics: {str(e)}")

    def _calculate_analytics(self, sites: List[Dict]) -> Dict[str, Any]:
        """Calculate analytics from sites data"""
        total_sites = len(sites)
        
        # Country distribution
        countries = {}
        states = {}
        geo_fencing_enabled = 0
        sites_with_mobile = 0
        sites_with_email = 0
        
        for site in sites:
            # Country stats
            country = site.get('country', 'Unknown')
            countries[country] = countries.get(country, 0) + 1
            
            # State stats
            state = site.get('state', 'Unknown')
            states[state] = states.get(state, 0) + 1
            
            # Feature stats
            if site.get('geo_fencing_enabled'):
                geo_fencing_enabled += 1
            if site.get('mobile'):
                sites_with_mobile += 1
            if site.get('to_email'):
                sites_with_email += 1
        
        return {
            "total_sites": total_sites,
            "distribution": {
                "by_country": countries,
                "by_state": states
            },
            "features": {
                "geo_fencing_enabled": geo_fencing_enabled,
                "sites_with_mobile": sites_with_mobile,
                "sites_with_email": sites_with_email,
                "completion_rates": {
                    "mobile": round((sites_with_mobile / total_sites) * 100, 2) if total_sites > 0 else 0,
                    "email": round((sites_with_email / total_sites) * 100, 2) if total_sites > 0 else 0,
                    "geo_fencing": round((geo_fencing_enabled / total_sites) * 100, 2) if total_sites > 0 else 0
                }
            }
        }

    def _generate_help_response(self) -> Dict[str, Any]:
        """Generate help response"""
        help_message = """🤖 I'm your AI assistant for site management! Here's what I can help you with:

**Site Operations:**
• Create new sites - "I want to create a new site"
• Update existing sites - "Update site information"  
• Delete sites - "Delete a site"
• Show all sites - "Show me all sites"
• Search sites - "Find sites in Mumbai"

**User Management:**
• Assign users to sites - "Assign users to a site"
• Remove users from sites - "Remove users from a site"

**Analytics & Reports:**
• Get site analytics - "Show me site analytics"
• Generate reports - "Give me a summary report"

Just tell me what you'd like to do in natural language, and I'll guide you through the process step by step!"""
        
        return {
            "message": help_message,
            "status": "help",
            "context": {}
        }

    def _generate_error_response(self, message: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "message": f"❌ {message}",
            "status": "error",
            "context": {}
        }

    def _serialize_context(self, context: ConversationContext) -> Dict[str, Any]:
        """Serialize context for response"""
        return {
            "state": context.state.value,
            "intent": context.intent.value if context.intent else None,
            "gathered_data": context.gathered_data,
            "required_fields": context.required_fields,
            "optional_fields": context.optional_fields,
            "current_field": context.current_field,
            "session_id": context.session_id
        }

    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        context = self.get_conversation_context(session_id)
        return context.conversation_history


class OllamaIntegratedSiteManager:
    """Extended Site Manager with integrated Ollama conversational agent"""
    
    def __init__(self, auth_manager: AuthenticationManager, model_name: str = "llama3.1:8b"):
        self.site_manager = SiteManager(auth_manager)
        self.conversational_agent = OllamaConversationalAgent(model_name, self.site_manager)
        self.auth_manager = auth_manager
    
    def chat(self, message: str, session_id: str, user_id: str = None) -> Dict[str, Any]:
        """Main chat interface"""
        return self.conversational_agent.process_message(message, session_id, user_id)
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation for session"""
        self.conversational_agent.clear_conversation(session_id)
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversational_agent.get_conversation_history(session_id)
    
    # Delegate all site manager methods
    def create_site(self, site_data: SiteData) -> Dict[str, Any]:
        return self.site_manager.create_site(site_data)
    
    def update_site(self, location_id: int, site_data: SiteData) -> Dict[str, Any]:
        return self.site_manager.update_site(location_id, site_data)
    
    def get_all_sites(self) -> Dict[str, Any]:
        return self.site_manager.get_all_sites()
    
    def get_site_by_id(self, location_id: int) -> Optional[Dict[str, Any]]:
        return self.site_manager.get_site_by_id(location_id)
    
    def delete_site(self, location_id: int) -> Dict[str, Any]:
        return self.site_manager.delete_site(location_id)
    
    def assign_users_to_site(self, location_id: int, user_ids: List[int]) -> Dict[str, Any]:
        return self.site_manager.assign_users_to_site(location_id, user_ids)
    
    def unassign_users_from_site(self, mapped_location_ids: List[int]) -> Dict[str, Any]:
        return self.site_manager.unassign_users_from_site(mapped_location_ids)
    
    def get_site_users(self, location_id: int, search_keyword: str = "") -> Dict[str, Any]:
        return self.site_manager.get_site_users(location_id, search_keyword)


# Test function for Ollama connection
def test_ollama_connection():
    """Test if Ollama is running and model is available"""
    try:
        import ollama
        
        print("🔍 Testing Ollama connection...")
        
        # List available models
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        
        print(f"Available models: {available_models}")
        
        # Test llama3.1:8b specifically
        if "llama3.1:8b" in available_models:
            print("✅ llama3.1:8b found!")
            
            # Test generation
            response = ollama.generate(
                model="llama3.1:8b",
                prompt="Hello",
                options={"num_predict": 5}
            )
            
            print(f"Test response: {response['response']}")
            print("✅ Ollama connection successful!")
            return True
            
        else:
            print("❌ llama3.1:8b not found")
            print("Run: ollama pull llama3.1:8b")
            return False
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False

if __name__ == "__main__":
    test_ollama_connection()