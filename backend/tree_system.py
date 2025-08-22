"""
Tree-based Intent System for PulsePro Site Management
Clear structure: Intent Extraction -> Action Confirmation -> Execution
Only handles: CREATE, READ (VIEW), DELETE
"""

import json
import logging
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import uuid
from datetime import datetime

# Import existing components
from site_manager import SiteManager, AuthenticationManager

logger = logging.getLogger(__name__)

class Intent(Enum):
    """Only three supported intents"""
    CREATE = "CREATE"
    READ = "READ"  # VIEW/SHOW sites
    DELETE = "DELETE"
    UNKNOWN = "UNKNOWN"

class ProcessState(Enum):
    """Process states in the tree"""
    INTENT_EXTRACTION = "intent_extraction"
    DATA_COLLECTION = "data_collection"
    CONFIRMATION = "confirmation"
    EXECUTION = "execution"
    COMPLETED = "completed"

@dataclass
class ProcessContext:
    """Context for the tree process"""
    session_id: str
    state: ProcessState = ProcessState.INTENT_EXTRACTION
    intent: Optional[Intent] = None
    collected_data: Dict[str, Any] = None
    ready_for_execution: bool = False
    
    def __post_init__(self):
        if self.collected_data is None:
            self.collected_data = {}

class TreeBasedSystem:
    """Tree-based intent processing system"""
    
    def __init__(self, site_manager: SiteManager):
        self.site_manager = site_manager
        self.sessions: Dict[str, ProcessContext] = {}
        
        # Initialize Ollama client for LLM operations
        try:
            import ollama
            self.ollama_client = ollama.Client(host='http://localhost:11434')
            # Test connection
            self.ollama_client.list()
            self.llm_available = True
            logger.info("✅ LLM (Ollama) connection successful")
        except Exception as e:
            logger.warning(f"LLM not available: {e}")
            self.ollama_client = None
            self.llm_available = False

    def process_message(self, message: str, session_id: str = None) -> Dict[str, Any]:
        """Main entry point for processing user messages"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get or create context
        if session_id not in self.sessions:
            self.sessions[session_id] = ProcessContext(session_id=session_id)
        
        context = self.sessions[session_id]
        
        try:
            # Route based on current state
            if context.state == ProcessState.INTENT_EXTRACTION:
                return self._handle_intent_extraction(message, context)
            elif context.state == ProcessState.DATA_COLLECTION:
                return self._handle_data_collection(message, context)
            elif context.state == ProcessState.CONFIRMATION:
                return self._handle_confirmation(message, context)
            else:
                # Reset if in unknown state
                context.state = ProcessState.INTENT_EXTRACTION
                return self._handle_intent_extraction(message, context)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._create_error_response(str(e), session_id)

    def _handle_intent_extraction(self, message: str, context: ProcessContext) -> Dict[str, Any]:
        """Step 1: Extract and confirm intent"""
        intent = self._extract_intent(message)
        
        if intent == Intent.UNKNOWN:
            return {
                "message": "I can only help with these operations:\n• CREATE a new site\n• VIEW/READ all sites\n• DELETE a site\n\nPlease specify what you'd like to do.",
                "status": "intent_needed",
                "session_id": context.session_id,
                "process_state": "intent_extraction"
            }
        
        context.intent = intent
        
        # For READ operations, no additional data needed - move to confirmation
        if intent == Intent.READ:
            context.state = ProcessState.CONFIRMATION
            return {
                "message": f"Intent confirmed: {intent.value}\nYou want to VIEW all sites. Confirm to proceed? (yes/no)",
                "status": "confirmation_needed",
                "session_id": context.session_id,
                "process_state": "confirmation",
                "intent": intent.value
            }
        
        # For CREATE and DELETE, move to data collection
        context.state = ProcessState.DATA_COLLECTION
        return self._request_required_data(context)

    def _extract_intent(self, message: str) -> Intent:
        """Extract intent using LLM or fallback keyword matching"""
        if self.llm_available:
            return self._extract_intent_with_llm(message)
        else:
            return self._extract_intent_fallback(message)

    def _extract_intent_with_llm(self, message: str) -> Intent:
        """Extract intent using LLM"""
        prompt = f"""Analyze this user message and determine their intent. Respond with ONLY one word:
- CREATE (if they want to create/add/make a new site)
- READ (if they want to view/show/list/see sites)  
- DELETE (if they want to delete/remove a site)
- UNKNOWN (if unclear or not related to site operations)

User message: "{message}"

Intent:"""

        try:
            response = self.ollama_client.generate(
                model="llama3.1:8b",
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "num_predict": 5,
                    "stop": ["\n", " ", "."]
                }
            )
            
            intent_text = response['response'].strip().upper()
            
            # Map to enum
            intent_mapping = {
                'CREATE': Intent.CREATE,
                'READ': Intent.READ,
                'DELETE': Intent.DELETE
            }
            
            return intent_mapping.get(intent_text, Intent.UNKNOWN)
            
        except Exception as e:
            logger.error(f"LLM intent extraction failed: {e}")
            return self._extract_intent_fallback(message)

    def _extract_intent_fallback(self, message: str) -> Intent:
        """Fallback keyword-based intent extraction"""
        message_lower = message.lower()
        
        # Keywords for each intent
        create_keywords = ['create', 'add', 'new', 'make', 'build']
        read_keywords = ['show', 'list', 'view', 'see', 'display', 'get', 'read']
        delete_keywords = ['delete', 'remove', 'destroy', 'drop']
        
        # Check for site-related keywords to ensure relevance
        site_keywords = ['site', 'location', 'office', 'store', 'branch']
        has_site_context = any(keyword in message_lower for keyword in site_keywords)
        
        if any(keyword in message_lower for keyword in create_keywords) and has_site_context:
            return Intent.CREATE
        elif any(keyword in message_lower for keyword in read_keywords) and has_site_context:
            return Intent.READ
        elif any(keyword in message_lower for keyword in delete_keywords) and has_site_context:
            return Intent.DELETE
        else:
            return Intent.UNKNOWN

    def _request_required_data(self, context: ProcessContext) -> Dict[str, Any]:
        """Step 2: Request required data based on intent"""
        if context.intent == Intent.CREATE:
            if not context.collected_data.get('location_name'):
                return {
                    "message": "Intent confirmed: CREATE\nWhat should be the name of the new site?",
                    "status": "data_needed",
                    "session_id": context.session_id,
                    "process_state": "data_collection",
                    "intent": context.intent.value,
                    "required_field": "location_name"
                }
        
        elif context.intent == Intent.DELETE:
            if not context.collected_data.get('location_name'):
                return {
                    "message": "Intent confirmed: DELETE\nWhich site do you want to delete? (provide the site name)",
                    "status": "data_needed", 
                    "session_id": context.session_id,
                    "process_state": "data_collection",
                    "intent": context.intent.value,
                    "required_field": "location_name"
                }
        
        # If we reach here, all data is collected
        context.state = ProcessState.CONFIRMATION
        return self._generate_confirmation_message(context)

    def _handle_data_collection(self, message: str, context: ProcessContext) -> Dict[str, Any]:
        """Step 2: Collect required data"""
        message = message.strip()
        
        # Check for cancellation
        if message.lower() in ['cancel', 'stop', 'quit', 'exit']:
            del self.sessions[context.session_id]
            return {
                "message": "Operation cancelled.",
                "status": "cancelled",
                "session_id": context.session_id
            }
        
        # Store the location name
        if not context.collected_data.get('location_name'):
            context.collected_data['location_name'] = message
        
        # Move to confirmation
        context.state = ProcessState.CONFIRMATION
        return self._generate_confirmation_message(context)

    def _generate_confirmation_message(self, context: ProcessContext) -> Dict[str, Any]:
        """Step 3: Generate confirmation message"""
        if context.intent == Intent.CREATE:
            location_name = context.collected_data.get('location_name')
            return {
                "message": f"Intent: CREATE\nSite name: {location_name}\n\nConfirm to create this site? (yes/no)",
                "status": "confirmation_needed",
                "session_id": context.session_id,
                "process_state": "confirmation",
                "intent": context.intent.value,
                "data": context.collected_data
            }
        
        elif context.intent == Intent.DELETE:
            location_name = context.collected_data.get('location_name')
            return {
                "message": f"Intent: DELETE\nSite name: {location_name}\n\nConfirm to delete this site? (yes/no)",
                "status": "confirmation_needed",
                "session_id": context.session_id,
                "process_state": "confirmation", 
                "intent": context.intent.value,
                "data": context.collected_data
            }
        
        elif context.intent == Intent.READ:
            return {
                "message": f"Intent: READ\nConfirm to view all sites? (yes/no)",
                "status": "confirmation_needed",
                "session_id": context.session_id,
                "process_state": "confirmation",
                "intent": context.intent.value
            }

    def _handle_confirmation(self, message: str, context: ProcessContext) -> Dict[str, Any]:
        """Step 3: Handle user confirmation"""
        message_lower = message.lower().strip()
        
        if message_lower in ['yes', 'y', 'confirm', 'ok', 'proceed']:
            # Execute the operation
            context.state = ProcessState.EXECUTION
            return self._execute_operation(context)
        
        elif message_lower in ['no', 'n', 'cancel']:
            # Cancel operation
            del self.sessions[context.session_id]
            return {
                "message": "Operation cancelled.",
                "status": "cancelled",
                "session_id": context.session_id
            }
        
        else:
            return {
                "message": "Please confirm with 'yes' or 'no':",
                "status": "confirmation_needed",
                "session_id": context.session_id,
                "process_state": "confirmation"
            }

    def _execute_operation(self, context: ProcessContext) -> Dict[str, Any]:
        """Step 4: Execute the confirmed operation"""
        try:
            if context.intent == Intent.CREATE:
                return self._execute_create(context)
            elif context.intent == Intent.READ:
                return self._execute_read(context)
            elif context.intent == Intent.DELETE:
                return self._execute_delete(context)
            else:
                raise ValueError(f"Unknown intent: {context.intent}")
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return self._create_error_response(f"Execution failed: {str(e)}", context.session_id)
        finally:
            # Clean up session after execution
            if context.session_id in self.sessions:
                del self.sessions[context.session_id]

    def _execute_create(self, context: ProcessContext) -> Dict[str, Any]:
        """Execute CREATE operation"""
        location_name = context.collected_data.get('location_name')
        
        try:
            # Use the site manager to create site
            result = self.site_manager.create_site_by_name_only(location_name)
            
            return {
                "message": f"✅ SUCCESS: Site '{location_name}' created successfully!",
                "status": "completed",
                "session_id": context.session_id,
                "operation": "CREATE",
                "result": result
            }
            
        except Exception as e:
            raise Exception(f"Failed to create site '{location_name}': {str(e)}")

    def _execute_read(self, context: ProcessContext) -> Dict[str, Any]:
        """Execute READ operation"""
        try:
            # Get all sites
            result = self.site_manager.get_all_sites()
            sites = result.get('locations', [])
            
            if sites:
                site_list = []
                for i, site in enumerate(sites, 1):
                    name = site.get('location_name', 'Unknown')
                    site_id = site.get('id', 'N/A')
                    site_list.append(f"{i}. {name} (ID: {site_id})")
                
                sites_display = '\n'.join(site_list)
                message = f"✅ SUCCESS: Found {len(sites)} sites:\n\n{sites_display}"
            else:
                message = "✅ SUCCESS: No sites found."
            
            return {
                "message": message,
                "status": "completed",
                "session_id": context.session_id,
                "operation": "READ",
                "result": result
            }
            
        except Exception as e:
            raise Exception(f"Failed to retrieve sites: {str(e)}")

    def _execute_delete(self, context: ProcessContext) -> Dict[str, Any]:
        """Execute DELETE operation"""
        location_name = context.collected_data.get('location_name')
        
        try:
            # First, find the site by name
            all_sites = self.site_manager.get_all_sites()
            sites = all_sites.get('locations', [])
            
            target_site = None
            for site in sites:
                if site.get('location_name', '').lower() == location_name.lower():
                    target_site = site
                    break
            
            if not target_site:
                return {
                    "message": f"❌ ERROR: Site '{location_name}' not found.",
                    "status": "completed",
                    "session_id": context.session_id,
                    "operation": "DELETE"
                }
            
            # Delete the site
            site_id = target_site.get('id')
            result = self.site_manager.delete_site(site_id)
            
            return {
                "message": f"✅ SUCCESS: Site '{location_name}' deleted successfully!",
                "status": "completed",
                "session_id": context.session_id,
                "operation": "DELETE",
                "result": result
            }
            
        except Exception as e:
            raise Exception(f"Failed to delete site '{location_name}': {str(e)}")

    def _create_error_response(self, error_message: str, session_id: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "message": f"❌ ERROR: {error_message}",
            "status": "error",
            "session_id": session_id
        }

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session"""
        if session_id not in self.sessions:
            return None
        
        context = self.sessions[session_id]
        return {
            "session_id": session_id,
            "state": context.state.value,
            "intent": context.intent.value if context.intent else None,
            "collected_data": context.collected_data,
            "ready_for_execution": context.ready_for_execution
        }

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def clear_all_sessions(self) -> int:
        """Clear all sessions and return count of cleared sessions"""
        count = len(self.sessions)
        self.sessions.clear()
        return count


# Test function
def test_tree_system():
    """Test the tree-based system"""
    try:
        # Initialize components
        auth_manager = AuthenticationManager()
        site_manager = SiteManager(auth_manager)
        tree_system = TreeBasedSystem(site_manager)
        
        print("🌳 Tree-based System initialized successfully!")
        
        # Test intent extraction
        test_messages = [
            "I want to create a new site",
            "Show me all sites",
            "Delete a site",
            "What's the weather like?"
        ]
        
        for msg in test_messages:
            intent = tree_system._extract_intent(msg)
            print(f"Message: '{msg}' -> Intent: {intent}")
        
        return tree_system
        
    except Exception as e:
        print(f"❌ Tree system initialization failed: {e}")
        return None

if __name__ == "__main__":
    test_tree_system()
