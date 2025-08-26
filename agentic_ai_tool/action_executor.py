"""
Action Executor
Executes actions based on analyzed intents using registered tools
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from intent_analyzer import IntentType, IntentResult

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of action execution"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    actions_taken: Optional[List[str]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.actions_taken is None:
            self.actions_taken = []

class ActionExecutor:
    """
    Executes actions based on analyzed intents using registered tools
    """
    
    def __init__(self, auth_manager=None):
        """Initialize action executor"""
        self.auth_manager = auth_manager
        self.tools = {}
        
    def register_tool(self, tool_name: str, tool_instance):
        """Register a tool for execution"""
        self.tools[tool_name] = tool_instance
        logger.info(f"Registered tool: {tool_name}")
    
    async def execute(self, intent_result: IntentResult) -> ExecutionResult:
        """
        Execute action based on intent result
        """
        try:
            intent = intent_result.intent
            parameters = intent_result.parameters
            
            # Route to appropriate tool based on intent
            if intent in [IntentType.CREATE_SITE, IntentType.LIST_SITES, IntentType.GET_SITE, 
                         IntentType.UPDATE_SITE, IntentType.DELETE_SITE]:
                return await self._execute_site_action(intent, parameters)
            
            elif intent in [IntentType.ASSIGN_USER, IntentType.UNASSIGN_USER, IntentType.LIST_SITE_USERS]:
                return await self._execute_user_action(intent, parameters)
            
            elif intent == IntentType.GREETING:
                return ExecutionResult(
                    success=True,
                    message="Hello! I'm your PulsePro AI assistant. I can help you manage sites and users. What would you like to do?",
                    actions_taken=["greeting"]
                )
            
            elif intent == IntentType.HELP:
                return await self._execute_help_action()
            
            else:
                return ExecutionResult(
                    success=False,
                    message="I'm not sure how to handle that request. Could you please try rephrasing?",
                    error="Unknown intent"
                )
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return ExecutionResult(
                success=False,
                message=f"An error occurred while executing the action: {str(e)}",
                error=str(e)
            )
    
    async def _execute_site_action(self, intent: IntentType, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute site-related actions"""
        if 'site' not in self.tools:
            return ExecutionResult(
                success=False,
                message="Site management tool is not available",
                error="Site tool not registered"
            )
        
        site_tool = self.tools['site']
        
        try:
            if intent == IntentType.CREATE_SITE:
                result = await site_tool.create_site(parameters)
                
            elif intent == IntentType.LIST_SITES:
                result = await site_tool.list_sites(parameters)
                
            elif intent == IntentType.GET_SITE:
                result = await site_tool.get_site(parameters)
                
            elif intent == IntentType.UPDATE_SITE:
                result = await site_tool.update_site(parameters)
                
            elif intent == IntentType.DELETE_SITE:
                result = await site_tool.delete_site(parameters)
                
            else:
                return ExecutionResult(
                    success=False,
                    message="Unsupported site action",
                    error=f"Intent {intent.value} not supported"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in site action execution: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to execute site action: {str(e)}",
                error=str(e)
            )
    
    async def _execute_user_action(self, intent: IntentType, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute user-related actions"""
        if 'user' not in self.tools:
            return ExecutionResult(
                success=False,
                message="User management tool is not available",
                error="User tool not registered"
            )
        
        user_tool = self.tools['user']
        
        try:
            if intent == IntentType.ASSIGN_USER:
                result = await user_tool.assign_user(parameters)
                
            elif intent == IntentType.UNASSIGN_USER:
                result = await user_tool.unassign_user(parameters)
                
            elif intent == IntentType.LIST_SITE_USERS:
                result = await user_tool.list_site_users(parameters)
                
            else:
                return ExecutionResult(
                    success=False,
                    message="Unsupported user action",
                    error=f"Intent {intent.value} not supported"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in user action execution: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to execute user action: {str(e)}",
                error=str(e)
            )
    
    async def _execute_help_action(self) -> ExecutionResult:
        """Execute help action"""
        help_message = """
I can help you with the following:

**Site Management:**
• Create a new site: "Create a site called Office Mumbai"
• List all sites: "Show me all sites"
• Get site details: "Show me details of Mumbai office"
• Update a site: "Update the Delhi office site"
• Delete a site: "Delete the Bangalore office"

**User Management:**
• Assign user to site: "Assign John to Mumbai office"
• Remove user from site: "Remove Jane from Delhi office" 
• List site users: "Show users for Mumbai office"

Just tell me what you'd like to do in natural language!
        """.strip()
        
        return ExecutionResult(
            success=True,
            message=help_message,
            actions_taken=["help"],
            data={
                'capabilities': [
                    'Site Management (CRUD operations)',
                    'User Assignment Management',
                    'Natural Language Processing'
                ]
            }
        )
