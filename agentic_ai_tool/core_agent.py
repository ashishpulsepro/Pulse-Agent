"""
Core AI Agent
Main agent orchestrator that handles user requests and coordinates actions
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from intent_analyzer import IntentAnalyzer
from action_executor import ActionExecutor
from response_generator import ResponseGenerator
from session_manager import SessionManager
from tools.site_tool import SiteTool
from tools.user_tool import UserTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent response status"""
    UNDERSTANDING = "understanding"
    READY_FOR_EXECUTION = "ready_for_execution" 
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"
    CLARIFICATION_NEEDED = "clarification_needed"

@dataclass
class AgentResponse:
    """Structured response from the agent"""
    message: str
    status: AgentStatus
    data: Optional[Dict[str, Any]] = None
    actions_taken: Optional[List[str]] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'message': self.message,
            'status': self.status.value,
            'data': self.data,
            'actions_taken': self.actions_taken,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat()
        }

class CoreAIAgent:
    """
    Main AI Agent that orchestrates all operations
    """
    
    def __init__(self, auth_manager=None):
        """Initialize the agent with required components"""
        self.intent_analyzer = IntentAnalyzer()
        self.action_executor = ActionExecutor(auth_manager)
        self.response_generator = ResponseGenerator()
        self.session_manager = SessionManager()
        
        # Initialize tools
        self.site_tool = SiteTool(auth_manager)
        self.user_tool = UserTool(auth_manager)
        
        # Register tools with action executor
        self.action_executor.register_tool('site', self.site_tool)
        self.action_executor.register_tool('user', self.user_tool)
        
        logger.info("CoreAIAgent initialized successfully")
    
    async def process_request(self, user_input: str, session_id: str = None) -> AgentResponse:
        """
        Main entry point for processing user requests
        """
        try:
            # Manage session
            if not session_id:
                session_id = self.session_manager.create_session()
            
            # Add user message to session
            self.session_manager.add_message(session_id, 'user', user_input)
            
            # Analyze user intent
            intent_result = await self.intent_analyzer.analyze(user_input, session_id)
            
            if intent_result.needs_clarification:
                response = AgentResponse(
                    message=intent_result.clarification_message,
                    status=AgentStatus.CLARIFICATION_NEEDED,
                    session_id=session_id
                )
            else:
                # Execute the action
                execution_result = await self.action_executor.execute(intent_result)
                
                # Generate response
                response_message = self.response_generator.generate_response(
                    intent_result, execution_result
                )
                
                # Determine status
                status = AgentStatus.COMPLETED if execution_result.success else AgentStatus.ERROR
                
                response = AgentResponse(
                    message=response_message,
                    status=status,
                    data=execution_result.data,
                    actions_taken=execution_result.actions_taken,
                    session_id=session_id
                )
            
            # Add agent response to session
            self.session_manager.add_message(
                session_id, 'agent', response.message, response.status.value
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            error_response = AgentResponse(
                message=f"I encountered an error while processing your request: {str(e)}",
                status=AgentStatus.ERROR,
                session_id=session_id
            )
            
            if session_id:
                self.session_manager.add_message(
                    session_id, 'agent', error_response.message, error_response.status.value
                )
            
            return error_response
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return self.session_manager.get_session_messages(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        return self.session_manager.clear_session(session_id)
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and available actions"""
        return {
            'capabilities': [
                'Site Management (create, read, update, delete sites)',
                'User Management (assign/unassign users to sites)',
                'Natural language understanding',
                'Context-aware conversations'
            ],
            'available_actions': {
                'site': self.site_tool.get_available_actions(),
                'user': self.user_tool.get_available_actions()
            },
            'supported_intents': self.intent_analyzer.get_supported_intents()
        }
