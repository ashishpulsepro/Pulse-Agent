"""
Intent Analyzer
Analyzes user input to determine intent and extract parameters
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Supported intent types"""
    # Site management
    CREATE_SITE = "create_site"
    LIST_SITES = "list_sites" 
    GET_SITE = "get_site"
    UPDATE_SITE = "update_site"
    DELETE_SITE = "delete_site"
    
    # User management
    ASSIGN_USER = "assign_user"
    UNASSIGN_USER = "unassign_user"
    LIST_SITE_USERS = "list_site_users"
    
    # General
    GREETING = "greeting"
    HELP = "help"
    UNKNOWN = "unknown"

@dataclass
class IntentResult:
    """Result of intent analysis"""
    intent: IntentType
    confidence: float
    parameters: Dict[str, Any]
    needs_clarification: bool = False
    clarification_message: str = ""
    
class IntentAnalyzer:
    """
    Analyzes user input to determine intent and extract parameters
    """
    
    def __init__(self):
        """Initialize intent patterns and rules"""
        self.intent_patterns = self._initialize_patterns()
        self.parameter_extractors = self._initialize_extractors()
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize regex patterns for each intent"""
        return {
            # Site management patterns
            IntentType.CREATE_SITE: [
                r"create\s+(?:a\s+)?(?:new\s+)?site(?:\s+(?:called|named))?\s+['\"]?([^'\"]+)['\"]?",
                r"add\s+(?:a\s+)?(?:new\s+)?site(?:\s+(?:called|named))?\s+['\"]?([^'\"]+)['\"]?",
                r"make\s+(?:a\s+)?(?:new\s+)?site(?:\s+(?:called|named))?\s+['\"]?([^'\"]+)['\"]?",
                r"set\s+up\s+(?:a\s+)?(?:new\s+)?site(?:\s+(?:called|named))?\s+['\"]?([^'\"]+)['\"]?",
                r"(?:i\s+want\s+to\s+|i\s+need\s+to\s+|can\s+you\s+)?create.*site.*['\"]?([^'\"]+)['\"]?",
            ],
            
            IntentType.LIST_SITES: [
                r"(?:show|list|get|display|view)\s+(?:all\s+)?(?:my\s+)?sites?",
                r"what\s+sites?\s+(?:do\s+)?(?:i\s+)?have",
                r"show\s+me\s+(?:all\s+)?(?:the\s+)?sites?",
                r"list\s+(?:all\s+)?(?:my\s+)?sites?",
                r"get\s+(?:all\s+)?(?:my\s+)?sites?",
            ],
            
            IntentType.GET_SITE: [
                r"(?:show|get|find|display)\s+(?:me\s+)?(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
                r"details?\s+(?:of\s+|for\s+)?(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
                r"info(?:rmation)?\s+(?:about\s+|on\s+)?(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
            ],
            
            IntentType.UPDATE_SITE: [
                r"(?:update|edit|modify|change)\s+(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
                r"edit\s+(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
                r"modify\s+(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
            ],
            
            IntentType.DELETE_SITE: [
                r"(?:delete|remove|drop)\s+(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
                r"remove\s+(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
                r"get\s+rid\s+of\s+(?:the\s+)?site\s+['\"]?([^'\"]+)['\"]?",
            ],
            
            # User management patterns  
            IntentType.ASSIGN_USER: [
                r"assign\s+(?:user\s+)?['\"]?([^'\"]+)['\"]?\s+to\s+(?:site\s+)?['\"]?([^'\"]+)['\"]?",
                r"add\s+(?:user\s+)?['\"]?([^'\"]+)['\"]?\s+to\s+(?:site\s+)?['\"]?([^'\"]+)['\"]?",
                r"give\s+(?:user\s+)?['\"]?([^'\"]+)['\"]?\s+access\s+to\s+(?:site\s+)?['\"]?([^'\"]+)['\"]?",
            ],
            
            IntentType.UNASSIGN_USER: [
                r"(?:unassign|remove)\s+(?:user\s+)?['\"]?([^'\"]+)['\"]?\s+from\s+(?:site\s+)?['\"]?([^'\"]+)['\"]?",
                r"remove\s+(?:user\s+)?['\"]?([^'\"]+)['\"]?\s+from\s+(?:site\s+)?['\"]?([^'\"]+)['\"]?",
            ],
            
            IntentType.LIST_SITE_USERS: [
                r"(?:show|list|get)\s+users?\s+(?:for\s+|in\s+|at\s+)?(?:site\s+)?['\"]?([^'\"]+)['\"]?",
                r"who\s+(?:has\s+access\s+to\s+|is\s+assigned\s+to\s+)?(?:site\s+)?['\"]?([^'\"]+)['\"]?",
                r"users?\s+(?:for\s+|in\s+|at\s+)?(?:site\s+)?['\"]?([^'\"]+)['\"]?",
            ],
            
            # General patterns
            IntentType.GREETING: [
                r"^(?:hi|hello|hey|greetings?)(?:\s+there)?(?:\s*[,!.])?$",
                r"good\s+(?:morning|afternoon|evening)",
                r"how\s+are\s+you",
                r"what'?s\s+up",
            ],
            
            IntentType.HELP: [
                r"help",
                r"what\s+can\s+you\s+do",
                r"how\s+(?:do\s+i|can\s+i)",
                r"commands?",
                r"options?",
            ],
        }
    
    def _initialize_extractors(self) -> Dict[str, callable]:
        """Initialize parameter extraction functions"""
        return {
            'site_name': self._extract_site_name,
            'user_name': self._extract_user_name,
            'location': self._extract_location,
            'address': self._extract_address,
        }
    
    async def analyze(self, user_input: str, session_id: str = None) -> IntentResult:
        """
        Analyze user input and return intent with parameters
        """
        user_input = user_input.strip().lower()
        
        # Try to match against patterns
        best_match = self._find_best_intent_match(user_input)
        
        if best_match['intent'] == IntentType.UNKNOWN:
            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                parameters={},
                needs_clarification=True,
                clarification_message="I'm not sure what you want me to do. Could you please rephrase? I can help you with site management, user assignments, and more."
            )
        
        # Extract parameters based on intent
        parameters = self._extract_parameters(best_match['intent'], user_input, best_match['matches'])
        
        # Check if we need clarification for required parameters
        clarification = self._check_for_clarification(best_match['intent'], parameters)
        
        return IntentResult(
            intent=best_match['intent'],
            confidence=best_match['confidence'],
            parameters=parameters,
            needs_clarification=clarification['needed'],
            clarification_message=clarification['message']
        )
    
    def _find_best_intent_match(self, user_input: str) -> Dict[str, Any]:
        """Find the best matching intent pattern"""
        best_match = {
            'intent': IntentType.UNKNOWN,
            'confidence': 0.0,
            'matches': []
        }
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    # Calculate confidence based on pattern coverage
                    confidence = len(match.group(0)) / len(user_input)
                    
                    if confidence > best_match['confidence']:
                        best_match = {
                            'intent': intent,
                            'confidence': confidence,
                            'matches': match.groups()
                        }
        
        return best_match
    
    def _extract_parameters(self, intent: IntentType, user_input: str, matches: List[str]) -> Dict[str, Any]:
        """Extract parameters based on intent and regex matches"""
        parameters = {}
        
        if intent in [IntentType.CREATE_SITE, IntentType.GET_SITE, IntentType.UPDATE_SITE, IntentType.DELETE_SITE]:
            if matches and matches[0]:
                parameters['site_name'] = matches[0].strip()
        
        elif intent == IntentType.ASSIGN_USER:
            if len(matches) >= 2:
                parameters['user_name'] = matches[0].strip()
                parameters['site_name'] = matches[1].strip()
        
        elif intent == IntentType.UNASSIGN_USER:
            if len(matches) >= 2:
                parameters['user_name'] = matches[0].strip()
                parameters['site_name'] = matches[1].strip()
        
        elif intent == IntentType.LIST_SITE_USERS:
            if matches and matches[0]:
                parameters['site_name'] = matches[0].strip()
        
        # Extract additional context from the full input
        additional_params = self._extract_additional_context(user_input)
        parameters.update(additional_params)
        
        return parameters
    
    def _extract_additional_context(self, user_input: str) -> Dict[str, Any]:
        """Extract additional context like location, address, etc."""
        context = {}
        
        # Extract location mentions
        location_patterns = [
            r"in\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!|\?)",
            r"at\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!|\?)",
            r"located\s+in\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|!|\?)",
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if len(location) > 2:  # Filter out small words
                    context['location'] = location
                    break
        
        # Extract address mentions
        address_patterns = [
            r"(?:address|addr)[:=]\s*([^,\n]+)",
            r"(?:at\s+address|located\s+at)\s+([^,\n]+)",
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                context['address'] = match.group(1).strip()
                break
        
        return context
    
    def _check_for_clarification(self, intent: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check if clarification is needed for required parameters"""
        
        required_params = {
            IntentType.CREATE_SITE: ['site_name'],
            IntentType.GET_SITE: ['site_name'],
            IntentType.UPDATE_SITE: ['site_name'],
            IntentType.DELETE_SITE: ['site_name'],
            IntentType.ASSIGN_USER: ['user_name', 'site_name'],
            IntentType.UNASSIGN_USER: ['user_name', 'site_name'],
            IntentType.LIST_SITE_USERS: ['site_name'],
        }
        
        if intent not in required_params:
            return {'needed': False, 'message': ''}
        
        missing_params = []
        for param in required_params[intent]:
            if param not in parameters or not parameters[param]:
                missing_params.append(param)
        
        if missing_params:
            clarification_messages = {
                'site_name': "What's the name of the site?",
                'user_name': "Which user are you referring to?",
            }
            
            if len(missing_params) == 1:
                message = clarification_messages.get(missing_params[0], f"I need the {missing_params[0].replace('_', ' ')}.")
            else:
                message = f"I need more information: {', '.join([p.replace('_', ' ') for p in missing_params])}"
            
            return {'needed': True, 'message': message}
        
        return {'needed': False, 'message': ''}
    
    def _extract_site_name(self, text: str) -> Optional[str]:
        """Extract site name from text"""
        # Implementation for site name extraction
        pass
    
    def _extract_user_name(self, text: str) -> Optional[str]:
        """Extract user name from text"""
        # Implementation for user name extraction  
        pass
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from text"""
        # Implementation for location extraction
        pass
    
    def _extract_address(self, text: str) -> Optional[str]:
        """Extract address from text"""
        # Implementation for address extraction
        pass
    
    def get_supported_intents(self) -> List[str]:
        """Return list of supported intents"""
        return [intent.value for intent in IntentType if intent != IntentType.UNKNOWN]
