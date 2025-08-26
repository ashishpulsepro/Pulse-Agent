"""
Response Generator
Generates natural language responses based on execution results
"""

import logging
from typing import Dict, Any, List

from intent_analyzer import IntentType, IntentResult
from action_executor import ExecutionResult

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates natural language responses based on execution results
    """
    
    def __init__(self):
        """Initialize response generator"""
        self.response_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize response templates for different scenarios"""
        return {
            IntentType.CREATE_SITE.value: {
                'success': "Great! I've successfully created the site '{site_name}'{location_info}. The site ID is {site_id}.",
                'error': "I couldn't create the site '{site_name}'. {error_details}",
                'partial': "I've created the site '{site_name}' but some details need to be completed later."
            },
            
            IntentType.LIST_SITES.value: {
                'success': "Here are your sites:\n\n{sites_list}",
                'success_empty': "You don't have any sites yet. Would you like me to create one?",
                'error': "I couldn't retrieve your sites. {error_details}"
            },
            
            IntentType.GET_SITE.value: {
                'success': "Here are the details for '{site_name}':\n\n{site_details}",
                'not_found': "I couldn't find a site named '{site_name}'. Would you like me to show you all available sites?",
                'error': "I couldn't retrieve the site details. {error_details}"
            },
            
            IntentType.UPDATE_SITE.value: {
                'success': "I've successfully updated the site '{site_name}'. {update_details}",
                'not_found': "I couldn't find a site named '{site_name}' to update.",
                'error': "I couldn't update the site '{site_name}'. {error_details}"
            },
            
            IntentType.DELETE_SITE.value: {
                'success': "I've successfully deleted the site '{site_name}'.",
                'not_found': "I couldn't find a site named '{site_name}' to delete.",
                'error': "I couldn't delete the site '{site_name}'. {error_details}",
                'confirmation': "Are you sure you want to delete the site '{site_name}'? This action cannot be undone."
            },
            
            IntentType.ASSIGN_USER.value: {
                'success': "I've successfully assigned {user_name} to the site '{site_name}'.",
                'already_assigned': "{user_name} is already assigned to the site '{site_name}'.",
                'user_not_found': "I couldn't find a user named '{user_name}'.",
                'site_not_found': "I couldn't find a site named '{site_name}'.",
                'error': "I couldn't assign {user_name} to the site '{site_name}'. {error_details}"
            },
            
            IntentType.UNASSIGN_USER.value: {
                'success': "I've successfully removed {user_name} from the site '{site_name}'.",
                'not_assigned': "{user_name} is not assigned to the site '{site_name}'.",
                'user_not_found': "I couldn't find a user named '{user_name}'.",
                'site_not_found': "I couldn't find a site named '{site_name}'.",
                'error': "I couldn't remove {user_name} from the site '{site_name}'. {error_details}"
            },
            
            IntentType.LIST_SITE_USERS.value: {
                'success': "Here are the users assigned to '{site_name}':\n\n{users_list}",
                'success_empty': "No users are currently assigned to the site '{site_name}'.",
                'site_not_found': "I couldn't find a site named '{site_name}'.",
                'error': "I couldn't retrieve the users for site '{site_name}'. {error_details}"
            }
        }
    
    def generate_response(self, intent_result: IntentResult, execution_result: ExecutionResult) -> str:
        """
        Generate a natural language response based on intent and execution results
        """
        try:
            # If execution result already has a message, use it for non-data operations
            if execution_result.message and intent_result.intent in [IntentType.GREETING, IntentType.HELP]:
                return execution_result.message
            
            # Get the appropriate template
            template_key = intent_result.intent.value
            templates = self.response_templates.get(template_key, {})
            
            if execution_result.success:
                return self._generate_success_response(
                    intent_result, execution_result, templates
                )
            else:
                return self._generate_error_response(
                    intent_result, execution_result, templates
                )
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I completed the action but had trouble generating a proper response. {execution_result.message or str(e)}"
    
    def _generate_success_response(self, intent_result: IntentResult, 
                                 execution_result: ExecutionResult, 
                                 templates: Dict[str, str]) -> str:
        """Generate success response"""
        
        intent = intent_result.intent
        parameters = intent_result.parameters
        data = execution_result.data or {}
        
        # Handle different success scenarios
        if intent == IntentType.CREATE_SITE:
            template = templates.get('success', 'Site created successfully.')
            location_info = f" in {parameters.get('location', '')}" if parameters.get('location') else ""
            return template.format(
                site_name=parameters.get('site_name', 'Unknown'),
                location_info=location_info,
                site_id=data.get('id', 'N/A')
            )
        
        elif intent == IntentType.LIST_SITES:
            sites = data.get('locations', [])
            if not sites:
                return templates.get('success_empty', 'No sites found.')
            
            sites_list = self._format_sites_list(sites)
            return templates.get('success', 'Here are your sites:').format(sites_list=sites_list)
        
        elif intent == IntentType.GET_SITE:
            site_details = self._format_site_details(data)
            return templates.get('success', 'Site details retrieved.').format(
                site_name=parameters.get('site_name', 'Unknown'),
                site_details=site_details
            )
        
        elif intent == IntentType.UPDATE_SITE:
            update_details = "The changes have been saved."
            return templates.get('success', 'Site updated successfully.').format(
                site_name=parameters.get('site_name', 'Unknown'),
                update_details=update_details
            )
        
        elif intent == IntentType.DELETE_SITE:
            return templates.get('success', 'Site deleted successfully.').format(
                site_name=parameters.get('site_name', 'Unknown')
            )
        
        elif intent == IntentType.ASSIGN_USER:
            return templates.get('success', 'User assigned successfully.').format(
                user_name=parameters.get('user_name', 'Unknown'),
                site_name=parameters.get('site_name', 'Unknown')
            )
        
        elif intent == IntentType.UNASSIGN_USER:
            return templates.get('success', 'User unassigned successfully.').format(
                user_name=parameters.get('user_name', 'Unknown'),
                site_name=parameters.get('site_name', 'Unknown')
            )
        
        elif intent == IntentType.LIST_SITE_USERS:
            users = data.get('users', [])
            if not users:
                return templates.get('success_empty', 'No users found.').format(
                    site_name=parameters.get('site_name', 'Unknown')
                )
            
            users_list = self._format_users_list(users)
            return templates.get('success', 'Users retrieved successfully.').format(
                site_name=parameters.get('site_name', 'Unknown'),
                users_list=users_list
            )
        
        # Fallback
        return execution_result.message or "Operation completed successfully."
    
    def _generate_error_response(self, intent_result: IntentResult, 
                               execution_result: ExecutionResult, 
                               templates: Dict[str, str]) -> str:
        """Generate error response"""
        
        intent = intent_result.intent
        parameters = intent_result.parameters
        error_details = execution_result.error or "An unexpected error occurred."
        
        # Handle specific error cases
        if "not found" in error_details.lower():
            if intent in [IntentType.GET_SITE, IntentType.UPDATE_SITE, IntentType.DELETE_SITE]:
                return templates.get('not_found', 'Site not found.').format(
                    site_name=parameters.get('site_name', 'Unknown')
                )
            elif intent == IntentType.LIST_SITE_USERS:
                return templates.get('site_not_found', 'Site not found.').format(
                    site_name=parameters.get('site_name', 'Unknown')
                )
        
        # Use error template
        template = templates.get('error', 'An error occurred: {error_details}')
        
        if intent in [IntentType.CREATE_SITE, IntentType.UPDATE_SITE, IntentType.DELETE_SITE, IntentType.GET_SITE]:
            return template.format(
                site_name=parameters.get('site_name', 'Unknown'),
                error_details=error_details
            )
        elif intent in [IntentType.ASSIGN_USER, IntentType.UNASSIGN_USER]:
            return template.format(
                user_name=parameters.get('user_name', 'Unknown'),
                site_name=parameters.get('site_name', 'Unknown'),
                error_details=error_details
            )
        elif intent == IntentType.LIST_SITE_USERS:
            return template.format(
                site_name=parameters.get('site_name', 'Unknown'),
                error_details=error_details
            )
        
        return execution_result.message or f"An error occurred: {error_details}"
    
    def _format_sites_list(self, sites: List[Dict[str, Any]]) -> str:
        """Format sites list for display"""
        if not sites:
            return "No sites available."
        
        formatted_sites = []
        for i, site in enumerate(sites, 1):
            site_info = f"{i}. **{site.get('location_name', 'Unknown')}**"
            
            details = []
            if site.get('address_field1'):
                details.append(f"Address: {site['address_field1']}")
            if site.get('id'):
                details.append(f"ID: {site['id']}")
            
            if details:
                site_info += f"\n   {' | '.join(details)}"
            
            formatted_sites.append(site_info)
        
        return "\n\n".join(formatted_sites)
    
    def _format_site_details(self, site_data: Dict[str, Any]) -> str:
        """Format site details for display"""
        if not site_data:
            return "No details available."
        
        details = []
        
        # Basic information
        if site_data.get('location_name'):
            details.append(f"**Name:** {site_data['location_name']}")
        if site_data.get('id'):
            details.append(f"**ID:** {site_data['id']}")
        
        # Address information
        if site_data.get('address_field1'):
            details.append(f"**Address:** {site_data['address_field1']}")
        if site_data.get('address_field2'):
            details.append(f"**Address 2:** {site_data['address_field2']}")
        
        # Contact information
        if site_data.get('mobile'):
            details.append(f"**Mobile:** {site_data['mobile']}")
        if site_data.get('to_email'):
            details.append(f"**Email:** {site_data['to_email']}")
        
        # Location information
        if site_data.get('pincode'):
            details.append(f"**Pincode:** {site_data['pincode']}")
        if site_data.get('reporting_timezone'):
            details.append(f"**Timezone:** {site_data['reporting_timezone']}")
        
        return "\n".join(details) if details else "No details available."
    
    def _format_users_list(self, users: List[Dict[str, Any]]) -> str:
        """Format users list for display"""
        if not users:
            return "No users found."
        
        formatted_users = []
        for i, user in enumerate(users, 1):
            user_info = f"{i}. **{user.get('name', user.get('username', 'Unknown'))}**"
            
            details = []
            if user.get('email'):
                details.append(f"Email: {user['email']}")
            if user.get('role'):
                details.append(f"Role: {user['role']}")
            
            if details:
                user_info += f"\n   {' | '.join(details)}"
            
            formatted_users.append(user_info)
        
        return "\n\n".join(formatted_users)
