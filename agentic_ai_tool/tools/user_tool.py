"""
User Tool
Handles all user-related operations including site assignments
"""

import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add backend directory to path to import site_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from site_manager import SiteManager, PulseProAPIException
except ImportError:
    # Fallback for development
    class SiteManager:
        def __init__(self, auth_manager): pass
        def get_all_sites(self): return {'locations': []}
        def get_site_users(self, site_id): return {'users': []}
        def assign_users_to_site(self, site_id, user_ids): return {'success': True}
        def unassign_users_from_site(self, mapped_ids): return {'success': True}
    PulseProAPIException = Exception

# Import relative modules
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from action_executor import ExecutionResult

logger = logging.getLogger(__name__)

class UserTool:
    """
    Tool for handling user-related operations
    """
    
    def __init__(self, auth_manager=None):
        """Initialize user tool with authentication manager"""
        self.auth_manager = auth_manager
        if auth_manager and SiteManager:
            self.site_manager = SiteManager(auth_manager)
        else:
            self.site_manager = None
            logger.warning("UserTool initialized without auth_manager or SiteManager")
    
    async def assign_user(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Assign a user to a site"""
        try:
            if not self.site_manager:
                return ExecutionResult(
                    success=False,
                    message="User management is not configured properly",
                    error="No authentication manager available"
                )
            
            user_name = parameters.get('user_name')
            site_name = parameters.get('site_name')
            
            if not user_name or not site_name:
                return ExecutionResult(
                    success=False,
                    message="Both user name and site name are required",
                    error="Missing required parameters"
                )
            
            # First find the site ID
            site_id = await self._find_site_id(site_name)
            if not site_id:
                return ExecutionResult(
                    success=False,
                    message=f"Site '{site_name}' not found",
                    error="Site not found"
                )
            
            # Get all users for this site to find the user ID
            users_data = self.site_manager.get_site_users(site_id)
            all_users = users_data.get('users', [])
            
            # Find the user
            user_id = None
            for user in all_users:
                if (user.get('name', '').lower() == user_name.lower() or 
                    user.get('username', '').lower() == user_name.lower() or
                    user.get('email', '').lower() == user_name.lower()):
                    user_id = user.get('id')
                    break
            
            if not user_id:
                return ExecutionResult(
                    success=False,
                    message=f"User '{user_name}' not found",
                    error="User not found"
                )
            
            # Check if user is already assigned
            assigned_users = [u for u in all_users if u.get('is_assigned', False)]
            if any(u.get('id') == user_id for u in assigned_users):
                return ExecutionResult(
                    success=True,
                    message=f"User '{user_name}' is already assigned to site '{site_name}'",
                    actions_taken=[f"user_already_assigned:{user_name}:{site_name}"]
                )
            
            # Assign the user
            result = self.site_manager.assign_users_to_site(site_id, [user_id])
            
            return ExecutionResult(
                success=True,
                message=f"Successfully assigned '{user_name}' to site '{site_name}'",
                data=result,
                actions_taken=[f"assigned_user:{user_name}:{site_name}"]
            )
            
        except PulseProAPIException as e:
            logger.error(f"API error assigning user: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to assign user '{user_name}' to site '{site_name}': {str(e)}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error assigning user: {e}")
            return ExecutionResult(
                success=False,
                message="An unexpected error occurred while assigning the user",
                error=str(e)
            )
    
    async def unassign_user(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Unassign a user from a site"""
        try:
            if not self.site_manager:
                return ExecutionResult(
                    success=False,
                    message="User management is not configured properly",
                    error="No authentication manager available"
                )
            
            user_name = parameters.get('user_name')
            site_name = parameters.get('site_name')
            
            if not user_name or not site_name:
                return ExecutionResult(
                    success=False,
                    message="Both user name and site name are required",
                    error="Missing required parameters"
                )
            
            # First find the site ID
            site_id = await self._find_site_id(site_name)
            if not site_id:
                return ExecutionResult(
                    success=False,
                    message=f"Site '{site_name}' not found",
                    error="Site not found"
                )
            
            # Get all users for this site
            users_data = self.site_manager.get_site_users(site_id)
            all_users = users_data.get('users', [])
            
            # Find the assigned user and their mapping ID
            mapped_location_id = None
            for user in all_users:
                if (user.get('name', '').lower() == user_name.lower() or 
                    user.get('username', '').lower() == user_name.lower() or
                    user.get('email', '').lower() == user_name.lower()):
                    if user.get('is_assigned', False):
                        mapped_location_id = user.get('mapped_location_id')
                        break
            
            if not mapped_location_id:
                return ExecutionResult(
                    success=False,
                    message=f"User '{user_name}' is not assigned to site '{site_name}'",
                    error="User not assigned to site"
                )
            
            # Unassign the user
            result = self.site_manager.unassign_users_from_site([mapped_location_id])
            
            return ExecutionResult(
                success=True,
                message=f"Successfully removed '{user_name}' from site '{site_name}'",
                data=result,
                actions_taken=[f"unassigned_user:{user_name}:{site_name}"]
            )
            
        except PulseProAPIException as e:
            logger.error(f"API error unassigning user: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to remove user '{user_name}' from site '{site_name}': {str(e)}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error unassigning user: {e}")
            return ExecutionResult(
                success=False,
                message="An unexpected error occurred while removing the user",
                error=str(e)
            )
    
    async def list_site_users(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """List all users for a specific site"""
        try:
            if not self.site_manager:
                return ExecutionResult(
                    success=False,
                    message="User management is not configured properly",
                    error="No authentication manager available"
                )
            
            site_name = parameters.get('site_name')
            if not site_name:
                return ExecutionResult(
                    success=False,
                    message="Site name is required",
                    error="Missing site_name parameter"
                )
            
            # First find the site ID
            site_id = await self._find_site_id(site_name)
            if not site_id:
                return ExecutionResult(
                    success=False,
                    message=f"Site '{site_name}' not found",
                    error="Site not found"
                )
            
            # Get users for the site
            result = self.site_manager.get_site_users(site_id)
            all_users = result.get('users', [])
            
            # Filter to get only assigned users
            assigned_users = [user for user in all_users if user.get('is_assigned', False)]
            
            return ExecutionResult(
                success=True,
                message=f"Retrieved {len(assigned_users)} users for site '{site_name}'",
                data={'users': assigned_users, 'site_name': site_name, 'site_id': site_id},
                actions_taken=[f"listed_site_users:{site_name}"]
            )
            
        except PulseProAPIException as e:
            logger.error(f"API error listing site users: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to retrieve users for site '{site_name}': {str(e)}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error listing site users: {e}")
            return ExecutionResult(
                success=False,
                message="An unexpected error occurred while retrieving site users",
                error=str(e)
            )
    
    async def _find_site_id(self, site_name: str) -> Optional[int]:
        """Find site ID by name"""
        try:
            all_sites = self.site_manager.get_all_sites()
            sites = all_sites.get('locations', [])
            
            for site in sites:
                if site.get('location_name', '').lower() == site_name.lower():
                    return site.get('id')
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding site ID: {e}")
            return None
    
    def get_available_actions(self) -> List[str]:
        """Return list of available actions"""
        return [
            'assign_user',
            'unassign_user', 
            'list_site_users'
        ]
