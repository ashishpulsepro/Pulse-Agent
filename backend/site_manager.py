"""
PulsePro AI Site Management Backend System
A comprehensive backend for managing sites with CRUD operations and user assignments
"""

import requests
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SiteData:
    """Data structure for site information"""
    location_name: str
    address_field1: str
    address_field2: str = ""
    country_id: int = 1
    state_id: int = 283
    city_id: int = 34384
    pincode: str = ""
    mobile: str = ""
    location_number: str = ""
    location_code: str = ""
    to_email: str = ""
    cc_email: str = ""
    reporting_timezone: str = "UTC"
    geo_fencing_enabled: bool = False
    geo_fencing_distance: int = 0
    lat: float = 0.0
    lng: float = 0.0
    map_link: str = ""
    city_list: List = None
    state_list: List = None
    has_custom_field: bool = False
    is_schedule_active: bool = False

    def __post_init__(self):
        if self.city_list is None:
            self.city_list = []
        if self.state_list is None:
            self.state_list = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for API calls"""
        return asdict(self)

@dataclass
class AuthTokens:
    """Data structure for authentication tokens"""
    access_token: str
    refresh_token: str
    expires_at: Optional[datetime] = None

class PulseProAPIException(Exception):
    """Custom exception for API errors"""
    pass

class AuthenticationManager:
    """Handles authentication and token management"""
    
    def __init__(self, base_url: str = "https://staging-api.pulsepro.ai"):
        self.base_url = base_url
        self.tokens: Optional[AuthTokens] = None
    
    def set_refresh_token(self, refresh_token: str) -> None:
        """Set the refresh token for authentication"""
        self.tokens = AuthTokens(access_token="", refresh_token=refresh_token)
    
    def refresh_access_token(self) -> str:
        """Refresh the access token using refresh token"""
        if not self.tokens or not self.tokens.refresh_token:
            raise PulseProAPIException("No refresh token available")
        
        url = f"{self.base_url}/api/refresh/"
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json',
            'Origin': 'https://staging.pulsepro.ai',
            'Referer': 'https://staging.pulsepro.ai/',
        }
        
        payload = {"refresh": self.tokens.refresh_token}
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            access_token = data.get('access')
            
            if not access_token:
                raise PulseProAPIException("No access token in response")
            
            self.tokens.access_token = access_token
            logger.info("Access token refreshed successfully")
            
            return access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to refresh access token: {e}")
            raise PulseProAPIException(f"Token refresh failed: {e}")
    
    def get_access_token(self) -> str:
        """Get current access token, refresh if needed"""
        if not self.tokens or not self.tokens.access_token:
            return self.refresh_access_token()
        return self.tokens.access_token

class SiteManager:
    """Main class for managing sites and user assignments"""
    
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
    
    def create_site(self, site_data: SiteData) -> Dict[str, Any]:
        """
        Create a new site
        
        Args:
            site_data: SiteData object with site information
            
        Returns:
            Dict containing the API response
        """
        url = f"{self.base_url}/customer/add_location/"
        headers = self._get_headers()
        
        # Validate required fields
        required_fields = ['location_name', 'address_field1', 'country_id', 'state_id', 'city_id', 'reporting_timezone']
        for field in required_fields:
            if not getattr(site_data, field):
                raise ValueError(f"Required field '{field}' is missing or empty")
        
        try:
            response = requests.post(url, headers=headers, json=site_data.to_dict())
            response.raise_for_status()
            
            logger.info(f"Site '{site_data.location_name}' created successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create site: {e}")
            raise PulseProAPIException(f"Site creation failed: {e}")
    
    def update_site(self, location_id: int, site_data: SiteData) -> Dict[str, Any]:
        """
        Update an existing site
        
        Args:
            location_id: ID of the location to update
            site_data: SiteData object with updated information
            
        Returns:
            Dict containing the API response
        """
        url = f"{self.base_url}/customer/edit_location/save/?location_id={location_id}"
        headers = self._get_headers()
        
        try:
            response = requests.post(url, headers=headers, json=site_data.to_dict())
            response.raise_for_status()
            
            logger.info(f"Site ID {location_id} updated successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update site {location_id}: {e}")
            raise PulseProAPIException(f"Site update failed: {e}")
    
    def get_all_sites(self) -> Dict[str, Any]:
        """
        Get all sites
        
        Returns:
            Dict containing all sites and metadata
        """
        url = f"{self.base_url}/customer/locations/"
        headers = self._get_headers()
        
        try:
            response = requests.post(url, headers=headers, json={})
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved {len(data.get('locations', []))} sites")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get sites: {e}")
            raise PulseProAPIException(f"Failed to retrieve sites: {e}")
    
    def get_site_by_id(self, location_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific site by ID
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dict containing site information or None if not found
        """
        sites_data = self.get_all_sites()
        sites = sites_data.get('locations', [])
        
        for site in sites:
            if site.get('id') == location_id:
                return site
        
        return None
    
    def delete_site(self, location_id: int) -> Dict[str, Any]:
        """
        Delete a site
        
        Args:
            location_id: ID of the location to delete
            
        Returns:
            Dict containing the API response
        """
        url = f"{self.base_url}/customer/locations/{location_id}/"
        headers = self._get_headers()
        
        try:
            response = requests.delete(url, headers=headers)
            response.raise_for_status()
            
            logger.info(f"Site ID {location_id} deleted successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete site {location_id}: {e}")
            raise PulseProAPIException(f"Site deletion failed: {e}")
    
    def assign_users_to_site(self, location_id: int, user_ids: List[int]) -> Dict[str, Any]:
        """
        Assign multiple users to a site
        
        Args:
            location_id: ID of the location
            user_ids: List of user IDs to assign
            
        Returns:
            Dict containing the API response
        """
        url = f"{self.base_url}/customer/add_location_to_multiple_user/"
        headers = self._get_headers()
        
        payload = {
            "location_id": location_id,
            "user_id_list": user_ids
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            logger.info(f"Assigned {len(user_ids)} users to site ID {location_id}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to assign users to site {location_id}: {e}")
            raise PulseProAPIException(f"User assignment failed: {e}")
    
    def unassign_users_from_site(self, mapped_location_ids: List[int]) -> Dict[str, Any]:
        """
        Unassign users from a site
        
        Args:
            mapped_location_ids: List of mapped location IDs to remove
            
        Returns:
            Dict containing the API response
        """
        url = f"{self.base_url}/customer/delete_user_to_location_mapping/"
        headers = self._get_headers()
        
        payload = {
            "mapped_location_id_list": mapped_location_ids
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            logger.info(f"Unassigned {len(mapped_location_ids)} user mappings")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to unassign users: {e}")
            raise PulseProAPIException(f"User unassignment failed: {e}")
    
    def get_site_users(self, location_id: int, search_keyword: str = "") -> Dict[str, Any]:
        """
        Get all users available for assignment to a site
        
        Args:
            location_id: ID of the location
            search_keyword: Optional search keyword to filter users
            
        Returns:
            Dict containing users with their assignment status
        """
        url = f"{self.base_url}/customer/get_all_users_added_and_not_added_to_location/{location_id}/"
        headers = self._get_headers()
        
        payload = {
            "search_keyword": search_keyword
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved {len(data.get('users', []))} users for site ID {location_id}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get users for site {location_id}: {e}")
            raise PulseProAPIException(f"Failed to retrieve users: {e}")

class AIAgentProcessor:
    """AI Agent that processes natural language requests for site management"""
    
    def __init__(self, site_manager: SiteManager):
        self.site_manager = site_manager
        self.command_mappings = {
            # Site CRUD operations
            'create': ['create', 'add', 'new', 'make'],
            'update': ['update', 'edit', 'modify', 'change'],
            'delete': ['delete', 'remove', 'destroy'],
            'show': ['show', 'list', 'get', 'display', 'view'],
            
            # User assignment operations
            'assign': ['assign', 'add user', 'attach'],
            'unassign': ['unassign', 'remove user', 'detach']
        }
    
    def process_request(self, request: str) -> Dict[str, Any]:
        """
        Process natural language request and execute corresponding action
        
        Args:
            request: Natural language request
            
        Returns:
            Dict containing the result and response message
        """
        request_lower = request.lower()
        
        try:
            if any(word in request_lower for word in self.command_mappings['create']):
                return self._handle_create_request(request)
            
            elif any(word in request_lower for word in self.command_mappings['update']):
                return self._handle_update_request(request)
            
            elif any(word in request_lower for word in self.command_mappings['delete']):
                return self._handle_delete_request(request)
            
            elif any(word in request_lower for word in self.command_mappings['show']):
                return self._handle_show_request(request)
            
            elif any(word in request_lower for word in self.command_mappings['assign']):
                return self._handle_assign_request(request)
            
            elif any(word in request_lower for word in self.command_mappings['unassign']):
                return self._handle_unassign_request(request)
            
            else:
                return {
                    'success': False,
                    'message': 'I can help you with: creating, updating, deleting, showing sites, and assigning/unassigning users to sites.',
                    'available_commands': list(self.command_mappings.keys())
                }
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                'success': False,
                'message': f'An error occurred: {str(e)}'
            }
    
    def _handle_create_request(self, request: str) -> Dict[str, Any]:
        """Handle site creation request"""
        # This would ideally use NLP to extract site information
        # For now, return a template for required information
        return {
            'success': False,
            'message': 'To create a site, I need the following information:',
            'required_fields': [
                'location_name (required)',
                'address_field1 (required)', 
                'country_id (required)',
                'state_id (required)',
                'city_id (required)',
                'reporting_timezone (required)'
            ],
            'example': {
                'location_name': 'Mumbai Office',
                'address_field1': 'Bandra West',
                'country_id': 1,
                'state_id': 61,
                'city_id': 19788,
                'reporting_timezone': 'Asia/Kolkata'
            }
        }
    
    def _handle_update_request(self, request: str) -> Dict[str, Any]:
        """Handle site update request"""
        return {
            'success': False,
            'message': 'To update a site, please provide the location_id and the fields to update.'
        }
    
    def _handle_delete_request(self, request: str) -> Dict[str, Any]:
        """Handle site deletion request"""
        return {
            'success': False,
            'message': 'To delete a site, please provide the location_id.'
        }
    
    def _handle_show_request(self, request: str) -> Dict[str, Any]:
        """Handle show sites request"""
        try:
            sites_data = self.site_manager.get_all_sites()
            sites = sites_data.get('locations', [])
            
            # Format response for better readability
            formatted_sites = []
            for site in sites:
                formatted_sites.append({
                    'id': site.get('id'),
                    'name': site.get('location_name'),
                    'address': site.get('full_address'),
                    'city': site.get('city'),
                    'state': site.get('state'),
                    'country': site.get('country')
                })
            
            return {
                'success': True,
                'message': f'Found {len(formatted_sites)} sites',
                'data': {
                    'sites': formatted_sites,
                    'total_count': sites_data.get('total_count', len(formatted_sites))
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to retrieve sites: {str(e)}'
            }
    
    def _handle_assign_request(self, request: str) -> Dict[str, Any]:
        """Handle user assignment request"""
        return {
            'success': False,
            'message': 'To assign users to a site, please provide location_id and user_ids list.'
        }
    
    def _handle_unassign_request(self, request: str) -> Dict[str, Any]:
        """Handle user unassignment request"""
        return {
            'success': False,
            'message': 'To unassign users from a site, please provide mapped_location_id_list.'
        }

# Usage Example and Helper Functions
def initialize_system(refresh_token: str) -> AIAgentProcessor:
    """Initialize the complete system"""
    
    # Initialize authentication
    auth_manager = AuthenticationManager()
    auth_manager.set_refresh_token(refresh_token)
    
    # Initialize site manager
    site_manager = SiteManager(auth_manager)
    
    # Initialize AI agent processor
    ai_agent = AIAgentProcessor(site_manager)
    
    return ai_agent

def demo_usage():
    """Demonstration of how to use the system"""
    
    # Replace with actual refresh token
    REFRESH_TOKEN = "your_refresh_token_here"
    
    # Initialize system
    ai_agent = initialize_system(REFRESH_TOKEN)
    
    # Example requests
    requests_examples = [
        "show all sites",
        "create a new site",
        "delete site with id 123",
        "assign users to site",
        "list all locations"
    ]
    
    for request in requests_examples:
        print(f"\nProcessing: '{request}'")
        response = ai_agent.process_request(request)
        print(f"Response: {json.dumps(response, indent=2)}")

if __name__ == "__main__":
    demo_usage()