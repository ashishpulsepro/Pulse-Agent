"""
site_manager.py
Core site management functionality with clean, reusable classes
"""

import requests
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

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
        
        # Initialize tokens with refresh token from environment if available
        refresh_token = os.getenv('refresh')
        if refresh_token:
            self.tokens = AuthTokens(access_token="", refresh_token=refresh_token)
    
    def set_refresh_token(self, refresh_token: str) -> None:
        """Set the refresh token for authentication"""
        self.tokens = AuthTokens(access_token="", refresh_token=refresh_token)
    
    def refresh_access_token(self) -> str:
        """Refresh the access token using refresh token"""
       
        url = f"{self.base_url}/api/refresh/"
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json',
            'Origin': 'https://staging.pulsepro.ai',
            'Referer': 'https://staging.pulsepro.ai/',
        }

        refresh=os.getenv('refresh')
        print("refresh token: ", refresh)
        payload = {"refresh": refresh}
        print("got it")
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            access_token = data.get('access')
            print("access token: ", access_token)
            
            # Initialize tokens if None, then set access token
            if self.tokens is None:
                self.tokens = AuthTokens(access_token="", refresh_token=refresh or "")
            
            self.tokens.access_token = access_token
            logger.info("Access token refreshed successfully")

            print("Access token refreshed successfully")
            
            return access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to refresh access token: {e}")
            raise PulseProAPIException(f"Token refresh failed: {e}")
    
    def get_access_token(self) -> str:
        """Get current access token, refresh if needed"""
        if not self.tokens or not self.tokens.access_token:
            print("in")
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
        """Create a new site"""
        print("Creating site with data: ", site_data.to_dict())
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
        """Update an existing site"""
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
    
    def create_site_by_name_only(self, location_name: str) -> Dict[str, Any]:
        """Create a site with only the location name"""
        url = f"{self.base_url}/customer/save_loc_by_only_name/"
        headers = self._get_headers()
        
        payload = {
            "location_name": location_name
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Site '{location_name}' created successfully with minimal data")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create site by name only: {e}")
            try:
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")
            except:
                pass
            raise PulseProAPIException(f"Site creation by name failed: {e}")

    def get_all_sites(self) -> Dict[str, Any]:
        """Get all sites"""
        url = f"{self.base_url}/customer/locations/"
        headers = self._get_headers()
        
        try:
            # Try POST with empty JSON first
            response = requests.post(url, headers=headers, json={})
            
            # If POST fails, try GET request
            if response.status_code == 400:
                print("POST request failed, trying GET...")
                response = requests.get(url, headers=headers)
            
            response.raise_for_status()
            
            data = response.json()
            print(f"Sites API response: {data}")  # Debug print
            logger.info(f"Retrieved {len(data.get('locations', []))} sites")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get sites: {e}")
            # Print response content for debugging
            try:
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")
            except:
                pass
            raise PulseProAPIException(f"Failed to retrieve sites: {e}")
    
    def get_site_by_id(self, location_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific site by ID"""
        sites_data = self.get_all_sites()
        sites = sites_data.get('locations', [])
        
        for site in sites:
            if site.get('id') == location_id:
                return site
        
        return None
    
    def delete_site(self, location_id: int) -> Dict[str, Any]:
        """Delete a site"""
        url = f"{self.base_url}/customer/locations/{location_id}/"
        headers = self._get_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            logger.info(f"Site ID {location_id} deleted successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete site {location_id}: {e}")
            raise PulseProAPIException(f"Site deletion failed: {e}")
    
    def assign_users_to_site(self, location_id: int, user_ids: List[int]) -> Dict[str, Any]:
        """Assign multiple users to a site"""
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
        """Unassign users from a site"""
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
        """Get all users available for assignment to a site"""
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
        
    def get_all_users(self,limit=50, offset=0):
        url = "https://staging-api.pulsepro.ai/customer/get_team_list/"
        headers = self._get_headers()

        payload = {
        "count": "",
        "limit": limit,
        "offset": offset,
        "orderDir": "desc",
        "orderBy": "id",
        "search_keyword": ""
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            team = data.get("myTeam", [])
            names = [member.get("member_name") for member in team]
            return names
        else:
            print("Error:", response.status_code, response.text)
            return None


class PermissionManager:
    """Class for managing permission sets"""
    
    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
        self.base_url = auth_manager.base_url
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication"""
        access_token = self.auth_manager.get_access_token()
        print("access token in permission manager: ", access_token)
        return {
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'Origin': 'https://staging.pulsepro.ai',
            'Referer': 'https://staging.pulsepro.ai/',
        }
    
    def get_all_permission_sets(self) -> List[Dict[str, Any]]:
        """Get all permission sets with id and name"""
        url = f"{self.base_url}/customer/get_all_permission_bundles/"
        headers = self._get_headers()
    
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        
            data = response.json()
            permission_sets = [
                {
                    'id': perm.get('id'),
                    'name': perm.get('name')
                } 
                for perm in data
            ]
            logger.info(f"Retrieved {len(permission_sets)} permission sets")
            return permission_sets
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get permission sets: {e}")
            raise PulseProAPIException(f"Failed to retrieve permission sets: {e}")
        
    def get_permission_set_id_by_name(self, name: str) -> Optional[int]:
        """Get permission set ID by name"""
        permission_sets = self.get_all_permission_sets()
        for perm in permission_sets:
            if perm['name'].lower() == name.lower():
                return perm['id']
        return None



class UserManager:   
    """Class for managing users"""
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
    
    def create_user(self, first_name: str,last_name:str, email: str, permission_set_ids: List[int]) -> Dict[str, Any]:
        """Create a new user"""
        url = f"{self.base_url}/customer/add_team_member/"
        headers = self._get_headers()
        
        payload = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "permissionSets": permission_set_ids
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            logger.info(f"User '{first_name}' created successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create user '{first_name}': {e}")
            raise PulseProAPIException(f"User creation failed: {e}")
    
    def delete_user(self, user_id: int) -> Dict[str, Any]:
        """Delete a user"""
        url = f"{self.base_url}/customer/get_team_list/{user_id}/"
        headers = self._get_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            logger.info(f"User ID {user_id} deleted successfully")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete user ID {user_id}: {e}")
            raise PulseProAPIException(f"User deletion failed: {e}")

    def get_all_users(self,limit=50, offset=0):
        url = "https://staging-api.pulsepro.ai/customer/get_team_list/"
        headers = self._get_headers()

        payload = {
        "count": "",
        "limit": limit,
        "offset": offset,
        "orderDir": "desc",
        "orderBy": "id",
        "search_keyword": ""
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            team = data.get("myTeam", [])
            
            users=[{
                'id':user.get("id"),
                'name':user.get("member_name"),
            } for user in team]

            return users
        else:
            print("Error:", response.status_code, response.text)
            return None
        
    def get_user_id_by_name(self, name: str) -> Optional[int]:
        """Get user ID by name"""
        users = self.get_all_users()
        for user in users:
            if user['name'].lower() == name.lower():
                return user['id']
        return None