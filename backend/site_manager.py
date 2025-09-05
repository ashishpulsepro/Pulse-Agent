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

        existing_sites=self.get_all_sites().get('locations')
        for site in existing_sites:
            if site.get('location_name').lower() == location_name.lower():
                return {
                    'message':f"Site **{location_name}** already exist"
                }
        
        payload = {
            "location_name": location_name
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Site '{location_name}' created successfully with minimal data")
            return {
                'message': f"✅ Site **{location_name}** created successfully!"
            }
            
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
            

            if response.status_code == 200:
                logger.info(f"Site ID {location_id} deleted successfully")
                return {"success": True, "message": "Site deleted successfully"}
            
            elif response.status_code == 400:
                return {
                "success": False,
                "message": response.json().get("message", "Failed to delete site"),
                }
            else:
                response.raise_for_status()
            
            
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
    
    def assign_permission_sets_to_user(self, user_id: int, permission_set_ids: List[int]) -> Dict[str, Any]:
        """Assign multiple permission sets to a user"""
        url = f"{self.base_url}/customer/assign_permission_set_to_user/"
        headers = self._get_headers()
        print("permission set ids: ", permission_set_ids)
        print("user id: ", user_id)
        
        
        payload = {
            "user_id": [user_id],
            "permission_set_id": permission_set_ids
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            logger.info(f"Assigned {len(permission_set_ids)} permission sets to user ID {user_id}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to assign permission sets to user {user_id}: {e}")
            raise PulseProAPIException(f"Permission set assignment failed: {e}")
        
    def unassign_permission_sets_from_user(self, user_id: int, permission_set_ids: List[int]) -> Dict[str, Any]:
        """Unassign multiple permission sets from a user"""
        url = f"{self.base_url}/customer/remove_permission_set_from_user/"
        headers = self._get_headers()
        print("permission set ids: ", permission_set_ids)
        print("user id: ", user_id) 
        
        payload = {
            "user_id": [user_id],
            "permission_set_id": permission_set_ids
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            logger.info(f"Unassigned {len(permission_set_ids)} permission sets from user ID {user_id}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to unassign permission sets from user {user_id}: {e}")
            raise PulseProAPIException(f"Permission set unassignment failed: {e}")



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
            existing_users=self.get_all_users()
            for user in existing_users:
                if user["email"] == email:
                    return {
                        'message':f"User **{first_name}** with email **{email}** already exist"
                    }
                
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            logger.info(f"User '{first_name}' created successfully")
            response.json()
            return {
                'message': f"✅ User **{first_name} {last_name}** created successfully!"
            }
            
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
                'user':user.get("user_id"),
                'email':user.get("email"),
                'permission_set':user.get("permission_bundles")
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
                print("user: ", user['id'])
                return user['id']
        return None
    
    def get_user_by_name(self, name: str) -> Optional[int]:
        """Get user ID by name"""
        users = self.get_all_users()
        for user in users:
            if user['name'].lower() == name.lower():
                print("user: ", user['user'])
                return user['user']
        return None
    
    def get_customer_access_settings(self) -> Dict[str, Any]:
        """Get customer access settings"""
        url = f"{self.base_url}/customer/get_customer_access_settings/"
        headers = self._get_headers()
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            logger.info("Retrieved customer access settings")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get customer access settings: {e}")
            raise PulseProAPIException(f"Failed to retrieve customer access settings: {e}")
        
    def update_customer_setting(self, accessToAllSite: bool, accessToAllChecklist: bool) -> Dict[str, Any]:
        """Update customer access settings"""
        url = f"{self.base_url}/customer/update_customer_setting/"
        headers = self._get_headers()
        
        payload = {
            "accessToAllSite": accessToAllSite,
            "accessToAllChecklist": accessToAllChecklist
        }
        print("in update_customer_setting")
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info("Updated customer access settings")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update customer access settings: {e}")
            raise PulseProAPIException(f"Failed to update customer access settings: {e}")
        

    def get_all_groups(self) -> List[Dict[str,Any]]:
        """Get all the groups"""
        print("inside get all groups")
        url=f"{self.base_url}/customer/get_groups/"
        headers =self._get_headers()
        print("header")

        payload={"count":"","limit":100,"offset":0,"orderDir":"desc","orderBy":"id","search_keyword":""}
        
        print("payload")

        try: 
            response=requests.post(url,headers=headers,json=payload)
            response.raise_for_status()
            print("groups: ",response.json())

            data=response.json().get("groupList")

            return data
        except requests.exceptions.RequestException as e:
            logger.error(e)

    def get_group_id_by_name(self,name:str)->int:

        groups=self.get_all_groups()
        for group in groups:
            if group.get("name").lower()==name.lower():
                return group.get("id")
        return 0
    
    def create_a_group(self,group_name:str)->Dict[str,Any]:
        """creation of group"""

        url=f"{self.base_url}/customer/add_group/"
        headers=self._get_headers()

        payload={
            "group_name":group_name
        }
        print(payload)
        try:
            response=requests.post(url,headers=headers,json=payload)
            response.raise_for_status()
            data=response.json()
            return data
        except requests.exceptions.RequestException as e:
            logger.error(e)
        


    def add_multiple_user_to_group(self,userIds:List , groupId:int) ->Dict[str,Any]:
        """Add multiple user to groups"""

        url=f"{self.base_url}/customer/add_multiple_user_to_group/"
        headers=self._get_headers()

        payload={
            "userIds":userIds,
            "group_id":groupId
        }

        try:
            response=requests.post(url,headers=headers,json=payload)
            response.raise_for_status()

            data=response.json()

            return data
        except requests.exceptions.RequestException as e:
            logger.error(e)


    def delete_multiple_group_user(self, group_user_ids:List)->Dict[str,Any]:
        """delete multiple user from group"""
        url=f"{self.base_url}/customer/delete_multiple_group_user/"
        headers=self._get_headers()

        payload={
            "group_user_ids":group_user_ids
        }
        print("inside delete_multiple_group_user ")

        try:
            response=requests.post(url,headers=headers,json=payload)
            response.raise_for_status()

            data=response.json()
            return data
        except requests.exceptions.RequestException as e:
            logger.error(e)

    def delete_group(self,groupId:int)->Dict[str,Any]:
        """Delete group forcefully"""
        users=self.get_users_added_to_group(groupId=groupId)
        print("users: ",users)
        user_ids=[
                user.get("id")
               for user in users ]
        print("user_ids: ",user_ids)
        if user_ids:
            result=self.delete_multiple_group_user(user_ids)

        url=f"{self.base_url}/customer/delete_group/{groupId}/"
        headers=self._get_headers()

        try:
            response=requests.get(url,headers=headers)
            response.raise_for_status()
            data=response.json()
            return data
        except requests.exceptions.RequestException as e:
            logger.error(e)

    def get_users_added_to_group(self,groupId:int)->List[Dict[str,Any]]:
        """get all the users already added to a group with groupId"""

        url=f"{self.base_url}/customer/get_group_users/{groupId}/"
        headers=self._get_headers()
        print("inside get_users_added_to_group")

        try:
            response=requests.get(url,headers=headers)
            response.raise_for_status()
            print("response: ",response.json())

            data=response.json().get("users")
            return data if data else []
        except requests.exceptions.RequestException as e:
            logger.error(e)

    def get_all_members_not_added_to_group(self,group_id:int)->List[Dict[str,Any]]:
        """get all the users not added to the group"""

        url=f"{self.base_url}/customer/get_all_members_not_added_to_group/{group_id}/"
        headers=self._get_headers()

        try:
            response=requests.get(url,headers=headers)
            response.raise_for_status()

            data=response.json().get("users")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(e)




from collections import defaultdict
from typing import List, Dict, Any
import requests
from sqlalchemy import text





class TemplateManager:
    """Manage templates for users"""
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

    def get_all_templates_with_id(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/customer/forms/"
        headers = self._get_headers()

        payload = {
            "limit": 50,
            "offset": 0,
            "template_status": "Active",
            "orderDir": "desc",
            "orderBy": "id",
            "search_keyword": ""
        }

        try:
            response = requests.post(url, headers=headers,json=payload)
            response.raise_for_status()

            result = [ {
                "id": template.get("id"),
                "name": template.get("form_name"),
                "created_by": template.get("created_by"),
                "is_scheduled": template.get("scheduled"),
                "created_by_id": template.get("created_by_id")

            } for template in response.json().get("forms", [])]

            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve templates: {e}")
            return {"error": str(e)}
        

    def get_all_templates(self) -> List[str]:

        url = f"{self.base_url}/customer/forms/"
        headers = self._get_headers()

        payload = {
            "limit": 50,
            "offset": 0,
            "template_status": "Active",
            "orderDir": "desc",
            "orderBy": "id",
            "search_keyword": ""
        }

        try:
            response = requests.post(url, headers=headers,json=payload)
            response.raise_for_status()

            result = [ {

                "name": template.get("form_name"),
                "created_by": template.get("created_by"),

            } for template in response.json().get("forms", [])]

            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve templates: {e}")
            return {"error": str(e)}


    def get_all_template_added_not_added_to_user(self, user_id: int) ->  List[Dict[str, Any]]:
        """Get all templates added and not added to a specific user"""

        url = f"{self.base_url}/customer/get_all_templates_added_and_not_added_to_user/{user_id}/"
        headers = self._get_headers()

        payload={}

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            print("response: ", response.json())
            json_data = response.json()
            result = json_data.get("templates", []) if isinstance(json_data, dict) else []
            print("Templates retrieved: ", result)
            final_result=[
                {
                    "id": template.get("id"),
                    "template_name": template.get("template_name")
                }
                for template in result if isinstance(template, dict)
            ]   

            return final_result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve templates: {e}")
            return {"error": str(e)}

    def get_assign_user_template_id_by_name(self, template_name: str,user_id:int) -> Optional[int]:
        """Get assigned template IDs for a user"""
        print("calling get_assign_user_template_id_by_name")
        all_templates = self.get_all_template_added_not_added_to_user(user_id=user_id)
        try:
            for template in all_templates:
                if template.get("template_name").lower() == template_name.lower():
                    return template.get("id") 
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve assigned templates: {e}")
            return 0

    def get_template_id_by_name(self, template_name: str) -> Optional[int]:
        """Get template ID by name (case-insensitive)"""
        all_templates = self.get_all_templates_with_id()
        template_name_lower = template_name.lower()
    
        for template in all_templates:
            if template.get("name", "").lower() == template_name_lower:
                return template.get("id")
        return None


    def delete_template(self, template_id: int) -> bool:
        """Delete template by ID"""
        url = f"{self.base_url}/customer/del_form/{template_id}"
        headers = self._get_headers()

        try:
            print("requesting deletion of template ID:", template_id)
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete template: {e}")
            return False

    def assign_templates_to_user(self, user_id: int, template_ids: List[Any]) -> bool:
        """Assign templates to a user"""
        url = f"{self.base_url}/customer/map_checklist_to_user/{user_id}/"
        headers = self._get_headers()

        formatted_template_ids = [{"id": template_id} for template_id in template_ids]
        print("formatted template IDs: ", formatted_template_ids)

        payload = {
            "checklists": formatted_template_ids,
            "type":"is_invited_user"
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to assign templates to user: {e}")
            return False

    def unassign_templates_from_user(self, user_id: int, template_ids: List[Any]) -> bool:
        """Unassign templates from a user"""
        url = f"{self.base_url}/customer/delete_mapped_checklist_from_user/{user_id}/"
        headers = self._get_headers()

        formatted_template_ids = [{"id": template_id} for template_id in template_ids]

        payload = {
            "checklists": formatted_template_ids,
            "type": "is_invited_user"
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to unassign templates from user: {e}")
            return False
        


    def get_industry_list(self) -> List[Dict[str, Any]]:
        """Get list of industries"""
        url = f"{self.base_url}/checklist/get_industry_list/"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            industries = [
                {
                    'id': industry.get('id'),
                    'name': industry.get('name')
                } 
                for industry in data
            ]
            logger.info(f"Retrieved {len(industries)} industries")
            return industries

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get industries: {e}")
            raise PulseProAPIException(f"Failed to retrieve industries: {e}")
        

    def get_industry_id_by_name(self, industry_name: str) -> Optional[int]:
        """Get industry ID by name"""
        industries = self.get_industry_list()
        for industry in industries:
            if industry['name'].lower() == industry_name.lower():
                return industry['id']
        return None



    def get_checklist_industry_wise(self, industry_id: int) -> List[Dict[str, Any]]:
        """Get checklists by industry ID"""
        url = f"{self.base_url}/checklist/get_checklist_industry_wise/"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers,json={"industry_id":industry_id})
            response.raise_for_status()

            data = response.json()
            checklists = [
                {
                    'id': checklist.get('id'),
                    'name': checklist.get('checklist_name'),
                    'industry_id': checklist.get('industry')
                } 
                for checklist in data.get('checklists', [])
            ]
            logger.info(f"Retrieved {len(checklists)} checklists for industry ID {industry_id}")
            return checklists
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get checklists for industry ID {industry_id}: {e}")
            raise PulseProAPIException(f"Failed to retrieve checklists: {e}")



    def create_checklist(self, checklist_id: int,checklist_name:str) -> Dict[str, Any]:
        """Save a new checklist"""
        url = f"{self.base_url}/customer/get_checklist_convert_into_meta/{checklist_id}/"
        headers = self._get_headers()

        existing_checklists=self.get_all_templates()

        for checklist in existing_checklists:
            if checklist.get('name').lower()==checklist_name.lower():
                return{
                    'message': f"Checklist **{checklist_name}** already exist"
                }


        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            logger.info(f"Checklist  saved successfully")
            response.json()
            return{
                'message':f"✅ Template **{checklist_name}** created successfully!"
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to save checklist '{checklist_id}': {e}")
            raise PulseProAPIException(f"Checklist save failed: {e}")




    def get_all_ready_made_checklists(self) -> List[Dict[str, Any]]:
        """Get top 2 checklists per industry by download_count"""
        url = f"{self.base_url}/checklist/get_checklist_industry_wise/"
        headers = self._get_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            checklists = data.get("checklists", [])

            # Group by industry
            industry_checklists = defaultdict(list)
            for checklist in checklists:
                industry_id = checklist.get("industry")
                industry_checklists[industry_id].append(checklist)

            # Pick top 2 checklists per industry by download_count
            top_checklists = []
            for industry_id, items in industry_checklists.items():
                top_items = sorted(
                    items,
                    key=lambda x: x.get("download_count", 0),
                    reverse=True
                )[:5]
                top_checklists.extend(top_items)

            return [
                {
                    "id": c.get("id"),
                    "name": c.get("checklist_name"),
                    "industry_id": c.get("industry")
                }
                for c in top_checklists
            ]

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to retrieve checklists: {e}")


    def get_checklist_id_by_name(self, checklist_name: str) -> int:
        """Get checklist details by ID"""
        all_checklists = self.get_all_ready_made_checklists()
        for checklist in all_checklists:
            if checklist.get('name').lower() == checklist_name.lower():
                return checklist.get('id')
        return None