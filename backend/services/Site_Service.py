import requests
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

import os
import requests
from fastapi import Depends, HTTPException, status, Request
from services.Authentication_Service import AuthenticationManager

from collections import defaultdict
from typing import List, Dict, Any
import requests
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class PulseProAPIException(Exception):
    """Custom exception for API errors"""
    pass



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









class SiteManager:
    """Main class for managing sites and user assignments"""
    
    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
        self.base_url = auth_manager.base_url
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication"""
        access_token = self.auth_manager.get_access_token()
        print("get_header")
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
                    'message':f"Site **{location_name}** already exist\n Would you like to create site again or start new operation"
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
                'message': f"✅ Site **{location_name}** created successfully! \n Would you like to create site again or start new operation"
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
        print("get_all_site")
        
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
