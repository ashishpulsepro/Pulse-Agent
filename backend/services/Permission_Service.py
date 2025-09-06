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
