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
                        'message':f"User **{first_name}** with email **{email}** already exist \n Would you like to create user again or start new operation"
                    }
                
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            logger.info(f"User '{first_name}' created successfully")
            response.json()
            return {
                'message': f"✅ User **{first_name} {last_name}** created successfully!\n Would you like to create user again or start new operation"
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


    @staticmethod
    def get_current_user(request: Request)->Dict[str,Any]:
        """Dependency to verify user via /get_profile API"""

        url=f"https://staging-api.pulsepro.ai/common/get_profile/"
        print("inside get current user")
        print("header: ", request.headers)

        auth_header = request.headers.get("authorization")
        print("auth_header: ", auth_header)
        refresh=''
        if auth_header.lower().startswith("bearer "):
            refresh = auth_header.split(" ", 1)[1]
        else:
            refresh = auth_header
        
        auth = AuthenticationManager()
        auth.set_refresh_token(refresh_token=refresh)

        access=auth.get_access_token()
        token_head=f"Bearer {access}"

        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing",
            )
        

        try:
            response = requests.get(
                url,
                headers={"Authorization": token_head, "Accept": "application/json, text/plain, */*"}
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                )
            return response.json()   # 👈 return the full profile dict
        except requests.RequestException as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Profile service unavailable: {str(e)}",
            )




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
