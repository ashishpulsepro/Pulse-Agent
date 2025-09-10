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
                    'message': f"Checklist **{checklist_name}** already exist \n Would you like to create checklist again or try other new operation"
                }


        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            logger.info(f"Checklist  saved successfully")
            response.json()
            return{
                'message':f"✅ Template **{checklist_name}** created successfully!\n Would you like to create checklist again or try other new operation"
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