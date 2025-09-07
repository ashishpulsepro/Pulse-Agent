"""Local copies of managers so backend_lang is self-contained.
Minimal adjustments: removed debug prints where noisy.
"""
from __future__ import annotations
import os, requests, logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class AuthTokens:
    access_token: str
    refresh_token: str
    expires_at: Optional[datetime] = None

class PulseProAPIException(Exception):
    pass

class AuthenticationManager:
    def __init__(self, base_url: str = "https://staging-api.pulsepro.ai"):
        self.base_url = base_url
        self.tokens: Optional[AuthTokens] = None
        refresh_token = os.getenv('refresh')
        if refresh_token:
            self.tokens = AuthTokens(access_token="", refresh_token=refresh_token)

    def refresh_access_token(self) -> str:
        url = f"{self.base_url}/api/refresh/"
        headers = {'Accept':'application/json','Content-Type':'application/json'}
        refresh = os.getenv('refresh')
        if not refresh:
            raise PulseProAPIException("Missing refresh token env var 'refresh'")
        payload = {"refresh": refresh}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            access = data.get('access')
            if not access:
                raise PulseProAPIException("No access token in refresh response")
            if self.tokens is None:
                self.tokens = AuthTokens(access_token=access, refresh_token=refresh)
            else:
                self.tokens.access_token = access
            return access
        except requests.RequestException as e:
            raise PulseProAPIException(f"Refresh failed: {e}")

    def get_access_token(self) -> str:
        if not self.tokens or not self.tokens.access_token:
            return self.refresh_access_token()
        return self.tokens.access_token

class BaseManager:
    def __init__(self, auth: AuthenticationManager):
        self.auth = auth
        self.base_url = auth.base_url

    def _headers(self):
        token = self.auth.get_access_token()
        return {
            'Accept':'application/json, text/plain, */*',
            'Authorization': f'Bearer {token}',
            'Content-Type':'application/json',
            'Origin':'https://staging.pulsepro.ai',
            'Referer':'https://staging.pulsepro.ai/'
        }

class SiteManager(BaseManager):
    def create_site_by_name_only(self, location_name: str):
        url = f"{self.base_url}/customer/save_loc_by_only_name/"
        payload = {"location_name": location_name}
        resp = requests.post(url, headers=self._headers(), json=payload)
        if resp.status_code == 200:
            return {'message': f"✅ Site **{location_name}** created successfully!"}
        if resp.status_code == 400:
            try:
                return {'message': resp.json().get('message','Failed')}
            except Exception:
                pass
        resp.raise_for_status()
        return {'message': 'Unknown response'}

    def get_all_sites(self):
        url = f"{self.base_url}/customer/locations/"
        r = requests.post(url, headers=self._headers(), json={})
        if r.status_code == 400:
            r = requests.get(url, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def get_site_users(self, location_id: int):
        url = f"{self.base_url}/customer/get_all_users_added_and_not_added_to_location/{location_id}/"
        r = requests.post(url, headers=self._headers(), json={"search_keyword": ""})
        r.raise_for_status()
        return r.json()

    def assign_users_to_site(self, location_id: int, user_ids: List[int]):
        url = f"{self.base_url}/customer/add_location_to_multiple_user/"
        payload = {"location_id": location_id, "user_id_list": user_ids}
        r = requests.post(url, headers=self._headers(), json=payload)
        r.raise_for_status()
        return r.json()

class PermissionManager(BaseManager):
    def get_all_permission_sets(self):
        url = f"{self.base_url}/customer/get_all_permission_bundles/"
        r = requests.get(url, headers=self._headers())
        r.raise_for_status()
        data = r.json()
        return [{ 'id': p.get('id'), 'name': p.get('name') } for p in data]

    def get_permission_set_id_by_name(self, name: str) -> Optional[int]:
        for p in self.get_all_permission_sets():
            if p['name'].lower() == name.lower():
                return p['id']
        return None

class UserManager(BaseManager):
    def get_all_users(self, limit=50, offset=0):
        url = f"{self.base_url}/customer/get_team_list/"
        payload = {"count":"","limit":limit,"offset":offset,"orderDir":"desc","orderBy":"id","search_keyword":""}
        r = requests.post(url, headers=self._headers(), json=payload)
        r.raise_for_status()
        team = r.json().get('myTeam', [])
        return [{
            'id': m.get('id'),
            'name': m.get('member_name'),
            'user': m.get('user_id'),
            'email': m.get('email')
        } for m in team]

    def create_user(self, first_name: str, last_name: str, email: str, permission_set_ids: List[int]):
        existing = self.get_all_users()
        if any(u.get('email') == email for u in existing):
            return {'message': f"User **{first_name}** with email **{email}** already exist"}
        url = f"{self.base_url}/customer/add_team_member/"
        payload = {"first_name": first_name, "last_name": last_name, "email": email, "permissionSets": permission_set_ids}
        r = requests.post(url, headers=self._headers(), json=payload)
        r.raise_for_status()
        return {'message': f"✅ User **{first_name} {last_name}** created successfully!"}

class TemplateManager(BaseManager):
    def get_all_templates_with_id(self):
        url = f"{self.base_url}/customer/forms/"
        payload = {"limit":50,"offset":0,"template_status":"Active","orderDir":"desc","orderBy":"id","search_keyword":""}
        r = requests.post(url, headers=self._headers(), json=payload)
        r.raise_for_status()
        return [{
            'id': t.get('id'),
            'name': t.get('form_name'),
            'created_by': t.get('created_by')
        } for t in r.json().get('forms', [])]

    def get_checklist_id_by_name(self, checklist_name: str):
        for t in self.get_all_templates_with_id():
            if t['name'].lower() == checklist_name.lower():
                return t['id']
        return None

    def create_checklist(self, checklist_id: int, checklist_name: str):
        url = f"{self.base_url}/customer/get_checklist_convert_into_meta/{checklist_id}/"
        r = requests.get(url, headers=self._headers())
        if r.status_code == 200:
            return {'message': f"✅ Template **{checklist_name}** created successfully!"}
        r.raise_for_status()
        return {'message': 'Unknown response'}
