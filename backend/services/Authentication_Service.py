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


from collections import defaultdict
from typing import List, Dict, Any
import requests
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()



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
       
    def set_refresh_token(self, refresh_token: str) -> None:
        """Set the refresh token for authentication"""
        self.tokens = AuthTokens(access_token="", refresh_token=refresh_token)
    
    def refresh_access_token(self) -> str:
        """Refresh the access token using refresh token"""
        print("in refresh_access_token")
        url = f"{self.base_url}/api/refresh/"
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json',
            'Origin': 'https://staging.pulsepro.ai',
            'Referer': 'https://staging.pulsepro.ai/',
        }

        refresh = self.tokens.refresh_token 
        if not refresh:
            raise PulseProAPIException("No refresh token available")
        print("refresh token in refresh_access_token: ", refresh)
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
