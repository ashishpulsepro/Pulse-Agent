from fastapi import File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
from typing import Dict, Any
import os
from pathlib import Path

from services.Authentication_Service import AuthenticationManager


class UploadFileService:
    """Handles Upload of files to PulsePro API"""
    
    # File validation settings
    ALLOWED_EXTENSIONS = {'.xlsx', '.xls','.csv'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    
    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
        self.base_url = auth_manager.base_url

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication"""
        access_token = self.auth_manager.get_access_token()
        return {
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {access_token}',
            'Origin': 'https://staging.pulsepro.ai',
            'Referer': 'https://staging.pulsepro.ai/',
        }

    @staticmethod
    def validate_file(file: UploadFile) -> bool:
        """Validate uploaded file"""
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in UploadFileService.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Only {', '.join(UploadFileService.ALLOWED_EXTENSIONS)} files are allowed."
            )
        return True

    @staticmethod
    def validate_file_size(file_content: bytes) -> bool:
        """Validate file size"""
        if len(file_content) > UploadFileService.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {UploadFileService.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        return True

    @staticmethod
    def validate_form_name(form_name: str) -> str:
        """Validate and clean form name"""
        if not form_name or not form_name.strip():
            raise HTTPException(
                status_code=400,
                detail="Form name cannot be empty"
            )
        return form_name.strip()

    def upload_to_pulsepro_api(
        self,
        file_content: bytes,
        filename: str,
        form_name: str
    ) -> Dict[Any, Any]:
        """Upload file to PulsePro API using bytes method"""
        try:
            # Get headers without Content-Type (requests will set it for multipart)
            headers = {
                'Accept': 'application/json, text/plain, */*',
                'Authorization': f'Bearer {self.auth_manager.get_access_token()}',
                'Origin': 'https://staging.pulsepro.ai',
                'Referer': 'https://staging.pulsepro.ai/',
            }
            
            url = f"{self.base_url}/customer/upload_form/"
            
            # Prepare multipart form data
            files = {
                'upload_file': (filename, file_content, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            }
            data = {
                'form_name': form_name
            }
            
            # Make API call to PulsePro
            response = requests.post(
                url=url,
                headers=headers,
                files=files,
                data=data,
                timeout=60  # Increased timeout for large files
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"PulsePro API error: {response.text}"
                )
                
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=408,
                detail="Request timeout. Please try again with a smaller file."
            )
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to PulsePro API. Please try again later."
            )
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Network error: {str(e)}"
            )

    async def process_file_upload(
        self,
        file: UploadFile,
        form_name: str
    ) -> Dict[Any, Any]:
        """Complete file upload process with all validations"""
        try:
            # Validate file type
            self.validate_file(file)
            
            # Validate form name
            clean_form_name = self.validate_form_name(form_name)
            
            # Read file content
            file_content = await file.read()
            
            # Validate file size
            self.validate_file_size(file_content)
            
            # Upload to PulsePro API
            result = self.upload_to_pulsepro_api(
                file_content=file_content,
                filename=file.filename,
                form_name=clean_form_name
            )
            
            return {
                "success": True,
                "message": "File uploaded successfully",
                "form_name": clean_form_name,
                "filename": file.filename,
                "file_size": len(file_content),
                "pulsepro_response": result
            }
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during file upload: {str(e)}"
            )