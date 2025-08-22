"""
main.py - Simplified FastAPI endpoints for Site Manager
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Import your site manager
from site_manager import SiteManager, AuthenticationManager, SiteData, PulseProAPIException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Site Management API",
    description="Simple site management system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize site manager with default authentication
auth_manager = AuthenticationManager()
site_manager = SiteManager(auth_manager)

# ============================================
# PYDANTIC MODELS
# ============================================

class StandardResponse(BaseModel):
    """Standard API response format"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class SiteCreateByNameRequest(BaseModel):
    """Simple site creation request with only name"""
    location_name: str = Field(..., min_length=1, description="Name of the location")

class SiteCreateRequest(BaseModel):
    """Site creation request model"""
    location_name: str = Field(..., min_length=1, description="Name of the location")
    address_field1: str = Field(..., min_length=1, description="Primary address")
    address_field2: Optional[str] = Field("", description="Secondary address")
    country_id: int = Field(1, ge=1, description="Country ID (1=India, 2=USA)")
    state_id: int = Field(..., ge=1, description="State ID (numeric identifier)")
    city_id: int = Field(..., ge=1, description="City ID (numeric identifier)")
    pincode: Optional[str] = Field("", description="Postal/ZIP code")
    mobile: Optional[str] = Field("", description="Mobile/phone number")
    location_number: Optional[str] = Field("", description="Internal location number")
    location_code: Optional[str] = Field("", description="Internal location code")
    to_email: Optional[str] = Field("", description="Primary email address")
    cc_email: Optional[str] = Field("", description="CC email address")
    reporting_timezone: str = Field("UTC", description="Timezone for reporting")
    geo_fencing_enabled: bool = Field(False, description="Enable geo-fencing")
    geo_fencing_distance: int = Field(0, ge=0, description="Geo-fencing distance in meters")
    lat: float = Field(0.0, description="Latitude coordinate")
    lng: float = Field(0.0, description="Longitude coordinate")
    map_link: Optional[str] = Field("", description="Google Maps link")
    has_custom_field: bool = Field(False, description="Has custom fields")
    is_schedule_active: bool = Field(False, description="Is scheduling active")

    @field_validator('to_email', 'cc_email')
    @classmethod
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

class SiteUpdateRequest(BaseModel):
    """Site update request model"""
    location_name: Optional[str] = Field(None, min_length=1, description="Name of the location")
    address_field1: Optional[str] = Field(None, min_length=1, description="Primary address")
    address_field2: Optional[str] = Field(None, description="Secondary address")
    country_id: Optional[int] = Field(None, ge=1, description="Country ID")
    state_id: Optional[int] = Field(None, ge=1, description="State ID")
    city_id: Optional[int] = Field(None, ge=1, description="City ID")
    pincode: Optional[str] = Field(None, description="Postal/ZIP code")
    mobile: Optional[str] = Field(None, description="Mobile/phone number")
    location_number: Optional[str] = Field(None, description="Internal location number")
    location_code: Optional[str] = Field(None, description="Internal location code")
    to_email: Optional[str] = Field(None, description="Primary email address")
    cc_email: Optional[str] = Field(None, description="CC email address")
    reporting_timezone: Optional[str] = Field(None, description="Timezone for reporting")
    geo_fencing_enabled: Optional[bool] = Field(None, description="Enable geo-fencing")
    geo_fencing_distance: Optional[int] = Field(None, ge=0, description="Geo-fencing distance")
    lat: Optional[float] = Field(None, description="Latitude coordinate")
    lng: Optional[float] = Field(None, description="Longitude coordinate")
    map_link: Optional[str] = Field(None, description="Google Maps link")
    has_custom_field: Optional[bool] = Field(None, description="Has custom fields")
    is_schedule_active: Optional[bool] = Field(None, description="Is scheduling active")

# ============================================
# SITE MANAGEMENT ENDPOINTS
# ============================================

@app.post("/sites/quick-create", response_model=StandardResponse, tags=["Site Management"])
async def create_site_by_name_only(site_request: SiteCreateByNameRequest):
    """Create a new site with only the location name (quick creation)"""
    try:
        result = site_manager.create_site_by_name_only(site_request.location_name)
        
        logger.info(f"Site '{site_request.location_name}' created successfully (quick create)")
        
        return StandardResponse(
            success=True,
            message=f"Site '{site_request.location_name}' created successfully",
            data=result
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except PulseProAPIException as e:
        logger.error(f"Site creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Site creation failed: {str(e)}"
        )

@app.post("/sites", response_model=StandardResponse, tags=["Site Management"])
async def create_site(site_request: SiteCreateRequest):
    """Create a new site with the provided information"""
    try:
        # Convert request to SiteData
        site_data = SiteData(**site_request.dict())
        
        # Create site
        result = site_manager.create_site(site_data)
        
        logger.info(f"Site '{site_request.location_name}' created successfully")
        
        return StandardResponse(
            success=True,
            message=f"Site '{site_request.location_name}' created successfully",
            data=result
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except PulseProAPIException as e:
        logger.error(f"Site creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Site creation failed: {str(e)}"
        )

@app.get("/sites", response_model=StandardResponse, tags=["Site Management"])
async def get_all_sites():
    """Get all sites with their details"""
    try:
        result = site_manager.get_all_sites()
        sites_count = len(result.get('locations', []))
        
        logger.info(f"Retrieved {sites_count} sites")
        
        return StandardResponse(
            success=True,
            message=f"Successfully retrieved {sites_count} sites",
            data=result
        )
        
    except PulseProAPIException as e:
        logger.error(f"Failed to retrieve sites: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sites: {str(e)}"
        )

@app.get("/sites/{location_id}", response_model=StandardResponse, tags=["Site Management"])
async def get_site_by_id(location_id: int):
    """Get a specific site by its ID"""
    try:
        result = site_manager.get_site_by_id(location_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Site with ID {location_id} not found"
            )
        
        return StandardResponse(
            success=True,
            message=f"Site {location_id} retrieved successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except PulseProAPIException as e:
        logger.error(f"Failed to retrieve site {location_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve site: {str(e)}"
        )

@app.put("/sites/{location_id}", response_model=StandardResponse, tags=["Site Management"])
async def update_site(location_id: int, site_request: SiteUpdateRequest):
    """Update an existing site with new information"""
    try:
        # Get current site data
        current_site = site_manager.get_site_by_id(location_id)
        if not current_site:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Site with ID {location_id} not found"
            )
        
        # Create update data - only include fields that are not None
        update_data = {}
        for field, value in site_request.dict().items():
            if value is not None:
                update_data[field] = value
        
        # Merge with existing data for required fields
        merged_data = {
            'location_name': update_data.get('location_name', current_site.get('location_name')),
            'address_field1': update_data.get('address_field1', current_site.get('address_field1')),
            'country_id': update_data.get('country_id', current_site.get('country_id')),
            'state_id': update_data.get('state_id', current_site.get('state_id')),
            'city_id': update_data.get('city_id', current_site.get('city_id')),
            'reporting_timezone': update_data.get('reporting_timezone', current_site.get('reporting_timezone', 'UTC')),
        }
        
        # Add optional fields
        optional_fields = [
            'address_field2', 'pincode', 'mobile', 'location_number', 'location_code',
            'to_email', 'cc_email', 'geo_fencing_enabled', 'geo_fencing_distance',
            'lat', 'lng', 'map_link', 'has_custom_field', 'is_schedule_active'
        ]
        
        for field in optional_fields:
            if field in update_data:
                merged_data[field] = update_data[field]
            else:
                merged_data[field] = current_site.get(field, "")
        
        # Convert to SiteData
        site_data = SiteData(**merged_data)
        
        # Update site
        result = site_manager.update_site(location_id, site_data)
        
        logger.info(f"Site {location_id} updated successfully")
        
        return StandardResponse(
            success=True,
            message=f"Site {location_id} updated successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except PulseProAPIException as e:
        logger.error(f"Failed to update site {location_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update site: {str(e)}"
        )

@app.delete("/sites/{location_id}", response_model=StandardResponse, tags=["Site Management"])
async def delete_site(location_id: int):
    """Delete a site by its ID"""
    try:
        # Check if site exists
        existing_site = site_manager.get_site_by_id(location_id)
        if not existing_site:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Site with ID {location_id} not found"
            )
        
        # Delete site
        result = site_manager.delete_site(location_id)
        
        logger.info(f"Site {location_id} deleted successfully")
        
        return StandardResponse(
            success=True,
            message=f"Site {location_id} ('{existing_site.get('location_name')}') deleted successfully",
            data=result
        )
        
    except HTTPException:
        raise
    except PulseProAPIException as e:
        logger.error(f"Failed to delete site {location_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete site: {str(e)}"
        )

# ============================================
# UTILITY ENDPOINTS
# ============================================

@app.get("/health", tags=["Utility"])
async def health_check():
    """System health check"""
    health_status = {
        "status": "healthy",
        "service": "Site Management API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "site_manager": "available"
        }
    }
    
    return health_status

@app.get("/", tags=["Utility"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Site Management API",
        "version": "1.0.0",
        "description": "Simple site management system",
        "features": [
            "🏢 Complete site CRUD operations",
            "📊 Data validation and error handling",
            "🔍 Site search and filtering"
        ],
        "endpoints": {
            "sites": ["GET /sites", "POST /sites", "POST /sites/quick-create", "GET /sites/{id}", "PUT /sites/{id}", "DELETE /sites/{id}"],
            "utility": ["GET /health", "GET /docs"]
        },
        "quick_start": {
            "1": "POST /sites/quick-create - Create a site with just a name",
            "2": "POST /sites - Create a detailed site",
            "3": "GET /sites - View all sites",
            "4": "Visit /docs for interactive documentation"
        }
    }

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "details": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================
# STARTUP
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Site Management API")
    print("=" * 40)
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🏠 Root Info: http://localhost:8000/")
    print("=" * 40)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )