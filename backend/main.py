"""
FastAPI Service Layer for PulsePro Site Management
RESTful API endpoints for the AI agent and direct site management
"""

from fastapi import FastAPI, HTTPException, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

# Import our custom classes (assuming they're in site_manager.py)
# from site_manager import SiteManager, AuthenticationManager, SiteData, AIAgentProcessor, initialize_system

class AuthRequest(BaseModel):
    refresh_token: str = Field(..., description="JWT refresh token")

class SiteCreateRequest(BaseModel):
    location_name: str = Field(..., description="Name of the location")
    address_field1: str = Field(..., description="Primary address")
    address_field2: Optional[str] = Field("", description="Secondary address")
    country_id: int = Field(1, description="Country ID")
    state_id: int = Field(..., description="State ID")
    city_id: int = Field(..., description="City ID")
    pincode: Optional[str] = Field("", description="Postal code")
    mobile: Optional[str] = Field("", description="Mobile number")
    location_number: Optional[str] = Field("", description="Location number")
    location_code: Optional[str] = Field("", description="Location code")
    to_email: Optional[str] = Field("", description="Primary email")
    cc_email: Optional[str] = Field("", description="CC email")
    reporting_timezone: str = Field("UTC", description="Timezone for reporting")
    geo_fencing_enabled: bool = Field(False, description="Enable geo-fencing")
    geo_fencing_distance: int = Field(0, description="Geo-fencing distance")
    lat: float = Field(0.0, description="Latitude")
    lng: float = Field(0.0, description="Longitude")
    map_link: Optional[str] = Field("", description="Map link")
    has_custom_field: bool = Field(False, description="Has custom fields")
    is_schedule_active: bool = Field(False, description="Schedule active")

class SiteUpdateRequest(SiteCreateRequest):
    pass

class UserAssignmentRequest(BaseModel):
    location_id: int = Field(..., description="Location ID")
    user_ids: List[int] = Field(..., description="List of user IDs to assign")

class UserUnassignmentRequest(BaseModel):
    mapped_location_ids: List[int] = Field(..., description="List of mapped location IDs to remove")

class AIRequest(BaseModel):
    query: str = Field(..., description="Natural language query for the AI agent")

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Global variables to store system components
auth_manager = None
site_manager = None
ai_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting PulsePro Site Management API...")
    yield
    # Shutdown
    print("Shutting down PulsePro Site Management API...")

app = FastAPI(
    title="PulsePro Site Management API",
    description="AI-powered site management system with CRUD operations and user assignments",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_auth_manager():
    """Dependency to get authenticated site manager"""
    global auth_manager, site_manager, ai_agent
    
    if not auth_manager or not site_manager or not ai_agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="System not initialized. Please authenticate first."
        )
    return auth_manager, site_manager, ai_agent

@app.post("/auth/initialize", response_model=StandardResponse)
async def initialize_system_endpoint(auth_request: AuthRequest):
    """Initialize the system with refresh token"""
    global auth_manager, site_manager, ai_agent
    
    try:
        # Initialize system components
        ai_agent = initialize_system(auth_request.refresh_token)
        auth_manager = ai_agent.site_manager.auth_manager
        site_manager = ai_agent.site_manager
        
        # Test the authentication by refreshing token
        auth_manager.refresh_access_token()
        
        return StandardResponse(
            success=True,
            message="System initialized successfully"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unassign users: {str(e)}"
        )

# AI Agent Endpoints

@app.post("/ai/query", response_model=StandardResponse)
async def process_ai_query(ai_request: AIRequest, deps=Depends(get_auth_manager)):
    """Process natural language query through AI agent"""
    _, _, ai_agent = deps
    
    try:
        result = ai_agent.process_request(ai_request.query)
        return StandardResponse(
            success=result.get('success', True),
            message=result.get('message', 'Query processed'),
            data=result.get('data')
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process AI query: {str(e)}"
        )

@app.get("/ai/help", response_model=StandardResponse)
async def get_ai_help():
    """Get help information for AI agent commands"""
    help_info = {
        "available_commands": {
            "create": "Create a new site - 'create a site', 'add new location'",
            "update": "Update existing site - 'update site with id 123', 'edit location'",
            "delete": "Delete a site - 'delete site 123', 'remove location'",
            "show": "Show sites - 'show all sites', 'list locations', 'get all sites'",
            "assign": "Assign users - 'assign users to site', 'add users to location'",
            "unassign": "Unassign users - 'remove users from site', 'unassign users'"
        },
        "examples": [
            "show all sites",
            "create a new site in Mumbai",
            "delete site with id 123",
            "assign user 456 to site 123",
            "list all locations in India"
        ]
    }
    
    return StandardResponse(
        success=True,
        message="AI Agent help information",
        data=help_info
    )

# Utility Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PulsePro Site Management API"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "PulsePro Site Management API",
        "version": "1.0.0",
        "description": "AI-powered site management system with CRUD operations and user assignments",
        "endpoints": {
            "auth": "/auth/initialize, /auth/refresh",
            "sites": "/sites (GET, POST), /sites/{id} (GET, PUT, DELETE)",
            "users": "/sites/{id}/users, /sites/assign-users, /sites/unassign-users",
            "ai": "/ai/query, /ai/help",
            "utility": "/health, /docs"
        }
    }

# Advanced Site Operations

@app.get("/sites/search", response_model=StandardResponse)
async def search_sites(
    name: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    deps=Depends(get_auth_manager)
):
    """Search sites with filters"""
    _, site_manager, _ = deps
    
    try:
        all_sites = site_manager.get_all_sites()
        sites = all_sites.get('locations', [])
        
        # Apply filters
        filtered_sites = sites
        
        if name:
            filtered_sites = [s for s in filtered_sites if name.lower() in s.get('location_name', '').lower()]
        
        if city:
            filtered_sites = [s for s in filtered_sites if city.lower() in s.get('city', '').lower()]
        
        if state:
            filtered_sites = [s for s in filtered_sites if state.lower() in s.get('state', '').lower()]
        
        if country:
            filtered_sites = [s for s in filtered_sites if country.lower() in s.get('country', '').lower()]
        
        return StandardResponse(
            success=True,
            message=f"Found {len(filtered_sites)} matching sites",
            data={
                "sites": filtered_sites,
                "total_count": len(filtered_sites),
                "filters_applied": {
                    "name": name,
                    "city": city,
                    "state": state,
                    "country": country
                }
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search sites: {str(e)}"
        )

@app.get("/analytics/sites-summary", response_model=StandardResponse)
async def get_sites_summary(deps=Depends(get_auth_manager)):
    """Get summary analytics for sites"""
    _, site_manager, _ = deps
    
    try:
        all_sites = site_manager.get_all_sites()
        sites = all_sites.get('locations', [])
        
        # Calculate analytics
        total_sites = len(sites)
        sites_by_country = {}
        sites_by_state = {}
        sites_with_geo_fencing = 0
        sites_with_mobile = 0
        sites_with_email = 0
        
        for site in sites:
            # Country distribution
            country = site.get('country', 'Unknown')
            sites_by_country[country] = sites_by_country.get(country, 0) + 1
            
            # State distribution
            state = site.get('state', 'Unknown')
            sites_by_state[state] = sites_by_state.get(state, 0) + 1
            
            # Feature usage
            if site.get('geo_fencing_enabled'):
                sites_with_geo_fencing += 1
            
            if site.get('mobile'):
                sites_with_mobile += 1
            
            if site.get('to_email'):
                sites_with_email += 1
        
        summary = {
            "total_sites": total_sites,
            "distribution": {
                "by_country": sites_by_country,
                "by_state": sites_by_state
            },
            "features": {
                "geo_fencing_enabled": sites_with_geo_fencing,
                "sites_with_mobile": sites_with_mobile,
                "sites_with_email": sites_with_email,
                "completion_rate": {
                    "mobile": round((sites_with_mobile / total_sites) * 100, 2) if total_sites > 0 else 0,
                    "email": round((sites_with_email / total_sites) * 100, 2) if total_sites > 0 else 0,
                    "geo_fencing": round((sites_with_geo_fencing / total_sites) * 100, 2) if total_sites > 0 else 0
                }
            }
        }
        
        return StandardResponse(
            success=True,
            message="Sites summary generated successfully",
            data=summary
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )

# Batch Operations

@app.post("/sites/batch-create", response_model=StandardResponse)
async def batch_create_sites(sites_data: List[SiteCreateRequest], deps=Depends(get_auth_manager)):
    """Create multiple sites in batch"""
    _, site_manager, _ = deps
    
    results = []
    errors = []
    
    for idx, site_request in enumerate(sites_data):
        try:
            site_data = SiteData(**site_request.dict())
            result = site_manager.create_site(site_data)
            results.append({
                "index": idx,
                "site_name": site_request.location_name,
                "success": True,
                "data": result
            })
        except Exception as e:
            errors.append({
                "index": idx,
                "site_name": site_request.location_name,
                "error": str(e)
            })
    
    return StandardResponse(
        success=len(errors) == 0,
        message=f"Batch operation completed. {len(results)} successful, {len(errors)} failed",
        data={
            "successful": results,
            "failed": errors,
            "summary": {
                "total_attempted": len(sites_data),
                "successful_count": len(results),
                "failed_count": len(errors)
            }
        }
    )

@app.delete("/sites/batch-delete", response_model=StandardResponse)
async def batch_delete_sites(location_ids: List[int] = Body(...), deps=Depends(get_auth_manager)):
    """Delete multiple sites in batch"""
    _, site_manager, _ = deps
    
    results = []
    errors = []
    
    for location_id in location_ids:
        try:
            result = site_manager.delete_site(location_id)
            results.append({
                "location_id": location_id,
                "success": True,
                "data": result
            })
        except Exception as e:
            errors.append({
                "location_id": location_id,
                "error": str(e)
            })
    
    return StandardResponse(
        success=len(errors) == 0,
        message=f"Batch deletion completed. {len(results)} successful, {len(errors)} failed",
        data={
            "successful": results,
            "failed": errors,
            "summary": {
                "total_attempted": len(location_ids),
                "successful_count": len(results),
                "failed_count": len(errors)
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )Exception(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )

@app.post("/auth/refresh", response_model=StandardResponse)
async def refresh_token(deps=Depends(get_auth_manager)):
    """Refresh access token"""
    auth_manager, _, _ = deps
    
    try:
        access_token = auth_manager.refresh_access_token()
        return StandardResponse(
            success=True,
            message="Token refreshed successfully",
            data={"access_token": access_token}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {str(e)}"
        )

# Site CRUD Operations

@app.post("/sites", response_model=StandardResponse)
async def create_site(site_request: SiteCreateRequest, deps=Depends(get_auth_manager)):
    """Create a new site"""
    _, site_manager, _ = deps
    
    try:
        # Convert request to SiteData
        site_data = SiteData(**site_request.dict())
        result = site_manager.create_site(site_data)
        
        return StandardResponse(
            success=True,
            message="Site created successfully",
            data=result
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create site: {str(e)}"
        )

@app.get("/sites", response_model=StandardResponse)
async def get_all_sites(deps=Depends(get_auth_manager)):
    """Get all sites"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.get_all_sites()
        return StandardResponse(
            success=True,
            message="Sites retrieved successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sites: {str(e)}"
        )

@app.get("/sites/{location_id}", response_model=StandardResponse)
async def get_site_by_id(location_id: int, deps=Depends(get_auth_manager)):
    """Get a specific site by ID"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.get_site_by_id(location_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Site with ID {location_id} not found"
            )
        
        return StandardResponse(
            success=True,
            message="Site retrieved successfully",
            data=result
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve site: {str(e)}"
        )

@app.put("/sites/{location_id}", response_model=StandardResponse)
async def update_site(location_id: int, site_request: SiteUpdateRequest, deps=Depends(get_auth_manager)):
    """Update an existing site"""
    _, site_manager, _ = deps
    
    try:
        # Convert request to SiteData
        site_data = SiteData(**site_request.dict())
        result = site_manager.update_site(location_id, site_data)
        
        return StandardResponse(
            success=True,
            message="Site updated successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update site: {str(e)}"
        )

@app.delete("/sites/{location_id}", response_model=StandardResponse)
async def delete_site(location_id: int, deps=Depends(get_auth_manager)):
    """Delete a site"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.delete_site(location_id)
        return StandardResponse(
            success=True,
            message="Site deleted successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete site: {str(e)}"
        )

# User Assignment Operations

@app.get("/sites/{location_id}/users", response_model=StandardResponse)
async def get_site_users(location_id: int, search_keyword: str = "", deps=Depends(get_auth_manager)):
    """Get all users available for assignment to a site"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.get_site_users(location_id, search_keyword)
        return StandardResponse(
            success=True,
            message="Users retrieved successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve users: {str(e)}"
        )

@app.post("/sites/assign-users", response_model=StandardResponse)
async def assign_users_to_site(assignment_request: UserAssignmentRequest, deps=Depends(get_auth_manager)):
    """Assign multiple users to a site"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.assign_users_to_site(
            assignment_request.location_id,
            assignment_request.user_ids
        )
        
        return StandardResponse(
            success=True,
            message="Users assigned successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign users: {str(e)}"
        )

@app.post("/sites/unassign-users", response_model=StandardResponse)
async def unassign_users_from_site(unassignment_request: UserUnassignmentRequest, deps=Depends(get_auth_manager)):
    """Unassign users from a site"""
    _, site_manager, _ = deps
    
    try:
        result = site_manager.unassign_users_from_site(unassignment_request.mapped_location_ids)
        
        return StandardResponse(
            success=True,
            message="Users unassigned successfully",
            data=result
        )
    
    except Exception as e:
        raise HTTP