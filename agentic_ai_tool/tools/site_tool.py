"""
Site Tool
Handles all site-related operations using the SiteManager
"""

import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add backend directory to path to import site_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from site_manager import SiteManager, SiteData, PulseProAPIException
except ImportError:
    # Create mock classes for development
    class SiteManager:
        def __init__(self, auth_manager): pass
        def create_site_by_name_only(self, name): return {'id': 1, 'name': name}
        def create_site(self, data): return {'id': 1}
        def get_all_sites(self): return {'locations': []}
        def update_site(self, id, data): return {'id': id}
        def delete_site(self, id): return {'deleted': True}
    
    class SiteData:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    PulseProAPIException = Exception

# Import relative modules
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from action_executor import ExecutionResult

logger = logging.getLogger(__name__)

class SiteTool:
    """
    Tool for handling site-related operations
    """
    
    def __init__(self, auth_manager=None):
        """Initialize site tool with authentication manager"""
        self.auth_manager = auth_manager
        if auth_manager:
            self.site_manager = SiteManager(auth_manager)
        else:
            self.site_manager = None
            logger.warning("SiteTool initialized without auth_manager")
    
    async def create_site(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Create a new site"""
        try:
            if not self.site_manager:
                return ExecutionResult(
                    success=False,
                    message="Site management is not configured properly",
                    error="No authentication manager available"
                )
            
            site_name = parameters.get('site_name')
            if not site_name:
                return ExecutionResult(
                    success=False,
                    message="Site name is required to create a site",
                    error="Missing site_name parameter"
                )
            
            # Check if this is a simple creation (name only) or detailed creation
            if len(parameters) == 1 and 'site_name' in parameters:
                # Simple creation with name only
                result = self.site_manager.create_site_by_name_only(site_name)
                
                return ExecutionResult(
                    success=True,
                    message=f"Successfully created site '{site_name}'",
                    data=result,
                    actions_taken=[f"created_site:{site_name}"]
                )
            else:
                # Detailed creation
                site_data = self._build_site_data(parameters)
                result = self.site_manager.create_site(site_data)
                
                return ExecutionResult(
                    success=True,
                    message=f"Successfully created detailed site '{site_name}'",
                    data=result,
                    actions_taken=[f"created_detailed_site:{site_name}"]
                )
                
        except PulseProAPIException as e:
            logger.error(f"API error creating site: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to create site '{site_name}': {str(e)}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error creating site: {e}")
            return ExecutionResult(
                success=False,
                message=f"An unexpected error occurred while creating the site",
                error=str(e)
            )
    
    async def list_sites(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """List all sites"""
        try:
            if not self.site_manager:
                return ExecutionResult(
                    success=False,
                    message="Site management is not configured properly",
                    error="No authentication manager available"
                )
            
            result = self.site_manager.get_all_sites()
            sites = result.get('locations', [])
            
            return ExecutionResult(
                success=True,
                message=f"Retrieved {len(sites)} sites",
                data=result,
                actions_taken=["listed_sites"]
            )
            
        except PulseProAPIException as e:
            logger.error(f"API error listing sites: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to retrieve sites: {str(e)}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error listing sites: {e}")
            return ExecutionResult(
                success=False,
                message="An unexpected error occurred while retrieving sites",
                error=str(e)
            )
    
    async def get_site(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Get specific site details"""
        try:
            if not self.site_manager:
                return ExecutionResult(
                    success=False,
                    message="Site management is not configured properly",
                    error="No authentication manager available"
                )
            
            site_name = parameters.get('site_name')
            if not site_name:
                return ExecutionResult(
                    success=False,
                    message="Site name is required",
                    error="Missing site_name parameter"
                )
            
            # First get all sites, then find the matching one
            all_sites = self.site_manager.get_all_sites()
            sites = all_sites.get('locations', [])
            
            # Find site by name (case-insensitive)
            matching_site = None
            for site in sites:
                if site.get('location_name', '').lower() == site_name.lower():
                    matching_site = site
                    break
            
            if not matching_site:
                return ExecutionResult(
                    success=False,
                    message=f"Site '{site_name}' not found",
                    error="Site not found"
                )
            
            return ExecutionResult(
                success=True,
                message=f"Retrieved details for site '{site_name}'",
                data=matching_site,
                actions_taken=[f"retrieved_site:{site_name}"]
            )
            
        except PulseProAPIException as e:
            logger.error(f"API error getting site: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to retrieve site '{site_name}': {str(e)}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error getting site: {e}")
            return ExecutionResult(
                success=False,
                message="An unexpected error occurred while retrieving the site",
                error=str(e)
            )
    
    async def update_site(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Update an existing site"""
        try:
            if not self.site_manager:
                return ExecutionResult(
                    success=False,
                    message="Site management is not configured properly",
                    error="No authentication manager available"
                )
            
            site_name = parameters.get('site_name')
            if not site_name:
                return ExecutionResult(
                    success=False,
                    message="Site name is required for updates",
                    error="Missing site_name parameter"
                )
            
            # First find the site to get its ID
            all_sites = self.site_manager.get_all_sites()
            sites = all_sites.get('locations', [])
            
            target_site = None
            for site in sites:
                if site.get('location_name', '').lower() == site_name.lower():
                    target_site = site
                    break
            
            if not target_site:
                return ExecutionResult(
                    success=False,
                    message=f"Site '{site_name}' not found",
                    error="Site not found"
                )
            
            # Build updated site data
            site_data = self._build_site_data_for_update(target_site, parameters)
            
            # Update the site
            result = self.site_manager.update_site(target_site['id'], site_data)
            
            return ExecutionResult(
                success=True,
                message=f"Successfully updated site '{site_name}'",
                data=result,
                actions_taken=[f"updated_site:{site_name}"]
            )
            
        except PulseProAPIException as e:
            logger.error(f"API error updating site: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to update site '{site_name}': {str(e)}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error updating site: {e}")
            return ExecutionResult(
                success=False,
                message="An unexpected error occurred while updating the site",
                error=str(e)
            )
    
    async def delete_site(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """Delete a site"""
        try:
            if not self.site_manager:
                return ExecutionResult(
                    success=False,
                    message="Site management is not configured properly",
                    error="No authentication manager available"
                )
            
            site_name = parameters.get('site_name')
            if not site_name:
                return ExecutionResult(
                    success=False,
                    message="Site name is required for deletion",
                    error="Missing site_name parameter"
                )
            
            # First find the site to get its ID
            all_sites = self.site_manager.get_all_sites()
            sites = all_sites.get('locations', [])
            
            target_site = None
            for site in sites:
                if site.get('location_name', '').lower() == site_name.lower():
                    target_site = site
                    break
            
            if not target_site:
                return ExecutionResult(
                    success=False,
                    message=f"Site '{site_name}' not found",
                    error="Site not found"
                )
            
            # Delete the site
            result = self.site_manager.delete_site(target_site['id'])
            
            return ExecutionResult(
                success=True,
                message=f"Successfully deleted site '{site_name}'",
                data=result,
                actions_taken=[f"deleted_site:{site_name}"]
            )
            
        except PulseProAPIException as e:
            logger.error(f"API error deleting site: {e}")
            return ExecutionResult(
                success=False,
                message=f"Failed to delete site '{site_name}': {str(e)}",
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error deleting site: {e}")
            return ExecutionResult(
                success=False,
                message="An unexpected error occurred while deleting the site",
                error=str(e)
            )
    
    def _build_site_data(self, parameters: Dict[str, Any]) -> SiteData:
        """Build SiteData object from parameters"""
        site_data = SiteData(
            location_name=parameters.get('site_name', ''),
            address_field1=parameters.get('address', 'Default Address'),
            address_field2=parameters.get('address2', ''),
            pincode=parameters.get('pincode', ''),
            mobile=parameters.get('mobile', ''),
            location_number=parameters.get('location_number', ''),
            location_code=parameters.get('location_code', ''),
            to_email=parameters.get('email', ''),
            cc_email=parameters.get('cc_email', ''),
            reporting_timezone=parameters.get('timezone', 'UTC'),
        )
        
        # Add location-based defaults if location is provided
        location = parameters.get('location', '').lower()
        if location:
            site_data.address_field1 = f"Address in {parameters['location']}"
        
        return site_data
    
    def _build_site_data_for_update(self, existing_site: Dict[str, Any], 
                                  parameters: Dict[str, Any]) -> SiteData:
        """Build SiteData object for updates, preserving existing values"""
        
        # Start with existing data
        site_data = SiteData(
            location_name=existing_site.get('location_name', ''),
            address_field1=existing_site.get('address_field1', ''),
            address_field2=existing_site.get('address_field2', ''),
            country_id=existing_site.get('country_id', 1),
            state_id=existing_site.get('state_id', 283),
            city_id=existing_site.get('city_id', 34384),
            pincode=existing_site.get('pincode', ''),
            mobile=existing_site.get('mobile', ''),
            location_number=existing_site.get('location_number', ''),
            location_code=existing_site.get('location_code', ''),
            to_email=existing_site.get('to_email', ''),
            cc_email=existing_site.get('cc_email', ''),
            reporting_timezone=existing_site.get('reporting_timezone', 'UTC'),
            geo_fencing_enabled=existing_site.get('geo_fencing_enabled', False),
            geo_fencing_distance=existing_site.get('geo_fencing_distance', 0),
            lat=existing_site.get('lat', 0.0),
            lng=existing_site.get('lng', 0.0),
            map_link=existing_site.get('map_link', ''),
            has_custom_field=existing_site.get('has_custom_field', False),
            is_schedule_active=existing_site.get('is_schedule_active', False),
        )
        
        # Update with new values from parameters
        if 'address' in parameters:
            site_data.address_field1 = parameters['address']
        if 'address2' in parameters:
            site_data.address_field2 = parameters['address2']
        if 'pincode' in parameters:
            site_data.pincode = parameters['pincode']
        if 'mobile' in parameters:
            site_data.mobile = parameters['mobile']
        if 'email' in parameters:
            site_data.to_email = parameters['email']
        if 'timezone' in parameters:
            site_data.reporting_timezone = parameters['timezone']
        
        return site_data
    
    def get_available_actions(self) -> List[str]:
        """Return list of available actions"""
        return [
            'create_site',
            'list_sites',
            'get_site',
            'update_site',
            'delete_site'
        ]
