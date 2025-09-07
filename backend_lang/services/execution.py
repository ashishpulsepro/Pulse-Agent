from typing import Dict, Any
from .managers import AuthenticationManager, SiteManager, UserManager, PermissionManager, TemplateManager
import logging

logger = logging.getLogger(__name__)

# Lazy singletons
_auth = None
_site = None
_user = None
_perm = None
_tmpl = None

def ensure_managers():
    global _auth, _site, _user, _perm, _tmpl
    if _auth is None:
        _auth = AuthenticationManager()
        _site = SiteManager(_auth)
        _user = UserManager(_auth)
        _perm = PermissionManager(_auth)
        _tmpl = TemplateManager(_auth)
    return _site, _user, _perm, _tmpl


def execute_operation(operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
    site, user, perm, tmpl = ensure_managers()

    logger.info(f"Executing operation={operation} data={data}")

    if operation == 'CREATE_SITE':
        name = data['location_name']
        return site.create_site_by_name_only(name)

    if operation == 'VIEW_SITES':
        return site.get_all_sites()

    if operation == 'CREATE_USER':
        ps_name = data['permission_set']
        ps_id = perm.get_permission_set_id_by_name(ps_name)
        if not ps_id:
            return {"error": f"Permission set '{ps_name}' not found"}
        return user.create_user(data['first_name'], data['last_name'], data['email'], [ps_id])

    if operation == 'ASSIGN_USERS_TO_SITE':
        all_sites = site.get_all_sites().get('locations', [])
        target = next((s for s in all_sites if s.get('location_name','').lower() == data['location_name'].lower()), None)
        if not target:
            return {"error": f"Site '{data['location_name']}' not found"}
        site_id = target.get('id')
        # get site users (added/not added) then pick IDs by name (assuming username field)
        site_users = site.get_site_users(site_id).get('users', [])
        matched_ids = [u.get('id') for u in site_users if u.get('username','').lower() in [n.lower() for n in data['user_list']]]
        if not matched_ids:
            return {"error": "No specified users found for assignment"}
        return site.assign_users_to_site(site_id, matched_ids)

    if operation == 'CREATE_TEMPLATE':
        template_name = data['template_name']
        tmpl_id = tmpl.get_checklist_id_by_name(template_name)
        if not tmpl_id:
            return {"error": f"Checklist source '{template_name}' not found"}
        return tmpl.create_checklist(tmpl_id, template_name)

    logger.warning(f"Unsupported operation requested: {operation}")
    return {"error": f"Unsupported operation {operation}"}
