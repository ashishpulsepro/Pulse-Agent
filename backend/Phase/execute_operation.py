




from services.Authentication_Service import AuthenticationManager

import logging
import json


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from db.db_services import clear_conversation_from_db



ollama_site_manager = None

def initialize_site_manager(auth_manager):
    """Initialize the Ollama site manager"""
    global ollama_site_manager
    
    try:        
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from services.Site_Service import SiteManager  # Replace with actual import
        
        ollama_site_manager = SiteManager(auth_manager)
        # Test the connection

        
        print("Ollama Site Manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize Ollama Site Manager: {e}")
        ollama_site_manager = None
        return False


def initialize_permission_manager(auth_manager):
    """Initialize the Ollama permission manager"""
    global ollama_permission_manager
    
    try:
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from services.Permission_Service import PermissionManager  # Replace with actual import
        
        ollama_permission_manager = PermissionManager(auth_manager)
        # Test the connection

        
        print("Ollama Permission Manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize Ollama Permission Manager: {e}")
        ollama_permission_manager = None
        return False
    
def initialize_user_manager(auth_manager):
    """Initialize the Ollama user manager"""
    global ollama_user_manager
    
    try:
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from services.User_Service import UserManager  # Replace with actual import
        
        ollama_user_manager = UserManager(auth_manager)
        # Test the connection

        
        print("Ollama User Manager initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize Ollama User Manager: {e}")
        ollama_user_manager = None
        return False

def initialize_template_manager(auth_manager):
    """Initialize the Ollama template manager"""
    global ollama_template_manager

    try:
        # Initialize your site manager here
        # Replace this with your actual SiteManager initialization
        from services.Template_Service import TemplateManager  # Replace with actual import

        ollama_template_manager = TemplateManager(auth_manager)
        # Test the connection

        print("Ollama Template Manager initialized successfully")
        return True

    except Exception as e:
        print(f"Failed to initialize Ollama Template Manager: {e}")
        ollama_template_manager = None
        return False








async def execute_site_operation(operation_data: dict,session_id:str,auth_manager:AuthenticationManager) -> dict:
    """Execute the site operation based on JSON data"""
    
    try:
        initialize_site_manager(auth_manager)
        initialize_user_manager(auth_manager)
        initialize_permission_manager(auth_manager)
        initialize_template_manager(auth_manager)
        print("in execute")
        operation_type = operation_data.get("operation_type")
        data = operation_data.get("data", {})
        print("data: ",data)
        if operation_type == "CREATE_SITE":
            location_name = data.get("location_name")
            print("in CREATE_SITE")
            result = ollama_site_manager.create_site_by_name_only(location_name)
            return {
                "success": True,
                "message": result['message'],
                "data": result
            }
        
        elif operation_type == "DELETE_SITE":
            location_name = data.get("location_name")
            
            # Find and delete site
            all_sites = ollama_site_manager.get_all_sites()
            site_id = None
            for site in all_sites.get('locations', []):
                if site.get('location_name', '').lower() == location_name.lower():
                    site_id = site.get('id')
                    break
            
            if site_id:
                result=ollama_site_manager.delete_site(site_id)    

                if isinstance(result, dict) and not result.get("success", True):
                    return {
                "success": False,
                "message": f"❌ Failed to delete site **{location_name}**: {result.get('message')}",
                "data": {"error": result.get("message"), "site_id": site_id}
                }   

                return {
                    "success": True,
                    "message": f"✅ Site **{location_name}** deleted successfully!",
                    "data": {"deleted": True, "site_id": site_id}
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Site **{location_name}** not found",
                    "data": {"error": "Site not found"}
                }
        
        elif operation_type == "VIEW_SITES":
            result = ollama_site_manager.get_all_sites()
            sites = result.get('locations', [])
            if sites:
                site_list = "\n".join([f"• **{site.get('location_name', 'Unknown')}**,  **{site.get('city')}**,  **{site.get('state')}**" for site in sites])
                message = f"📍 Found {len(sites)} sites:\n{site_list}"
            else:
                message = "📍 No sites found"
            
            return {
                "success": True,
                "message": message,
                "data": result
            }
        
        elif operation_type == "ASSIGN_USERS_TO_SITE":
            location_name = data.get("location_name")
            user_names = data.get("user_list", [])
            
            # Find site ID
            all_sites = ollama_site_manager.get_all_sites()
            site_id = None
            for site in all_sites.get('locations', []):
                if site.get('location_name', '').lower() == location_name.lower():
                    site_id = site.get('id')
                    break

            if not site_id:
                return {
                    "success": False,
                    "message": f"❌ Site **{location_name}** not found",
                    "data": {"error": "Site not found"}
                }
            
            all_users = ollama_site_manager.get_site_users(location_id=site_id)
            user_ids= []
            for user in all_users.get('users', []) :
                if user.get('username', '').lower() in [uname.lower() for uname in user_names]:
                    user_ids.append(user.get('id'))
                    print("user ids: ",user_ids)


            if not user_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified users were found: {', '.join(user_names)}",
                    "data": {"error": "Users not found"}
                }
            
            # Assign users
            result = ollama_site_manager.assign_users_to_site(site_id, user_ids=user_ids)
            return {
                "success": True,
                "message": f"✅ Users assigned to site **{location_name}** successfully!",
                "data": result
            }
        
        elif operation_type == "UNASSIGN_USERS_FROM_SITE":
            location_name = data.get("location_name")
            user_names = data.get("user_list", [])
            
            # Find site ID
            all_sites = ollama_site_manager.get_all_sites()
            site_id = None
            for site in all_sites.get('locations', []):
                if site.get('location_name', '').lower() == location_name.lower():
                    site_id = site.get('id')
                    break

            if not site_id:
                return {
                    "success": False,
                    "message": f"❌ Site **{location_name}** not found",
                    "data": {"error": "Site not found"}
                }
            
            all_users = ollama_site_manager.get_site_users(location_id=site_id)
            user_ids= []
            for user in all_users.get('users', []) :
                if user.get('username', '').lower() in [uname.lower() for uname in user_names] and user.get('mapped'):
                    user_ids.append(user.get('id'))
                    print("user ids: ",user_ids)


            if not user_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified users were found: {', '.join(user_names)}",
                    "data": {"error": "Users not found"}
                }
            
            # Unassign users
            result = ollama_site_manager.unassign_users_from_site(mapped_location_ids=user_ids)
            return {
                "success": True,
                "message": f"✅ Users unassigned from site **{location_name}** successfully!",
                "data": {"user_ids": user_ids}
            }
         
        elif operation_type == "CREATE_USER":
            first_name = data.get("first_name")
            last_name = data.get("last_name")
            email = data.get("email")

            permission_set_id = ollama_permission_manager.get_permission_set_id_by_name(data.get("permission_set"))
            if not permission_set_id:
                return {
                    "success": False,
                    "message": f"❌ Permission set **{data.get('permission_set')}** not found",
                    "data": {"error": "Permission set not found"}
                }
            
            result = ollama_user_manager.create_user(first_name, last_name, email, permission_set_ids=permission_set_id)
            return {
                "success": True,
                "message": result.get("message"),
                "data": result
            }

        elif operation_type == "DELETE_USER":
            full_name = data.get("full_name")
            # first_name, last_name = full_name.split(' ', 1) if ' ' in full_name else (full_name, '')
            
            # Find user ID
            id=ollama_user_manager.get_user_id_by_name(full_name)
            if id:
                result = ollama_user_manager.delete_user(id)
                return {
                    "success": True,
                    "message": f"✅ User **{full_name}** deleted successfully!",
                    "data": {"user_id": id}
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ User **{full_name}** not found",
                    "data": {"error": "User not found"}
                }
            
        elif operation_type == "VIEW_USERS":
            users = ollama_user_manager.get_all_users() or []   # directly get list

            if users:
                user_list = "\n".join([
                    f"• {user.get('name', '')} (ID: {user.get('id', '')})"
                    for user in users
                ])
                message = f"👤 Found {len(users)} users:\n{user_list}"
            else:
                message = "👤 No users found"

            return {
                    "success": True,
                    "message": message,
                    "data": {"users": users}   # wrap list in a dict for consistency
            }

        
        elif operation_type == "VIEW_PERMISSION_SETS":
            print("in permission sets")
            result = ollama_permission_manager.get_all_permission_sets()
            print("result: ",result)
            if result:
                ps_list = "\n".join([f"• **{ps.get('name', 'Unknown')}**" for ps in result])
                message = f"🔑 Found {len(result)} permission sets:\n{ps_list}"
            else:
                message = "🔑 No permission sets found"
            
            return {
                "success": True,
                "message": message,
                "data": {"permission_sets": result} 
            }
        
        elif operation_type in ["ASSIGN_PERMISSION_SET_TO_USER", "UNASSIGN_PERMISSION_SET_FROM_USER"]:
            full_name = data.get("full_name")
            permission_sets = data.get("permission_set", [])
            
            # Find user ID
            user_id = ollama_user_manager.get_user_by_name(full_name)
            if not user_id:
                return {
                    "success": False,
                    "message": f"❌ User **{full_name}** not found",
                    "data": {"error": "User not found"}
                }
            
            # Get permission set IDs
            permission_set_ids = []
            for ps_name in permission_sets:
                ps_id = ollama_permission_manager.get_permission_set_id_by_name(ps_name)
                if ps_id:
                    permission_set_ids.append(ps_id)
            
            if not permission_set_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified permission sets were found: {', '.join(permission_sets)}",
                    "data": {"error": "Permission sets not found"}
                }
            
            if operation_type == "ASSIGN_PERMISSION_SET_TO_USER":
                result = ollama_permission_manager.assign_permission_sets_to_user(user_id, permission_set_ids)
                action = "assigned to"
            else:
                result = ollama_permission_manager.unassign_permission_sets_from_user(user_id, permission_set_ids)
                action = "unassigned from"
            
            return {
                "success": True,
                "message": f"✅ Permission sets **{action}** user **{full_name}** successfully!",
                "data": {"permission_set_ids": permission_set_ids}
            }

        elif operation_type == "SHOW_ALL_TEMPLATES":
            templates = ollama_template_manager.get_all_templates() or []   # directly get list

            if templates:
                template_list = "\n".join([
                    f"• {template.get('name', '')} (By : {template.get('created_by', '')})"
                    for template in templates
                ])
                message = f"📄 Found {len(templates)} templates:\n{template_list}"
            else:
                message = "📄 No templates found"

            return {
                "success": True,
                "message": message,
                "data": {"templates": templates}   # wrap list in a dict for consistency
            }

        elif operation_type == "DELETE_TEMPLATE":
            template_name = data.get("template_name")

            # Find template ID
            template_id = ollama_template_manager.get_template_id_by_name(template_name)
            print("Template ID: ", template_id)
            if not template_id:
                return {
                    "success": False,
                    "message": f"❌ Template **{template_name}** not found",
                    "data": {"error": "Template not found"}
                }

            result = ollama_template_manager.delete_template(template_id)
            print("Delete Template Result: ", result)
            return {
                "success": True,
                "message": f"✅ Template **{template_name}** deleted successfully!",
                "data": {"template_id": template_id}
            }
        
        elif operation_type == "ASSIGN_TEMPLATE_TO_USER":
            full_name = data.get("user_name")
            template_names = data.get("template_name", [])

            # Find user ID
            user_id = ollama_user_manager.get_user_by_name(full_name)
            print("user ID: ", user_id)
            if not user_id:
                return {
                    "success": False,
                    "message": f"❌ User **{full_name}** not found",
                    "data": {"error": "User not found"}
                }

            # Get template IDs
            template_ids = []
            for template_name in template_names:
                template_id = ollama_template_manager.get_assign_user_template_id_by_name(template_name,user_id)
                if template_id:
                    template_ids.append(template_id)
            print("template ids: ", template_ids)
            if not template_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified templates were found: {', '.join(template_names)}",
                    "data": {"error": "Templates not found"}
                }

            result = ollama_template_manager.assign_templates_to_user(user_id, template_ids)
            return {
                "success": True,
                "message": f"✅ Templates assigned to user **{full_name}** successfully!",
                "data": {"template_ids": template_ids}
            }

        elif operation_type == "UNASSIGN_TEMPLATE_FROM_USER":
            full_name = data.get("user_name")
            template_names = data.get("template_name", [])

            # Find user ID
            user_id = ollama_user_manager.get_user_by_name(full_name)
            if not user_id:
                return {
                    "success": False,
                    "message": f"❌ User **{full_name}** not found",
                    "data": {"error": "User not found"}
                }

            # Get template IDs
            template_ids = []
            for template_name in template_names:
                template_id = ollama_template_manager.get_assign_user_template_id_by_name(template_name,user_id)
                if template_id:
                    template_ids.append(template_id)

            if not template_ids:
                return {
                    "success": False,
                    "message": f"❌ None of the specified templates were found: {', '.join(template_names)}",
                    "data": {"error": "Templates not found"}
                }

            result = ollama_template_manager.unassign_templates_from_user(user_id, template_ids)
            return {
                "success": True,
                "message": f"✅ Templates unassigned from user **{full_name}** successfully!",
                "data": {"template_ids": template_ids}
            }
        
        elif operation_type == "CREATE_TEMPLATE":
            template_name = data.get("template_name")
            template_id= ollama_template_manager.get_checklist_id_by_name(template_name)

            if not template_id:
                return {
                    "success": False,
                    "message": f"❌ Template **{template_name}** does not exists. PLease select the right template name",
                    "data": {"error": "Template already exists"}
                }
            
            result=ollama_template_manager.create_checklist(template_id,template_name)
            return{
                "success": True,
                "message": result['message'],
                "data": result
            }
        
        elif operation_type == "AUTOMATE_CUSTOMER_ACCESS_SETTING":
            previous_access = ollama_user_manager.get_customer_access_settings()
            print("previous access: ", previous_access)

            print("accessToAllSite",field_exists(data,"accessToAllSite"))
            print("accessToAllChecklist",field_exists(data,"accessToAllChecklist"))

            if(field_exists(data,"accessToAllSite") and field_exists(data,"accessToAllChecklist")):
                print("1st Phase")
                result=ollama_user_manager.update_customer_setting(accessToAllSite=data.get("accessToAllSite"), accessToAllChecklist=data.get("accessToAllChecklist"))
                return{
                "success": True,
                "message": f"✅ Operation on location and checklist was successful !",
                "data": result
            }
            elif(field_exists(data,"accessToAllSite")):
                print("2nd Phase")
                result= ollama_user_manager.update_customer_setting(accessToAllSite=data.get("accessToAllSite"), accessToAllChecklist=previous_access.get("accessToAllChecklist"))
                return{
                "success": True,
                "message": f"✅ Operation on location was successful !",
                "data": result
            }
            elif(field_exists(data,"accessToAllChecklist")):
                print("3rd phase")
                result= ollama_user_manager.update_customer_setting(accessToAllSite=previous_access.get("accessToAllSite"), accessToAllChecklist=data.get("accessToAllChecklist"))
                return{
                "success": True,
                "message": f"✅ Operation on checklist was successful !",
                "data": result
            }

            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to perform it, try later",
                    "data": {"error": "Some internal error"}
                }
            
        elif operation_type=="VIEW_ALL_GROUPS":
            all_groups_list= ollama_user_manager.get_all_groups()
            groups_formatted="\n".join(
                f"{idx+1}. **{group['name']}**"
                for idx,group in enumerate(all_groups_list)
                )if all_groups_list else "Not Available"

            message = f"📍 Found {len(all_groups_list)} Groups:\n{groups_formatted}"

            return{
                "success":True,
                "message": message,
                "data": {"groups_list": groups_formatted}
            }
        
        elif operation_type=="DELETE_A_GROUP":
            group_name=data.get("group_name")

            group_id=ollama_user_manager.get_group_id_by_name(group_name)

            if not group_id:
                return{
                    "success":False,
                    "message":f"❌ Group **{group_name}** not found",
                    "data":{"error":"Group not found"}
                }
            print("group_id: ", group_id)
            result=ollama_user_manager.delete_group(group_id)

            return{
                "success":True,
                "message":f"✅  Group **{group_name}** deleted successfully",
                "data":{"group_id": group_id}

            }
        
        elif operation_type=="CREATE_A_GROUP":
            group_name=data.get("group_name")
            print("group name: ",group_name)


            result=ollama_user_manager.create_a_group(group_name)
            print("result : ", result)

            return{
                "success":True,
                "message":f"✅  Group **{group_name}** created successfully",
                "data":{"group_id": group_name}

            }

        elif operation_type=="ADD_USER_TO_GROUP":
            group_name=data.get("group_name")
            user_names=data.get("user_names")

            group_id=ollama_user_manager.get_group_id_by_name(group_name)
            userIds=[]

            for user_name in user_names:
                userIds.append(ollama_user_manager.get_user_by_name(user_name))

            if not userIds:
                return{
                    "success":False,
                    "message":f"User not found",
                    "data":{"error": "user not found"}
                }
            if not group_id:
                return{
                    "success":False,
                    "message":f"Group not found",
                    "data":{"error":"group not found"}
                }

            result=ollama_user_manager.add_multiple_user_to_group(userIds=userIds,groupId=group_id)
            return{
                "success":True,
                "message":f"✅  Users **{user_names}** assigned successfully to **{group_name}**",
                "data":{"group_name": group_name}

            }
        
        elif operation_type=="REMOVE_USER_FROM_GROUP":
            group_name=data.get("group_name")
            user_names=data.get("user_names")
            print("user_names: ",user_names)

            group_id=ollama_user_manager.get_group_id_by_name(group_name)
            print("group id: " ,group_id)
            userIds=[]

            result_user_list=ollama_user_manager.get_users_added_to_group(group_id)
            print("result_user_list : ", result_user_list)
            for user in result_user_list:
                if user_names and user.get("member_name") in user_names: 
                    userIds.append(user.get("id")) 
                
            print("user ids: ",userIds )
            if not userIds:
                return{
                    "success":False,
                    "message":f"User not found",
                    "data":{"error": "user not found"}
                }
            if not group_id:
                return{
                    "success":False,
                    "message":f"Group not found",
                    "data":{"error":"group not found"}
                }

            result=ollama_user_manager.delete_multiple_group_user(group_user_ids=userIds)
            return{
                "success":True,
                "message":f"✅  Users **{user_names}** removed successfully from **{group_name}**",
                "data":{"group_name": group_name}

            }            

        elif operation_type=="SHOW_USERS_ADDED_TO_GROUP":
            group_name=data.get("group_name")
            group_id=ollama_user_manager.get_group_id_by_name(group_name)

            result=ollama_user_manager.get_users_added_to_group(groupId=group_id)

            result_formatted="\n".join(
            f"{idx+1}.{res['member_name']}"
            for idx,res in enumerate(result)
            )

            return{
                "success":True,
                "message":f"Found {len(result)} assigned users for the group {group_name} \n {result_formatted}",
                "data":{"group name":group_name }
            }

        elif operation_type=="SHOW_USERS_ADDED_NOT_TO_GROUP":
            group_name=data.get("group_name")
            group_id=ollama_user_manager.get_group_id_by_name(group_name)

            result=ollama_user_manager.get_all_members_not_added_to_group(group_id=group_id)

            result_formatted="\n".join(
            f"{idx+1}.{res['name']}"
            for idx,res in enumerate(result)
            )

            return{
                "success":True,
                "message":f"Found {len(result)} non assigned users for the group {group_name} \n {result_formatted}",
                "data":{"group name":group_name }
            }
        
        elif operation_type=="DETAIL_SPECIFIC_USER":
            all_users=ollama_user_manager.get_all_users()
            print("all_users: ", all_users)
            selected_user=data.get("user_name")
            print("data: ",data)

            if not selected_user:
                return{
                "success": False,
                "message": f"❌ Failed operation: {operation_type}, Unknown user",
                "data": {"error": "Failed operation"}
                }

            for user in all_users:
                if user.get("name").lower()==selected_user.lower():
                    email=user.get("email")
                    permission=user.get("permission_set")
                    id=user.get("id")
                    name=user.get("name")
                    

            return{
                "success":True,
                "message":f" Here is the detail of the User:\n User : **{name}** (ID: **{id}**)\n Email: **{email}**\n Permissions: **{permission}**\n ",
                "data":{"user_id": id}
            }

        elif operation_type=="DETAIL_SPECIFIC_SITE":
            all_sites=ollama_site_manager.get_all_sites().get("locations")
            print("all_sites: ", all_sites)
            selected_site=data.get("site_name")
            print("data: ",data)

            if not selected_site:
                return{
                "success": False,
                "message": f"❌ Failed operation: {operation_type}, Unknown site",
                "data": {"error": "Failed operation"}
                }
            site_name=selected_site
            id=0
            city="UNKNOWN"
            state="UNKNOWN"
            country="UNKNOWN"
            geo_fencing_enabled=False
            geo_fencing_distance=0.0
            custom_field_data=[]
            full_address="UNKNOWN"
            pincode=826001
            mobile=1234567891

            for site in all_sites:
                if site.get("location_name").lower()==selected_site.lower():
                    city=site.get("city")
                    state=site.get("state")
                    id=site.get("id")
                    site_name=site.get("location_name")
                    full_address=site.get("full_address")
                    pincode=site.get("pincode") 
                    mobile=site.get("mobile")
                    country= site.get("country")
                    geo_fencing_enabled=site.get("geo_fencing_enabled")
                    geo_fencing_distance=site.get("geo_fencing_distance")
                    custom_field_data=site.get("custom_field_meta")

                    

            return{
                "success":True,
                "message":f" Here is the detail of the SIte:\n Site : **{site_name}** (ID: **{id}**)\n Address: **{full_address}**, PIN: **{pincode}**\n City: **{city}**, State: **{state}**, Country: **{country}**\n MOBILE: **{mobile}**\n Geofencing Enabled: **{geo_fencing_enabled}**, Geofencing Distance: **{geo_fencing_distance}**\n Custom Field Data:**{custom_field_data}** ",
                "data":{"site_id": id}
            }


        else:
            clear_conversation_from_db(session_id)
            return {
                "success": False,
                "message": f"❌ Unknown operation: {operation_type}",
                "data": {"error": "Unknown operation"}
            }
    


            
    


    except Exception as e:
        logger.error(f"Site operation error: {e}")
        return {
            "success": False,
            "message": f"❌ Operation failed: {str(e)}",
            "data": {"error": str(e)}
        }
    


def field_exists(response: dict, field: str) -> bool:
    "checks if field exist in response"
    print("inside field_exist , response:", response)
    if not isinstance(response, dict):
        return False
    return field in response

