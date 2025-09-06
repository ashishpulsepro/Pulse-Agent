



def get_json_response_prompt(operation_type):
    """
    Returns operation-specific JSON generation prompt based on the operation type.
    
    Args:
        operation_type (str): The type of operation (e.g., 'CREATE_SITE', 'DELETE_USER', etc.)
        
    Returns:
        str: The specific JSON generation prompt for the operation
    """
    
    # Base rules that apply to all JSON generation operations
    BASE_RULES = """You are a JSON generator for PulsePro operations.

ANALYZE the conversation history and extract the required data to generate a JSON response.

CRITICAL RULES:
- Extract the data from the conversation history
- Return ONLY the JSON object, nothing else
- No explanations, no text, no markdown, just pure JSON
- Use exact field names and structure as specified
- Follow the JSON format strictly

"""

    # Operation-specific JSON generation prompts
    prompts = {
        'CREATE_SITE': BASE_RULES + """====================CREATE_SITE JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- location_name: The name/location of the site to be created

EXACT JSON FORMAT TO RETURN:
{"data": {"location_name": "EXTRACTED_NAME"}, "operation_type": "CREATE_SITE"}

EXAMPLE:
If conversation mentions creating "Mumbai Office", return:
{"data": {"location_name": "Mumbai Office"}, "operation_type": "CREATE_SITE"}

Extract the location_name from the conversation and generate the JSON response.
""",

        'VIEW_SITES': BASE_RULES + """====================VIEW_SITES JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- No data extraction needed for this operation

EXACT JSON FORMAT TO RETURN:
{"data": {}, "operation_type": "VIEW_SITES"}

EXAMPLE:
Always return exactly:
{"data": {}, "operation_type": "VIEW_SITES"}

Generate the JSON response with empty data object.
""",

        'DELETE_SITE': BASE_RULES + """====================DELETE_SITE JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- location_name: The name/location of the site to be deleted

EXACT JSON FORMAT TO RETURN:
{"data": {"location_name": "EXTRACTED_NAME"}, "operation_type": "DELETE_SITE"}

EXAMPLE:
If conversation mentions deleting "Bangalore Hub", return:
{"data": {"location_name": "Bangalore Hub"}, "operation_type": "DELETE_SITE"}

Extract the location_name from the conversation and generate the JSON response.
""",

        'ASSIGN_USERS_TO_SITE': BASE_RULES + """====================ASSIGN_USERS_TO_SITE JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- location_name: The name/location of the site
- user_list: List of user names to be assigned to the site

EXACT JSON FORMAT TO RETURN:
{"data": {"location_name": "EXTRACTED_NAME", "user_list": ["USER1", "USER2"]}, "operation_type": "ASSIGN_USERS_TO_SITE"}

EXAMPLE:
If conversation mentions assigning "John, Sarah" to "Delhi Office", return:
{"data": {"location_name": "Delhi Office", "user_list": ["John", "Sarah"]}, "operation_type": "ASSIGN_USERS_TO_SITE"}

Extract the location_name and user_list from the conversation and generate the JSON response.
""",

        'UNASSIGN_USERS_FROM_SITE': BASE_RULES + """====================UNASSIGN_USERS_FROM_SITE JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- location_name: The name/location of the site
- user_list: List of user names to be unassigned from the site

EXACT JSON FORMAT TO RETURN:
{"data": {"location_name": "EXTRACTED_NAME", "user_list": ["USER1", "USER2"]}, "operation_type": "UNASSIGN_USERS_FROM_SITE"}

EXAMPLE:
If conversation mentions unassigning "John, Sarah" from "Delhi Office", return:
{"data": {"location_name": "Delhi Office", "user_list": ["John", "Sarah"]}, "operation_type": "UNASSIGN_USERS_FROM_SITE"}

Extract the location_name and user_list from the conversation and generate the JSON response.
""",

        'CREATE_USER': BASE_RULES + """====================CREATE_USER JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- first_name: User's first name
- last_name: User's last name  
- email: User's email address
- permission_set: Permission set assigned to the user

EXACT JSON FORMAT TO RETURN:
{"data": {"first_name": "FIRSTNAME", "last_name": "LASTNAME", "email": "EMAIL", "permission_set": "PERMISSIONSET"}, "operation_type": "CREATE_USER"}

EXAMPLE:
If conversation mentions creating user "John Doe" with email "john@company.com" and "Field User" permission, return:
{"data": {"first_name": "John", "last_name": "Doe", "email": "john@company.com", "permission_set": "Field User"}, "operation_type": "CREATE_USER"}

Extract all four fields from the conversation and generate the JSON response.
""",

        'DELETE_USER': BASE_RULES + """====================DELETE_USER JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- full_name: Complete name of the user to be deleted

EXACT JSON FORMAT TO RETURN:
{"data": {"full_name": "FULLNAME"}, "operation_type": "DELETE_USER"}

EXAMPLE:
If conversation mentions deleting user "John Doe", return:
{"data": {"full_name": "John Doe"}, "operation_type": "DELETE_USER"}

Extract the full_name from the conversation and generate the JSON response.
""",

        'VIEW_USERS': BASE_RULES + """====================VIEW_USERS JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- No data extraction needed for this operation

EXACT JSON FORMAT TO RETURN:
{"data": {}, "operation_type": "VIEW_USERS"}

EXAMPLE:
Always return exactly:
{"data": {}, "operation_type": "VIEW_USERS"}

Generate the JSON response with empty data object.
""",

        'VIEW_PERMISSION_SETS': BASE_RULES + """====================VIEW_PERMISSION_SETS JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- No data extraction needed for this operation

EXACT JSON FORMAT TO RETURN:
{"data": {}, "operation_type": "VIEW_PERMISSION_SETS"}

EXAMPLE:
Always return exactly:
{"data": {}, "operation_type": "VIEW_PERMISSION_SETS"}

Generate the JSON response with empty data object.
""",

        'ASSIGN_PERMISSION_SET_TO_USER': BASE_RULES + """====================ASSIGN_PERMISSION_SET_TO_USER JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- full_name: Complete name of the user
- permission_set: List of permission sets to be assigned to the user

EXACT JSON FORMAT TO RETURN:
{"data": {"full_name": "FULLNAME", "permission_set": ["PERMISSIONSET1", "PERMISSIONSET2"]}, "operation_type": "ASSIGN_PERMISSION_SET_TO_USER"}

EXAMPLE:
If conversation mentions assigning "Field User, Admin" permissions to "John Doe", return:
{"data": {"full_name": "John Doe", "permission_set": ["Field User", "Admin"]}, "operation_type": "ASSIGN_PERMISSION_SET_TO_USER"}

Extract the full_name and permission_set list from the conversation and generate the JSON response.
""",

        'UNASSIGN_PERMISSION_SET_FROM_USER': BASE_RULES + """====================UNASSIGN_PERMISSION_SET_FROM_USER JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- full_name: Complete name of the user
- permission_set: List of permission sets to be unassigned from the user

EXACT JSON FORMAT TO RETURN:
{"data": {"full_name": "FULLNAME", "permission_set": ["PERMISSIONSET1", "PERMISSIONSET2"]}, "operation_type": "UNASSIGN_PERMISSION_SET_FROM_USER"}

EXAMPLE:
If conversation mentions unassigning "Field User, Admin" permissions from "John Doe", return:
{"data": {"full_name": "John Doe", "permission_set": ["Field User", "Admin"]}, "operation_type": "UNASSIGN_PERMISSION_SET_FROM_USER"}

Extract the full_name and permission_set list from the conversation and generate the JSON response.
""",

        'SHOW_ALL_TEMPLATES': BASE_RULES + """====================SHOW_ALL_TEMPLATES JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- No data extraction needed for this operation

EXACT JSON FORMAT TO RETURN:
{"data": {}, "operation_type": "SHOW_ALL_TEMPLATES"}

EXAMPLE:
Always return exactly:
{"data": {}, "operation_type": "SHOW_ALL_TEMPLATES"}

Generate the JSON response with empty data object.
""",

        'DELETE_TEMPLATE': BASE_RULES + """====================DELETE_TEMPLATE JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- template_name: Full name of the template to be deleted

EXACT JSON FORMAT TO RETURN:
{"data": {"template_name": "TEMPLATENAME"}, "operation_type": "DELETE_TEMPLATE"}

EXAMPLE:
If conversation mentions deleting template "Checklist 1", return:
{"data": {"template_name": "Checklist 1"}, "operation_type": "DELETE_TEMPLATE"}

Extract the template_name from the conversation and generate the JSON response.
""",

        'ASSIGN_TEMPLATE_TO_USER': BASE_RULES + """====================ASSIGN_TEMPLATE_TO_USER JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- user_name: Complete name of the user
- template_name: Full name of the templates to be assigned , can be single or multiple

EXACT JSON FORMAT TO RETURN:
{"data": {"user_name": "USERNAME", "template_name": [
"TEMPLATENAME 1", "TEMPLATENAME 2"]}, "operation_type": "ASSIGN_TEMPLATE_TO_USER"}

EXAMPLE:
If conversation mentions assigning template "Checklist 1" and "Checklist 2" to user "John Doe", return:
{"data": {"user_name": "John Doe", "template_name": ["Checklist 1","Checklist 2"]}, "operation_type": "ASSIGN_TEMPLATE_TO_USER"}

Extract the user_name and template_name from the conversation and generate the JSON response.
""",

        'UNASSIGN_TEMPLATE_FROM_USER': BASE_RULES + """====================UNASSIGN_TEMPLATE_FROM_USER JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- user_name: Complete name of the user
- template_name: Full name of the templates to be unassigned , can be single or multiple

EXACT JSON FORMAT TO RETURN:
{"data": {"user_name": "USERNAME", "template_name": [
"TEMPLATENAME 1", "TEMPLATENAME 2"]}, "operation_type": "UNASSIGN_TEMPLATE_FROM_USER"}

EXAMPLE:
If conversation mentions unassigning template "Checklist 1" and "Checklist 2" from user "John Doe", return:
{"data": {"user_name": "John Doe", "template_name": ["Checklist 1","Checklist 2"]}, "operation_type": "UNASSIGN_TEMPLATE_FROM_USER"}

Extract the user_name and template_name from the conversation and generate the JSON response.
""",

        'CREATE_TEMPLATE': BASE_RULES + """====================CREATE_TEMPLATE JSON GENERATION====================   

REQUIRED DATA TO EXTRACT:
- industry_name: Name of the industry
- template_name: Name of the template to be created

EXACT JSON FORMAT TO RETURN:
{"data": {"industry_name": "INDUSTRYNAME", "template_name": "TEMPLATENAME"}, "operation_type": "CREATE_TEMPLATE"}

EXAMPLE:
If conversation mentions creating template "10 Point Hygiene Check" in "Food and Hospitality" industry, return:
{"data": {"industry_name": "Food and Hospitality", "template_name": "10 Point Hygiene Check"}, "operation_type": "CREATE_TEMPLATE"}

Extract the industry_name and template_name from the conversation and generate the JSON response.
                 """,
        'AUTOMATE_CUSTOMER_ACCESS_SETTING': BASE_RULES + """====================AUTOMATE_CUSTOMER_ACCESS_SETTING JSON GENERATION====================    
          REQUIRED DATA TO EXTRACT (True/False): Auto assign new templates to all users - True , Auto assign new locations to all users - True , Switch-off/Auto-unassign  new templates to all users - False , Switch-off/Auto unassign new locations to all users - False        
            
EXACT JSON FORMAT TO RETURN:
{"data": {"accessToAllSite":true/false,"accessToAllChecklist":true/false}, "operation_type": "AUTOMATE_CUSTOMER_ACCESS_SETTING"}

EXAMPLE:
If conversation mentions "Auto assign new locations to all users" , return:
{"data": {"accessToAllSite":true}, "operation_type": "AUTOMATE_CUSTOMER_ACCESS_SETTING"}
If conversation mentions "Switch off/autounassign new locations to all users" , return:
{"data": {"accessToAllSite":false}, "operation_type": "AUTOMATE_CUSTOMER_ACCESS_SETTING"}
If conversation mentions "Auto assign new templates to all users" , return:
{"data": {"accessToAllChecklist":true}, "operation_type": "AUTOMATE_CUSTOMER_ACCESS_SETTING"}
If conversation mentions "Switch off/autounassign new templates to all users" , return:
{"data": {"accessToAllChecklist":false}, "operation_type": "AUTOMATE_CUSTOMER_ACCESS_SETTING"}
If conersation mentions "Auto assign new templates to all users and Auto assign new locations to all users", return :
{"data": {"accessToAllSite":true,"accessToAllChecklist":true}, "operation_type": "AUTOMATE_CUSTOMER_ACCESS_SETTING"}
Extract the accessToAllSite and accessToAllChecklist from the conversation and generate the JSON response.
                """,

                'VIEW_ALL_GROUPS':BASE_RULES + """====================SHOW_ALL_TEMPLATES JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- No data extraction needed for this operation

EXACT JSON FORMAT TO RETURN:
{"data": {}, "operation_type": "VIEW_ALL_GROUPS"}

EXAMPLE:
Always return exactly:
{"data": {}, "operation_type": "VIEW_ALL_GROUPS"}

Generate the JSON response with empty data object.
""",

                'DELETE_A_GROUP':BASE_RULES + """====================DELETE_A_GROUP JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- group_name: Full name of the group to be deleted

EXACT JSON FORMAT TO RETURN:
{"data": {"group_name": "GROUP_NAME"}, "operation_type": "DELETE_A_GROUP"}

EXAMPLE:
If conversation mentions deleting group "Group 1", return:
{"data": {"group_name": "Group 1"}, "operation_type": "DELETE_A_GROUP"}

Extract the group_name from the conversation and generate the JSON response.
""",

                'CREATE_A_GROUP':BASE_RULES + """====================CREATE_A_GROUP JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- group_name: Full name of the group to be deleted

EXACT JSON FORMAT TO RETURN:
{"data": {"group_name": "GROUP_NAME"}, "operation_type": "CREATE_A_GROUP"}

EXAMPLE:
If conversation mentions creating group "Group 1", return:
{"data": {"group_name": "Group 1"}, "operation_type": "CREATE_A_GROUP"}

Extract the group_name from the conversation and generate the JSON response.
""",

                'ADD_USER_TO_GROUP':BASE_RULES + """====================ADD_USER_TO_GROUP JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- group_name: Complete name of the group
- user_names: Full Names of the users to be assigned , can be single or multiple

EXACT JSON FORMAT TO RETURN:
{"data": {"user_names": ["USERNAME 1","USERNAME 2","USERNAME 3"], "group_name":"GROUP 1"}, "operation_type": "ADD_USER_TO_GROUP"}

EXAMPLE:
If conversation mentions assigning user "User 1" and "User 2" to group "Group 1", return:
{"data": {"user_names": ["User 1","User 2"], "group_name":"Group 1"}, "operation_type": "ADD_USER_TO_GROUP"}

Extract the user_names and group_name from the conversation and generate the JSON response.
""",

                'REMOVE_USER_FROM_GROUP':BASE_RULES + """====================REMOVE_USER_FROM_GROUP JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- group_name: Complete name of the group
- user_names: Full Names of the users to be removed , can be single or multiple

EXACT JSON FORMAT TO RETURN:
{"data": {"user_names": ["USERNAME 1","USERNAME 2","USERNAME 3"], "group_name":"GROUP 1"}, "operation_type": "REMOVE_USER_FROM_GROUP"}

EXAMPLE:
If conversation mentions remove user "User 1" and "User 2" from group "Group 1", return:
{"data": {"user_names": ["User 1","User 2"], "group_name":"Group 1"}, "operation_type": "REMOVE_USER_FROM_GROUP"}

Extract the user_names and group_name from the conversation and generate the JSON response.
""",

                'SHOW_USERS_ADDED_TO_GROUP':BASE_RULES + """====================SHOW_USERS_ADDED_TO_GROUP JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- group_name: Full name of the group 

EXACT JSON FORMAT TO RETURN:
{"data": {"group_name": "GROUP_NAME"}, "operation_type": "SHOW_USERS_ADDED_TO_GROUP"}

EXAMPLE:
If conversation mentions show users of group "Group 1", return:
{"data": {"group_name": "Group 1"}, "operation_type": "SHOW_USERS_ADDED_TO_GROUP"}

Extract the group_name from the conversation and generate the JSON response.
""",

                'SHOW_USERS_ADDED_NOT_TO_GROUP':BASE_RULES + """====================SHOW_USERS_ADDED_NOT_TO_GROUP JSON GENERATION====================

REQUIRED DATA TO EXTRACT:
- group_name: Full name of the group 

EXACT JSON FORMAT TO RETURN:
{"data": {"group_name": "GROUP_NAME"}, "operation_type": "SHOW_USERS_ADDED_NOT_TO_GROUP"}

EXAMPLE:
If conversation mentions show users of group "Group 1", return:
{"data": {"group_name": "Group 1"}, "operation_type": "SHOW_USERS_ADDED_NOT_TO_GROUP"}

Extract the group_name from the conversation and generate the JSON response.
""",

                'DETAIL_SPECIFIC_USER':BASE_RULES +"""==========================DETAIL_SPECIFIC_USER=================================
REQUIRED DATA TO EXTRACT:
- user_name: full name of the user

EXACT JSON FORMAT TO RETURN:
{"data": {"user_name": "USER_NAME"}, "operation_type": "DETAIL_SPECIFIC_USER"}

EXAMPLE:
If conversation mentions show detail of "User 1", return:
{"data": {"user_name": "User 1"}, "operation_type": "DETAIL_SPECIFIC_USER"}

Extract the user_name from the conversation and generate the JSON response.
""",

                'DETAIL_SPECIFIC_SITE':BASE_RULES +"""==========================DETAIL_SPECIFIC_SITE=================================
REQUIRED DATA TO EXTRACT:
- site_name: full name of the site/location

EXACT JSON FORMAT TO RETURN:
{"data": {"site_name": "SITE_NAME"}, "operation_type": "DETAIL_SPECIFIC_SITE"}

EXAMPLE:
If conversation mentions show detail of Site 1 site, return:
{"data": {"site_name": "Site 1"}, "operation_type": "DETAIL_SPECIFIC_SITE"}

Extract the user_name from the conversation and generate the JSON response.
""",

    }
    
    # Return the specific prompt or a default message if operation not found
    return prompts.get(operation_type.upper(), "Invalid operation type for JSON generation. Please use one of: " + ", ".join(prompts.keys()))


