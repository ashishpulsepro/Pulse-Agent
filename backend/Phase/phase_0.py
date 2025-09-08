


from db.db_services import get_conversation_from_db
from LLM.initialize_llm import get_gemini_client






async def execute_phase_0(session_id: str, user_message: str, client=get_gemini_client(temperature=0.1)) -> str:
    """Phase 0: Optimized Intent detection - returns only the intent string"""

    # Get conversation history from MongoDB (optimized - only recent messages)
    db_messages = get_conversation_from_db(session_id)
    conversation_history = "\n".join([
    f"User: {msg['message']}"
    for msg in db_messages[-5:]  # Last 5 messages
    if msg['role'] == 'user'  # Only user messages
    ])
    
    # Valid intents list
    valid_intents = [
        "CREATE_SITE", "DELETE_SITE", "VIEW_SITES", 
        "ASSIGN_USERS_TO_SITE", "UNASSIGN_USERS_FROM_SITE", 
        "CREATE_USER", "DELETE_USER", "VIEW_USERS", 
        "VIEW_PERMISSION_SETS", "ASSIGN_PERMISSION_SET_TO_USER", 
        "UNASSIGN_PERMISSION_SET_FROM_USER","DETAIL_SPECIFIC_USER","DETAIL_SPECIFIC_SITE",
        "SHOW_ALL_TEMPLATES", "ASSIGN_TEMPLATE_TO_USER", "UNASSIGN_TEMPLATE_FROM_USER", "CREATE_TEMPLATE",
        "CREATE_A_GROUP","DELETE_A_GROUP","ADD_USER_TO_GROUP","REMOVE_USER_FROM_GROUP","VIEW_ALL_GROUPS","SHOW_USERS_ADDED_TO_GROUP","SHOW_USERS_ADDED_NOT_TO_GROUP",
        "DELETE_TEMPLATE","AUTOMATE_CUSTOMER_ACCESS_SETTING","UNKNOWN"
    ]
    
    # Improved intent detection prompt
    intent_prompt = f"""You are a PulsePro intent classifier.

ANALYZE the user message and conversation history to determine the user's intent. 
USER MESSAGE: "{user_message}"

CONVERSATION HISTORY:
{conversation_history}

ONLY VALID INTENTS:
{valid_intents}


INTENTS:
- AUTOMATE_CUSTOMER_ACCESS_SETTING: "Auto assign new locations to all users" or "Switch off/autounassign/remove new locations to all users" or "Auto assign new templates to all users" or "Switch off/autounassign/remove new templates to all users"
- CREATE_TEMPLATE: create/add/make/new template/checklist/form
- SHOW_ALL_TEMPLATES: show/list/see/display all templates/checklist/forms/
- ASSIGN_TEMPLATE_TO_USER: assign/give/allot template/checklist/form to user/employee/staff/person/man or assign/give/allot user/employee/staff/person/man  to template/checklist/form
- UNASSIGN_TEMPLATE_FROM_USER: remove/take/unassign/revoke template/checklist/form from user/employee/staff/person/man or remove/take/unassign/revoke user/employee/staff/person/man from template/checklist/form
- VIEW_PERMISSION_SETS: show/list/see permissions/roles/access
- ASSIGN_PERMISSION_SET_TO_USER: give/assign permissions/roles/access to user/employee or give/assign user/employee permissions/roles/access
- UNASSIGN_PERMISSION_SET_FROM_USER: remove/take permissions/roles/access from user/employee or remove/take user/employee permissions/roles/access
- DELETE_TEMPLATE: delete/remove template/checklist/form
- CREATE_SITE: create/add/make/new/open/set up site/office/location/branch
- DELETE_SITE: delete/remove/close site/office/location/branch
- VIEW_SITES: show/list/see/get all sites/offices/locations/branches
- ASSIGN_USERS_TO_SITE: assign/add/move user/employee to site/office/location or assign/add/move site/office/location to user/employee
- UNASSIGN_USERS_FROM_SITE: remove/take user/employee from site/office/location or remove/take site/office/location from user/employee
- CREATE_USER: create/add/make/set up/new user/account/employee/person
- DELETE_USER: delete/remove user/employee/account/person
- VIEW_USERS: show/list/see/get all users/employees/accounts/people
- VIEW_ALL_GROUPS: state/show/list/see all the groups in the system
- CREATE_A_GROUP: create/add/make new group/cluster
- DELETE_A_GROUP: delete/remove a group
- ADD_USER_TO_GROUP: add/assign/allot user to Group/cluster
- REMOVE_USER_FROM_GROUP: remove/delete/take user from Group/cluster
- SHOW_USERS_ADDED_TO_GROUP: show/list/display users/member added/assigned to a group/cluster
- SHOW_USERS_ADDED_NOT_TO_GROUP: show/list/display users/member not added/assigned to a group/cluster
- DETAIL_SPECIFIC_USER: show/list/display  detail/more about/of a specific user/member
- DETAIL_SPECIFIC_SITE: show/list/display  detail/more about/of a specific site/location
- UNKNOWN: hello/hi/chat/help/other topics/ about the platform

RULES:
0. If message contains "auto" or "automaticcaly" + "assign" or "unassign" + "location/site" or "checklist/template" ->return :  AUTOMATE_CUSTOMER_ACCESS_SETTING
1. If message contains "permission/role/access" + "assign/give" → ASSIGN_PERMISSION_SET_TO_USER
2. If message contains "permission/role/access" + "remove/revoke" → UNASSIGN_PERMISSION_SET_FROM_USER
3. If message contains "user" + "to" + "site/office" → ASSIGN_USERS_TO_SITE
4. If message contains "show","display","list","see"+ "template","forms" → SHOW_ALL_TEMPLATES
5. If message contains "remove" + "template" → DELETE_TEMPLATE
6. Look for key action words: create, delete, view, assign, remove
7. If unsure, return UNKNOWN
CRITICAL: Return ONLY the intent name from the list of valid intents (e.g., "CREATE_SITE" or "UNKNOWN"). No explanations, no other text.
"""

    try:
        # Get response from LLM with optimized settings
        response = client.generate_content(
           intent_prompt
        )
        
        detected_intent = (response.text or "").strip().upper()
        print(f"Detected intent: {detected_intent}")
        
        # Validate intent and return
        return detected_intent if detected_intent in valid_intents else "UNKNOWN"
        
    except Exception as e:
        print(f"Error in intent detection: {e}")
        return "UNKNOWN"
