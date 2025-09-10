


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
        "DELETE_TEMPLATE","AUTOMATE_CUSTOMER_ACCESS_SETTING","UPLOAD_CHECKLIST","UNKNOWN"
    ]
    
    # Improved intent detection prompt
    intent_prompt = f"""You are a PulsePro intent classifier.

====================ANALYSIS METHODOLOGY====================

Read the ENTIRE conversation from start to finish
Identify the user's FINAL/CURRENT intent (ignore earlier topics they moved away from)
Focus on the user's LAST clear request or direction
Look for confirmation words like "yes", "continue", "proceed", "build it"

====================USER MESSAGE ANALYSIS====================
USER MESSAGE: "{user_message}"
====================CONVERSATION CONTEXT====================
{conversation_history}
====================VALID INTENTS ONLY====================
{valid_intents}


====================INTENT DEFINITIONS====================

- AUTOMATE_CUSTOMER_ACCESS_SETTING: "Auto assign new locations to all users" or "Switch off/autounassign/remove new locations to all users" or "Auto assign new templates to all users" or "Switch off/autounassign/remove new templates to all users"
- CREATE_TEMPLATE: create/add/make/new template/checklist/form or (What are the checklists options for me?) or How can I build checklist
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
- UPLOAD_CHECKLIST: upload/custom/my template/checklist
- UNKNOWN: hello/hi/chat/help/other topics/ about the platform

====================DECISION LOGIC====================
Priority 1 - Final Intent Recognition:

If user says "yes", "continue", "build it", "proceed" after discussing a specific operation → Return that operation's intent
If user confirms they want to complete a process → Return the relevant CREATE_* intent

Priority 2 - Conversation Flow Analysis:

Trace the conversation to find what the user ultimately wants to accomplish
Ignore early topics if user moved to a different goal
Focus on the most recent clear direction

Priority 3 - Keyword Matching:

Auto + assign/unassign + location/template → AUTOMATE_CUSTOMER_ACCESS_SETTING
Permission + assign → ASSIGN_PERMISSION_SET_TO_USER
Permission + remove → UNASSIGN_PERMISSION_SET_FROM_USER
Template + create/build/make → CREATE_TEMPLATE
Template + show/list → SHOW_ALL_TEMPLATES
Site + create → CREATE_SITE
User + create → CREATE_USER

====================GENERIC EXAMPLES====================
Negation Patterns That Should Return UNKNOWN:

"I'll create [X] later" (postponement)
"I don't want to create [X]" (negation)
"Continue with default [X]" (current state)
"I'm the only [X]" (satisfaction)
"Not now, maybe later" (postponement)
"Skip [X] for now" (postponement)

Positive Patterns That Should Return Intent:

"Create [X]" (no negation words)
"Let's build [X]" (active confirmation)
"I want to make [X]" (clear intent)
"Yes, proceed with [X]" (confirmation)

Analysis Method:

Scan for negation/postponement words FIRST
If found near action words → Return UNKNOWN
If not found → Apply normal keyword matching

=========================================================

Look for key action words: create, delete, view, assign, remove
If unsure, return UNKNOWN
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
