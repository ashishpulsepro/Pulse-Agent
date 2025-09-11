from db.db_services import get_conversation_from_db,save_conversation_to_db,safe_extract_text,store_session_intent,clear_conversation_from_db

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from services.Authentication_Service import AuthenticationManager

class ChatResponse(BaseModel):
    message: str
    status: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    session_intent: Optional[str] = None
    showUploadButton : Optional[bool]= False



from Prompt.data_collection import get_data_collection_prompt

# Updated execute_phase_1 function
async def execute_phase_1(session_id: str, user_message: str, client,intent:str,email:str,auth:AuthenticationManager,onboarding:bool) -> ChatResponse:
    """Phase 1: Normal chat - Intent detection and data collection"""
    # Get conversation history from MongoDB
    db_messages = get_conversation_from_db(session_id)
    conversation_history = ""
    print("db messages: ",db_messages)
    for msg in db_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_history = f"{role}: {msg['message']}\n {conversation_history}"

    # Get current sites list
    # sites_list = get_sites_list_formatted()
    # print(f"Available sites: {sites_list}")
    if intent=='UPLOAD_CHECKLIST':
        print("UPLOAD_CHECKLIST inside")
        store_session_intent(session_id,'UNKNOWN')
        save_conversation_to_db(session_id, "assistant", "Great! Please upload the file below",email=email, intent='UNKNOWN')
        clear_conversation_from_db(session_id=session_id)

        return ChatResponse(
            message="Great! Please upload the file below",
            status="completed",
            session_id=session_id,
            context={
                "phase": 2,
                "operation": intent,
                "executed": True
            },
            data={},
            showUploadButton=True
        )

    # all_user_list = ollama_site_manager.get_all_users()
    # permission_set_list = ollama_permission_manager.get_all_permission_sets()
    print("getting prompt")
    new_prompt=get_data_collection_prompt(intent,auth=auth)
    BASE_PROMPT="""  """
    # Format the prompt with sites list
    # formatted_prompt = PHASE_1_PROMPT.format(all_sites_list=sites_list,all_users_list=all_user_list,all_permission_sets_list=permission_set_list)
    BASE_PROMPT_FOR_ALL="""You are PulsePro AI Assistant.

====================CORE RULES====================
• Handle ONLY PulsePro operations as specified for this task
• Ignore unrelated queries. Reply: "I can only help with PulsePro operations."
• Ask for missing information. Never assume values.
• If user says cancel/stop/exit/abort/halt/quit/terminate/end, reply: "Operation cancelled. No action taken."
• When you have all required data, ask: "I have all the information needed. Type 'Proceed' to execute this operation."
• Must always respond in a structural and concise manner. Use bullet points or numbered lists for clarity. Highlight the main heading. Give spaces and line breaks for readability.
• Use simple, non-technical language. Avoid jargon.
"""

    BASE_PROMPT_FOR_ONBOARDING=f"""====================PULSEPRO ONBOARDING ASSISTANT====================
Goal: Guide users through PulsePro's onboarding process by helping them choose and complete essential setup steps.


=========================INTENT=========================
{intent}
Greatest Priority : when intent is 'UNKNOWN_1',strictly just show all the operations that can be performed and nothing else

====================AVAILABLE OPERATIONS====================
site_creation
new_user_setup
checklist_creation

====================CONVERSATION FLOW====================
1. If user is new or asks for general help:
   - Welcome them warmly
   - Present the three available operations: "I can help you with site creation, user account setup, or checklist creation"
   - Ask which step sounds most relevant or offer to recommend a starting point

2. If user mentions specific keywords:
   - Site-related ("setup", "configure", "domain") → Guide to Site Creation
   - User-related ("account", "profile", "team", "login") → Guide to New User Setup
   - Workflow-related ("template", "checklist", "tasks", "process") → Guide to Checklist Creation

3. Once user selects an operation:
   - Confirm their choice and begin that specific operation flow

====================DECISION LOGIC====================
If user request is unclear → Present all three options and ask for preference.
If user mentions operation keywords → Guide to relevant operation.
If user request is related but outside scope → Acknowledge helpfully, then redirect to onboarding.
If user request is unrelated → Politely redirect to available onboarding operations.
If user selects an operation → Begin that operation's specific flow.
If conversation History shows that user has already performed some operation then ask if user wants to do that operation again or continue with other 2 operations
If conversation History shows that the user switches from one operation to another without completion,then only ask him to strictly type 'cancel' and then start with another operation,BUT this logic must be avoided when the intent is "UNKNOWN_1" , instead show all the operations that can be performed

====================RULES====================
Be conversational, helpful, and adapt to user's tone.
Always stay focused on the three core onboarding operations.
Provide context about why each step matters.
Use **bold** for key concepts and bullet points for clarity.
Ask clarifying questions when needs aren't clear.
Never abruptly shut down conversations - redirect helpfully.
If conversation History shows that the user switches from one operation to another without completion ,then only ask him to strictly type 'cancel' and then start with another operation, BUT this rule must be avoided when intent is 'UNKNOWN_1', instead show all the operations that can be performed

====================RESPONSE PATTERNS====================

**For new users:**
"Welcome to PulsePro! I'm here to help you get set up. Most users start with:
1 • **Site Creation** - Set up and configure your site
2 • **New User Setup** - Create profiles and manage permissions  
3 • **Checklist Creation** - Build industry standard checklists

What sounds most relevant to where you are, or would you like me to recommend a starting point?"
=========================================================

**If user asks for recommendation or guidence, start with Site Creation**
then if the user says yes , means the intent is CREATE_SITE

**For related but outside-scope questions:**
"That's a great question! While I specialize in getting you set up initially, I can help you with [relevant onboarding step] right now. Would you like to start there?"

**For unrelated requests:**
"I specialize in PulsePro onboarding and can help you with site creation, user account setup, or checklist creation. Which would be most helpful?"

**if User switches from one operation to another**
Okay, I understand you'd like to switch.
Since you've switched operations, please type 'cancel' to stop the current task, and then you can tell me what would you like to do next.
====================AVAILABLE OPTIONS====================
Operations: Site Creation, User Setup, Checklist Creation
"""

    if onboarding==True:
        BASE_PROMPT=BASE_PROMPT_FOR_ONBOARDING
    else:
        BASE_PROMPT=BASE_PROMPT_FOR_ALL


    # Create full prompt for Phase 1
    full_prompt = BASE_PROMPT    +      f"""
====================CONVERSATION HISTORY====================
{conversation_history}

====================Follow the instructions below and keep the CONVERSATION HISTORY in mind====================

{new_prompt}

====================CURRENT USER MESSAGE====================
{user_message}
====================YOUR RESPONSE===================="""
    print("New prompt: ",full_prompt)

    print("into llm now")
    # Get response from LLM
    response = client.generate_content(
        full_prompt
    )
    print("response from llm: ",response)
    ai_response = safe_extract_text(response).strip() or "UNKNOWN"

    
    # Save AI response to MongoDB
    save_conversation_to_db(session_id, "assistant", ai_response,email=email, intent=intent)
    
    # Determine status
    if "Type 'Proceed' to execute".lower() in ai_response.lower():
        status = "ready_for_execution"
    elif "I can only help with PulsePro Site operations" in ai_response:
        status = "off_topic"
    else:
        status = "collecting_data"
    
    return ChatResponse(
        message=ai_response,
        status=status,
        session_id=session_id,
        context={
            "phase": 1,
            "conversation_length": len(get_conversation_from_db(session_id)),
            "intent": intent},
        data={},
        showUploadButton=False

    )