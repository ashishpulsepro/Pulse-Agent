from db.db_services import get_conversation_from_db,save_conversation_to_db,safe_extract_text

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ChatResponse(BaseModel):
    message: str
    status: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    session_intent: Optional[str] = None


from Prompt.data_collection import get_data_collection_prompt

# Updated execute_phase_1 function
async def execute_phase_1(session_id: str, user_message: str, client,intent:str,email:str,onboarding:bool) -> ChatResponse:
    """Phase 1: Normal chat - Intent detection and data collection"""
    # Get conversation history from MongoDB
    db_messages = get_conversation_from_db(session_id)
    conversation_history = ""
    print("db messages: ",db_messages)
    for msg in db_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_history += f"{role}: {msg['message']}\n"

    # Get current sites list
    # sites_list = get_sites_list_formatted()
    # print(f"Available sites: {sites_list}")

    # all_user_list = ollama_site_manager.get_all_users()
    # permission_set_list = ollama_permission_manager.get_all_permission_sets()
    print("getting prompt")
    new_prompt=get_data_collection_prompt(intent)
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

    BASE_PROMPT_FOR_ONBOARDING=f"""
====================PULSEPRO ONBOARDING AGENT====================

====================CORE IDENTITY====================
**You are PulsePro's dedicated onboarding assistant.**
- **STRICT SCOPE**: Only handle the 3 onboarding operations below
- **INTELLIGENCE**: Understand user intent and guide them appropriately
- **COMMUNICATION**: Always use structured, professional formatting

====================AVAILABLE OPERATIONS====================
**I can help you with these PulsePro onboarding steps:**

✅ **1. Create a Site**
✅ **2. Create User Account** 
✅ **3. Create a Checklist**

====================INTELLIGENT RESPONSES====================

**When user asks for help/guidance:**
→ Show the 3 available steps and ask which they prefer

**When user mentions keywords like:**
- "setup", "configure", "new site" → Guide to **Create a Site**
- "account", "profile", "user", "login" → Guide to **Create User Account**  
- "template", "checklist", "list" → Guide to **Create a Checklist**

**For unrelated requests:**
→ **"Thank you for reaching out! I specialize in PulsePro onboarding only.**

**I can help you with:**
- Create a Site
- Create User Account  
- Create a Checklist

**Which step would you like assistance with?"**

====================RESPONSE STRUCTURE====================
**Always format responses with:**
- **Bold headings**
- • Bullet points for lists
- Clear line breaks
- Professional tone
- Structured layout

====================CANCELLATION====================
**Keywords**: cancel, stop, exit, abort, halt, quit, terminate, end
**Response**: "**Onboarding cancelled.** No action taken."

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
        data={}
    )