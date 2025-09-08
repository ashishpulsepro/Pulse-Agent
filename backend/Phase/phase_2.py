
import logging
import json
from db.db_services import get_conversation_from_db,clear_conversation_from_db,save_conversation_to_db
from LLM.initialize_llm import get_gemini_client
from Phase.execute_operation import execute_site_operation


from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    message: str
    status: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    session_intent: Optional[str] = None

from Prompt.json_conversion import get_json_response_prompt
async def execute_phase_2(session_id: str,intent:str,email:str, client=get_gemini_client(temperature=0.05)) -> ChatResponse:
    """Phase 2: Generate JSON and execute operation"""
    
    try:
        # Get conversation history from MongoDB for Phase 2
        db_messages = get_conversation_from_db(session_id)
        conversation_history = ""
        for msg in db_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_history += f"{role}: {msg['message']}\n"
        
        new_prompt=get_json_response_prompt(intent)
        # Create Phase 2 prompt
        phase_2_prompt = f"""{new_prompt}

====================CONVERSATION HISTORY====================
{conversation_history}

     JSON:"""
        print("Phase 2 prompt: ",phase_2_prompt)
        
        # Get JSON response from LLM
        response = client.generate_content(
            phase_2_prompt
        )
        
        json_response = response.text.strip()
        
        # Clean and parse JSON
        if json_response.startswith('```'):
            # Remove markdown formatting if present
            json_response = json_response.split('```')[1]
            if json_response.startswith('json'):
                json_response = json_response[4:]
        
        # Parse the JSON
        operation_data = json.loads(json_response)
        print("operation data : ", operation_data)
        
        # Execute the operation
        execution_result = await execute_site_operation(operation_data,session_id)
        
        # Clear conversation after execution
        clear_conversation_from_db(session_id)
        save_conversation_to_db(email=email,message=execution_result["message"],session_id=session_id,role="Assistant",intent='UNKNOWN')
        
        return ChatResponse(
            message=execution_result["message"],
            status="completed",
            session_id=session_id,
            context={
                "phase": 2,
                "operation": operation_data["operation_type"],
                "executed": True
            },
            data=execution_result["data"]
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Raw response: {json_response}")
        return ChatResponse(
            message="❌ Failed to parse operation data. Please try again.",
            status="error",
            session_id=session_id,
            context={"phase": 2, "error": "json_parse_error"},
            data={"raw_response": json_response}
        )
    
    except Exception as e:
        logger.error(f"Phase 2 execution error: {e}")
        return ChatResponse(
            message=f"❌ Failed to execute operation: {str(e)}",
            status="error",
            session_id=session_id,
            context={"phase": 2, "error": str(e)},
            data={}
        )
