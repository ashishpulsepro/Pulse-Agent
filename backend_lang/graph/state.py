from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ConversationState(BaseModel):
    session_id: str
    user_input: str = ''
    operation: Optional[str] = None
    collected: Dict[str, Any] = Field(default_factory=dict)
    missing: List[str] = Field(default_factory=list)
    done: bool = False
    response: str = ''
    error: Optional[str] = None
