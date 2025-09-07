from typing import Dict, Any
import logging
from .state import ConversationState
from backend_lang.utils.schema_loader import load_operation_schemas
from backend_lang.services.gemini_client import GeminiProvider
from backend_lang.services.execution import execute_operation
from pydantic import ValidationError

SCHEMAS = load_operation_schemas()
OPERATIONS = list(SCHEMAS.keys())
LLM = GeminiProvider()
logger = logging.getLogger(__name__)

OPERATION_DESCRIPTIONS = {
    "CREATE_SITE": "Create a new site/location by its name",
    "VIEW_SITES": "List all existing sites/locations",
    "CREATE_USER": "Create a new user with a permission set",
    "ASSIGN_USERS_TO_SITE": "Assign one or more existing users to a site",
    "CREATE_TEMPLATE": "Create a new checklist/template from an existing source"
}


def user_input_node(state: ConversationState, user_text: str) -> ConversationState:
    state.user_input = user_text
    return state


def intent_detection_node(state: ConversationState) -> ConversationState:
    if not state.operation:
        # Greeting / small talk detection
        if state.user_input.strip().lower() in {"hi","hello","hey","hey there","hi there"}:
            state.response = (
                "Hi! I can help with these operations: " + ", ".join(OPERATIONS) + ". "
                "For example: 'Create a site named Alpha' or 'List all sites'. What would you like to do?"
            )
            state.operation = None  # keep unset so user can pick one
            return state
        state.operation = LLM.classify_intent(state.user_input, OPERATIONS)
        logger.info(f"Detected intent candidate: {state.operation} for input='{state.user_input}'")
    # initialize missing fields list
    if state.operation in SCHEMAS:
        model = SCHEMAS[state.operation].build_pydantic_model()
        required_fields = [n for n, m in SCHEMAS[state.operation].fields.items() if m.get('required')]
        state.missing = [f for f in required_fields if f not in state.collected]
    else:
        # Provide natural language guidance via LLM helper
        state.response = LLM.reformulate_unknown(state.user_input, OPERATIONS)
        # leave error unset so client can prompt again
    return state


def slot_filling_node(state: ConversationState) -> ConversationState:
    if state.error or state.done:
        return state
    if not state.missing:
        return state
    extracted = LLM.extract_fields(state.user_input, state.operation, state.missing)
    logger.info(f"Extracted fields: {extracted}")
    for k, v in extracted.items():
        if k in state.missing:
            state.collected[k] = v
    # recompute missing
    state.missing = [f for f in state.missing if f not in state.collected]
    if state.missing:
        state.response = LLM.natural_followup(state.operation, state.missing)
    return state


def validation_node(state: ConversationState) -> ConversationState:
    if state.error or state.done:
        return state
    if state.missing:
        return state
    # Guard: if operation is unknown or not supported, skip
    if not state.operation or state.operation not in SCHEMAS:
        state.response = state.response or "I can't process that request yet. Try a supported operation."
        state.done = False
        return state
    model = SCHEMAS[state.operation].build_pydantic_model()
    try:
        model(**state.collected)  # validate only
        state.response = "All required information collected. Executing now..."
    except ValidationError as ve:
        state.error = f"VALIDATION_ERROR: {ve.errors()}"
    return state


def execution_node(state: ConversationState) -> ConversationState:
    if state.error or state.done:
        return state
    if state.missing:
        return state
    if not state.operation or state.operation not in SCHEMAS:
        # Don't mark done so user can continue providing a valid request
        if not state.response:
            state.response = "Please specify what you'd like to do. Supported operations: " + ", ".join(SCHEMAS.keys())
        return state
    try:
        result = execute_operation(state.operation, state.collected)
        logger.info(f"Execution result: {result}")
        if 'error' in result:
            state.error = result['error']
            state.response = LLM.summarize_result(state.operation, False, result)
        else:
            state.response = LLM.summarize_result(state.operation, True, result)
    except Exception as e:
        logger.exception(f"Execution crashed for op={state.operation}")
        state.error = f"EXECUTION_ERROR: {e}"
        state.response = "An internal error occurred while executing the operation."
    finally:
        state.done = True
    return state
