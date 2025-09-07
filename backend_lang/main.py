from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import traceback
from pydantic import BaseModel
import uuid, sys, os
from typing import Dict

# Support running either:
#   (A) from project root:  uvicorn backend_lang.main:app --reload
#   (B) from inside folder: uvicorn main:app --reload
# We attempt absolute import first; on failure we add parent dir to sys.path and retry relatively.
try:
    from backend_lang.graph.graph_builder import build_graph  # type: ignore
    from backend_lang.graph.state import ConversationState  # type: ignore
    from backend_lang.graph import nodes  # type: ignore
except ImportError:  # running inside the package directory
    pkg_root = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(pkg_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        from backend_lang.graph.graph_builder import build_graph  # type: ignore
        from backend_lang.graph.state import ConversationState  # type: ignore
        from backend_lang.graph import nodes  # type: ignore
    except ImportError:
        # Final fallback: relative imports when executed as plain script
        from graph.graph_builder import build_graph  # type: ignore
        from graph.state import ConversationState  # type: ignore
        from graph import nodes  # type: ignore

import logging

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="PulsePro LangGraph Agent")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"GLOBAL EXCEPTION {request.method} {request.url.path}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"error":"Internal Server Error","detail":str(exc)})

GRAPH = build_graph()
SESSIONS: Dict[str, ConversationState] = {}

class ChatIn(BaseModel):
    message: str
    session_id: str | None = None

class ChatOut(BaseModel):
    session_id: str
    response: str
    operation: str | None
    done: bool
    missing: list[str]
    error: str | None

@app.post('/chat', response_model=ChatOut)
async def chat(inp: ChatIn):
    logger.info(f"/chat IN session={inp.session_id} msg='{inp.message}'")
    session_id = inp.session_id or str(uuid.uuid4())
    state = SESSIONS.get(session_id) or ConversationState(session_id=session_id)
    # feed user input
    state = nodes.user_input_node(state, inp.message)
    # run graph steps manually for now
    state = nodes.intent_detection_node(state)
    if not state.error:
        # fill slots
        state = nodes.slot_filling_node(state)
        # if still missing return early
        if state.missing:
            SESSIONS[session_id] = state
            return ChatOut(session_id=session_id, response=state.response or 'Need more info', operation=state.operation, done=state.done, missing=state.missing, error=state.error)
        # validate
        state = nodes.validation_node(state)
        # execute
        state = nodes.execution_node(state)
    SESSIONS[session_id] = state
    out = ChatOut(session_id=session_id, response=state.response, operation=state.operation, done=state.done, missing=state.missing, error=state.error)
    logger.info(f"/chat OUT session={session_id} op={out.operation} missing={out.missing} done={out.done} error={out.error} resp='{(out.response or '')[:120]}'")
    return out

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time, traceback
    start = time.time()
    try:
        response: Response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled exception during request {request.method} {request.url.path}: {e}\n{traceback.format_exc()}")
        raise
    finally:
        duration = (time.time() - start) * 1000
        logger.info(f"REQ {request.method} {request.url.path} -> {getattr(request,'state',{}).__dict__.get('status_code','?')} took {duration:.1f}ms")

@app.get('/health')
async def health():
    return {"status":"ok"}

@app.get('/operations')
async def operations():
    from backend_lang.utils.schema_loader import load_operation_schemas
    schemas = load_operation_schemas()
    logger.debug(f"Loaded operations: {list(schemas.keys())}")
    return {"operations": list(schemas.keys())}

@app.get('/debug/session/{session_id}')
async def debug_session(session_id: str):
    state = SESSIONS.get(session_id)
    if not state:
        return {"exists": False}
    return {"exists": True, "state": state.dict()}

@app.get('/debug/env')
async def debug_env():
    return {
        "GEMINI_API_KEY_present": bool(os.getenv('GEMINI_API_KEY')),
        "refresh_present": bool(os.getenv('refresh')),
        "LOG_LEVEL": os.getenv('LOG_LEVEL'),
    }
