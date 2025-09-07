# PulsePro LangGraph Backend

Experimental conversational agent using LangGraph + Gemini Flash Lite.

## Features
- Intent detection over dynamic operation schemas
- Slot filling with multi-turn conversation
- Dynamic Pydantic model creation from JSON schema files in `schemas/`
- Execution mapped to existing PulsePro managers in `backend/site_manager.py`

## Run
Set environment variables:
```
set GEMINI_API_KEY=your_key_here
set refresh=<refresh_jwt>
```
Install deps (can reuse root venv):
```
pip install -r requirements.txt
```
Start API:
```
uvicorn backend_lang.main:app --reload --port 9000
```

## Chat Endpoint
POST http://localhost:9000/chat
```json
{ "message": "Create a site called Mumbai HO" }
```
Response shows missing fields or success output.

## Adding Operations
1. Add JSON schema to `schemas/` (operation key must be unique)
2. Implement execution logic in `services/execution.py` if new behavior
3. No restart code change needed - reload server to pick up schema.
