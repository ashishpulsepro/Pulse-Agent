import os, re, json, logging
from typing import List
try:
    import google.generativeai as genai  # type: ignore
except ImportError:  # allow running without package in some environments
    genai = None  # type: ignore

# Name of environment variable that should hold the Gemini API key
API_KEY_ENV = "AIzaSyCUxlOt5vBktkok8TniXH_1Cmv-nQjyWrk"
logger = logging.getLogger(__name__)

class GeminiProvider:
    def __init__(self, model: str = "gemini-2.5-flash-lite", temperature: float = 0.2):
        self.temperature = temperature
        self.available = False
        api_key = os.getenv(API_KEY_ENV)
        if api_key and genai is not None:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model)
                self.available = True
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.warning(f"Failed to init Gemini, falling back to heuristic: {e}")
        else:
            logger.warning("Gemini API key missing or library not installed; using heuristic intent detection only.")
            self.model = None

    def classify_intent(self, user_message: str, operations: List[str]) -> str:
        # LLM path
        if self.available and self.model:
            try:
                prompt = (
                    "You are an intent classifier. Choose the BEST EXACT operation name from this list: "
                    f"{operations}. If nothing matches return UNKNOWN.\nUser message: {user_message}\nOperation:"
                )
                logger.debug(f"CLASSIFY prompt: {(prompt[:300])}...")
                resp = self.model.generate_content(prompt)
                text_raw = (getattr(resp, 'text', '') or '').strip()
                logger.debug(f"CLASSIFY raw: {text_raw[:200]}")
                text = text_raw.upper()
                if text in operations:
                    return text
            except Exception as e:
                logger.warning(f"Gemini classify failed, fallback heuristic. {e}")
        # Heuristic fallback keywords
        msg = user_message.lower()
        kw_map = [
            (r"create +site|new +site|add +site", "CREATE_SITE"),
            (r"list +sites|all +sites|view +sites", "VIEW_SITES"),
            (r"create +user|add +user|new +user", "CREATE_USER"),
            (r"assign +.*user.+site|add +users? +to +site", "ASSIGN_USERS_TO_SITE"),
            (r"create +template|new +template|checklist", "CREATE_TEMPLATE"),
        ]
        for pattern, op in kw_map:
            if re.search(pattern, msg):
                if op in operations:
                    return op
        return 'UNKNOWN'

    def extract_fields(self, user_message: str, operation: str, missing_fields: List[str]) -> dict:
        if not missing_fields:
            return {}
        fields_str = ', '.join(missing_fields)
        prompt = f"User message: '{user_message}'. Operation: {operation}. Extract values for these fields if present: {fields_str}. Respond with a JSON object only containing found fields. If none found return an empty JSON object {{}}."
        if self.available and self.model:
            try:
                logger.debug(f"EXTRACT prompt: {prompt[:300]}...")
                resp = self.model.generate_content(prompt)
                raw = (getattr(resp, 'text', '') or '').strip()
                logger.debug(f"EXTRACT raw: {raw[:300]}")
            except Exception as e:
                logger.warning(f"Gemini extract failed, returning empty. {e}")
                return {}
        else:
            # simple heuristic: try to split "field: value" pairs
            raw = "{}"
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.lower().startswith('json'):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def ask_followup(self, operation: str, missing_fields: List[str]) -> str:
        fields_str = ', '.join(missing_fields)
        return f"I still need: {fields_str}. Please provide them."

    # --- Natural language generation helpers ---
    def natural_followup(self, operation: str, missing_fields: List[str]) -> str:
        if not missing_fields:
            return ""
        base = (
            f"To proceed with {operation.replace('_',' ').title()}, I still need: " +
            ', '.join(missing_fields) + ". "
        )
        if self.available and self.model:
            try:
                prompt = (
                    "Rewrite the following system prompt to sound like a friendly assistant asking a user for the missing information, keep it brief.\n" +
                    base
                )
                resp = self.model.generate_content(prompt)
                txt = (getattr(resp,'text','') or '').strip()
                if txt:
                    return txt
            except Exception:
                pass
        return base + "Please provide those details."

    def reformulate_unknown(self, user_message: str, operations: List[str]) -> str:
        ops_str = ', '.join(operations)
        base = (
            "I'm not completely sure what you want yet. "
            f"I can help with: {ops_str}. "
            "Briefly tell me which one you want and any key details in one sentence."
        )
        if self.available and self.model:
            try:
                prompt = (
                    "User asked: '" + user_message + "'\n" +
                    "Create a short, warm assistant reply that guides them to pick one of these operations: " + ops_str + "."
                )
                resp = self.model.generate_content(prompt)
                txt = (getattr(resp,'text','') or '').strip()
                if txt:
                    return txt
            except Exception:
                pass
        return base

    def summarize_result(self, operation: str, success: bool, raw: dict) -> str:
        if self.available and self.model:
            try:
                prompt = (
                    f"Operation: {operation}\nSuccess: {success}\nRaw Result JSON: {json.dumps(raw)[:800]}\n" \
                    "Write a concise single-sentence user-facing summary (no internal field names, no JSON)."
                )
                resp = self.model.generate_content(prompt)
                txt = (getattr(resp,'text','') or '').strip()
                if txt:
                    return txt
            except Exception:
                pass
        if success:
            return f"{operation.replace('_',' ').title()} completed successfully."
        return f"{operation.replace('_',' ').title()} failed: {raw.get('error','Unknown error')}"
