
import ollama
import uuid
from datetime import datetime
from typing import Dict, List, Optional

# Global storage for chat sessions (in production, use Redis or database)
chat_sessions: Dict[str, List[Dict]] = {}

def get_ollama_client():
    """Get Ollama client - simple dependency"""
    try:
        client = ollama.Client(host='http://localhost:11434')
        # Quick test to ensure connection
        client.list()
        return client
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama service unavailable: {str(e)}"
        )
    

    

import os
from dotenv import load_dotenv
from fastapi import HTTPException, status
import google.generativeai as genai

load_dotenv()

import google.generativeai as genai

def get_gemini_client(
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    top_p: float = 0.95,
    api_key: str = None
):
    """
    Initializes and returns a Gemini model client with the given config.

    Args:
        model (str): The Gemini model to use.
        temperature (float): Controls randomness (higher = more creative).
        max_output_tokens (int): Max tokens in output.
        top_p (float): Nucleus sampling value.
        api_key (str): Your Google API key. If None, expects env variable GOOGLE_API_KEY.

    Returns:
        genai.GenerativeModel: Configured Gemini client.
    """
    
    if api_key is None:
        import os
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("API key must be provided either via argument or GOOGLE_API_KEY env variable")

    # Configure client
    genai.configure(api_key=api_key)

    # Initialize model
    model_client = genai.GenerativeModel(
        model,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
        },
    )

    return model_client

