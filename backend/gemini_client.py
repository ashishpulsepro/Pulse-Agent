"""
Gemini client wrapper that mimics Ollama interface for easy switching
"""

import google.generativeai as genai
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    """Gemini client that mimics Ollama's interface"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model (using the latest available model)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        logger.info("Gemini client initialized successfully")
    
    def generate(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate response using Gemini, mimicking Ollama's interface
        
        Args:
            model: Model name (ignored for Gemini, using gemini-pro)
            prompt: The prompt text
            options: Generation options (temperature, max_tokens, etc.)
        
        Returns:
            Dict with 'response' key containing the generated text
        """
        try:
            # Extract options
            options = options or {}
            temperature = options.get('temperature', 0.7)
            max_tokens = options.get('num_predict', 1000)
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.8,
                top_k=40
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Return in Ollama-like format
            return {
                'response': response.text,
                'model': 'gemini-2.0-flash',
                'done': True
            }
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise Exception(f"Gemini generation failed: {str(e)}")
    
    def list(self) -> Dict[str, Any]:
        """
        List available models (mimicking Ollama interface)
        Returns a mock response for compatibility
        """
        return {
            'models': [
                {'model': 'gemini-2.0-flash', 'name': 'gemini-2.0-flash'}
            ]
        }
    
    def test_connection(self) -> bool:
        """Test if Gemini API is accessible"""
        try:
            # Simple test generation
            response = self.generate(
                model="gemini-2.0-flash",
                prompt="Say 'Hello, Gemini is working!'",
                options={"temperature": 0.1, "num_predict": 50}
            )
            return "Hello" in response.get('response', '')
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            return False
