#!/usr/bin/env python3

import requests
import json
import asyncio
from typing import Dict, Any

async def test_ollama_connection():
    """Test Ollama connection and model availability"""
    
    base_url = "http://localhost:11434"
    model = "llama3.1:8b"
    
    print("Testing Ollama connection...")
    
    try:
        # 1. Check if server is running
        print("1. Checking Ollama server status...")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        
        # 2. List available models
        models_data = response.json()
        available_models = [model["name"] for model in models_data.get("models", [])]
        print(f"   Available models: {available_models}")
        
        # 3. Check if target model exists
        if model in available_models:
            print(f"   ✓ Model '{model}' is available")
        else:
            print(f"   ✗ Model '{model}' not found!")
            return False
            
        # 4. Test a simple generation
        print("2. Testing model generation...")
        test_payload = {
            "model": model,
            "prompt": "Say 'Hello, I am working!' in exactly those words.",
            "stream": False
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=test_payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"   Response: {result.get('response', 'No response')}")
        print(f"   ✓ Model generation successful!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("   ✗ Cannot connect to Ollama server. Is it running?")
        return False
    except requests.exceptions.Timeout:
        print("   ✗ Request timed out. Server might be busy.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Request error: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return False

def test_sync_ollama():
    """Synchronous version for quick testing"""
    return asyncio.run(test_ollama_connection())

if __name__ == "__main__":
    success = test_sync_ollama()
    
    if success:
        print("\n🎉 Ollama is working perfectly!")
        print("You can now use your FastAPI integration.")
    else:
        print("\n❌ Ollama test failed. Check the errors above.")