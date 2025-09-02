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







from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://ashish:Radhey@123@cluster0.3uxl669.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# Test Ollama connection endpoint in main if needed
# @app.get("/ollama/test", response_model=StandardResponse)
# async def test_ollama_connection():
#     """Test Ollama connection and model availability"""
#     try:
#         import ollama
        
#         # Try to create client with explicit host (adjust if needed)
#         client = ollama.Client(host='http://localhost:11434')
        
#         # Test basic connection first
#         try:
#             models = client.list()
            
#             # Handle the models response properly
#             available_models = []
#             if hasattr(models, 'models'):
#                 # Handle case where models is an object with models attribute
#                 model_list = models.models
#             elif isinstance(models, dict) and 'models' in models:
#                 # Handle case where models is a dict with models key
#                 model_list = models['models']
#             else:
#                 # Handle other cases
#                 model_list = models if isinstance(models, list) else [models]
            
#             # Extract model names
#             for model in model_list:
#                 if hasattr(model, 'model'):
#                     # Handle model objects with model attribute
#                     available_models.append(model.model)
#                 elif hasattr(model, 'name'):
#                     # Handle model objects with name attribute
#                     available_models.append(model.name)
#                 elif isinstance(model, dict):
#                     # Handle dict models
#                     name = model.get('model') or model.get('name') or model.get('id')
#                     if name:
#                         available_models.append(name)
#                 else:
#                     # Fallback to string representation
#                     available_models.append(str(model))
#         except Exception as list_error:
#             return StandardResponse(
#                 success=False,
#                 message="Failed to list models from Ollama",
#                 data={
#                     "error": str(list_error),
#                     "suggestions": [
#                         "Check if Ollama is running: ollama serve",
#                         "Verify Ollama is accessible at http://localhost:11434",
#                         "Try: curl http://localhost:11434/api/tags"
#                     ]
#                 }
#             )
        
#         # Test llama3.1:8b specifically
#         model_available = "llama3.1:8b" in available_models
        
#         if model_available:
#             try:
#                 # Test generation with timeout
#                 response = client.generate(
#                     model="llama3.1:8b",
#                     prompt="who was the first president of the united states?",
#                     options={"num_predict": 500}
#                 )

#                 print(f"Test response: {response['response']}")
                
#                 return StandardResponse(
#                     success=True,
#                     message="Ollama connection successful",
#                     data={
#                         "available_models": available_models,
#                         "target_model": "llama3.1:8b",
#                         "model_status": "available",
#                         "test_response": response['response']
#                     }
#                 )
#             except Exception as gen_error:
#                 return StandardResponse(
#                     success=False,
#                     message="Model found but generation failed",
#                     data={
#                         "available_models": available_models,
#                         "target_model": "llama3.1:8b",
#                         "model_status": "available_but_failed",
#                         "error": str(gen_error),
#                         "suggestion": "Model may be corrupted, try: ollama pull llama3.1:8b"
#                     }
#                 )
#         else:
#             return StandardResponse(
#                 success=False,
#                 message="llama3.1:8b model not found",
#                 data={
#                     "available_models": available_models,
#                     "target_model": "llama3.1:8b",
#                     "model_status": "not_found",
#                     "suggestion": "Run: ollama pull llama3.1:8b"
#                 }
#             )
            
#     except ImportError:
#         return StandardResponse(
#             success=False,
#             message="Ollama Python package not installed",
#             data={
#                 "error": "ImportError: ollama module not found",
#                 "suggestion": "Install with: pip install ollama"
#             }
#         )
#     except Exception as e:
#         return StandardResponse(
#             success=False,
#             message="Ollama connection failed",
#             data={
#                 "error": str(e),
#                 "error_type": type(e).__name__,
#                 "suggestions": [
#                     "Make sure Ollama is running: ollama serve",
#                     "Check if port 11434 is accessible",
#                     "Verify Ollama installation: ollama --version",
#                     "Try manual test: curl http://localhost:11434/api/tags"
#                 ]
#             }
#         )


