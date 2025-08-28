#!/usr/bin/env python3
"""
Quick setup script to switch to Gemini AI
"""

import os
import sys
from pathlib import Path

def main():
    print("🔥 PulsePro AI Switch Setup - Gemini Integration")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        with open(".env", "w") as f:
            f.write("# Gemini API Configuration\n")
            f.write("GEMINI_API_KEY=\n\n")
            f.write("# PulsePro API Configuration\n")
            f.write("refresh=\n")
        print("✅ Created .env file")
    else:
        print("📁 .env file already exists")
    
    # Check for Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n🔑 GEMINI API KEY SETUP REQUIRED")
        print("=" * 40)
        print("1. Go to: https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Copy the API key")
        print("4. Add it to your .env file: GEMINI_API_KEY=your_key_here")
        print("\nOr set it as environment variable:")
        print("export GEMINI_API_KEY=your_key_here")
        
        print("\n⚠️  Without the API key, the system will fallback to Ollama")
    else:
        print("✅ Gemini API key found in environment")
    
    print(f"\n🚀 QUICK START")
    print("=" * 30)
    print("1. Set your GEMINI_API_KEY in .env file")
    print("2. cd backend")
    print("3. python main.py")
    print("4. cd ../frontend && npm run dev")
    
    print(f"\n🔧 Your setup:")
    print(f"   • Backend: /Users/neerajwasan/Desktop/Pulse-Agent/backend")
    print(f"   • Frontend: /Users/neerajwasan/Desktop/Pulse-Agent/frontend")
    print(f"   • Config: /Users/neerajwasan/Desktop/Pulse-Agent/.env")
    
    # Test import
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI package installed")
    except ImportError:
        print("❌ Google Generative AI package not found")
        print("   Run: pip install google-generativeai")
    
    print("\n🎯 The system will automatically use Gemini if API key is available,")
    print("   otherwise it falls back to Ollama.")

if __name__ == "__main__":
    main()
