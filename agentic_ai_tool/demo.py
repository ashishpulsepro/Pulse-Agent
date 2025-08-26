#!/usr/bin/env python3
"""
Simple demo script for the Agentic AI Tool
Demonstrates how to interact with the AI agent programmatically
"""

import asyncio
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from core_agent import CoreAIAgent

async def demo_agent():
    """Demonstrate the AI agent capabilities"""
    print("🤖 PulsePro Agentic AI Tool Demo")
    print("=" * 40)
    
    # Initialize the agent (without real auth for demo)
    agent = CoreAIAgent(auth_manager=None)
    
    # Demo conversations
    demo_requests = [
        "Hello! What can you help me with?",
        "Create a site called Mumbai Tech Park",
        "Show me all my sites", 
        "Help me understand what you can do",
        "Assign John Smith to Mumbai Tech Park",
        "Who has access to Mumbai Tech Park?",
    ]
    
    session_id = None
    
    for i, request in enumerate(demo_requests, 1):
        print(f"\n{i}. User: {request}")
        print("-" * 30)
        
        # Process the request
        response = await agent.process_request(request, session_id)
        
        # Use the session ID for context
        if not session_id:
            session_id = response.session_id
        
        # Display the response
        print(f"AI: {response.message}")
        print(f"Status: {response.status.value}")
        
        if response.actions_taken:
            print(f"Actions: {', '.join(response.actions_taken)}")
        
        print()
    
    # Show session stats
    print("📊 Session Summary")
    print("-" * 20)
    history = agent.get_session_history(session_id)
    print(f"Total messages in session: {len(history)}")
    print(f"Session ID: {session_id}")

if __name__ == "__main__":
    asyncio.run(demo_agent())
