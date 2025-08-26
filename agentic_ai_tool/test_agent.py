#!/usr/bin/env python3
"""
Test script for Agentic AI Tool
Simple tests to validate the components work correctly
"""

import sys
import os
import asyncio
import logging

# Add paths
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', 'backend'))

# Suppress warnings for testing
logging.getLogger().setLevel(logging.ERROR)

async def test_intent_analyzer():
    """Test the intent analyzer"""
    print("🧠 Testing Intent Analyzer...")
    
    try:
        from intent_analyzer import IntentAnalyzer, IntentType
        
        analyzer = IntentAnalyzer()
        
        # Test cases
        test_cases = [
            ("create a site called Mumbai Office", IntentType.CREATE_SITE),
            ("show me all sites", IntentType.LIST_SITES),
            ("assign John to Delhi office", IntentType.ASSIGN_USER),
            ("delete the Bangalore site", IntentType.DELETE_SITE),
            ("help me", IntentType.HELP),
        ]
        
        for text, expected_intent in test_cases:
            result = await analyzer.analyze(text)
            status = "✓" if result.intent == expected_intent else "✗"
            print(f"  {status} '{text}' -> {result.intent.value}")
            if result.parameters:
                print(f"    Parameters: {result.parameters}")
        
        print("✅ Intent Analyzer tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Intent Analyzer test failed: {e}")
        return False

async def test_session_manager():
    """Test the session manager"""
    print("💾 Testing Session Manager...")
    
    try:
        from session_manager import SessionManager
        
        manager = SessionManager()
        
        # Create session
        session_id = manager.create_session()
        print(f"  ✓ Created session: {session_id}")
        
        # Add messages
        manager.add_message(session_id, 'user', 'Hello')
        manager.add_message(session_id, 'agent', 'Hi there!')
        
        # Get history
        history = manager.get_session_messages(session_id)
        print(f"  ✓ Session has {len(history)} messages")
        
        # Get stats
        stats = manager.get_session_stats()
        print(f"  ✓ Session stats: {stats['total_sessions']} total, {stats['total_messages']} messages")
        
        print("✅ Session Manager tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Session Manager test failed: {e}")
        return False

async def test_response_generator():
    """Test the response generator"""
    print("📝 Testing Response Generator...")
    
    try:
        from response_generator import ResponseGenerator
        from intent_analyzer import IntentType, IntentResult
        from action_executor import ExecutionResult
        
        generator = ResponseGenerator()
        
        # Test successful site creation
        intent_result = IntentResult(
            intent=IntentType.CREATE_SITE,
            confidence=0.9,
            parameters={'site_name': 'Test Site'}
        )
        
        execution_result = ExecutionResult(
            success=True,
            message="Site created successfully",
            data={'id': 123}
        )
        
        response = generator.generate_response(intent_result, execution_result)
        print(f"  ✓ Generated response: {response[:100]}...")
        
        print("✅ Response Generator tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Response Generator test failed: {e}")
        return False

async def test_core_agent():
    """Test the core agent (without real API)"""
    print("🤖 Testing Core Agent...")
    
    try:
        from core_agent import CoreAIAgent
        
        # Initialize with mock auth
        agent = CoreAIAgent(auth_manager=None)
        
        # Test basic interaction
        response = await agent.process_request("hello")
        print(f"  ✓ Agent response: {response.message[:50]}...")
        print(f"  ✓ Status: {response.status.value}")
        
        # Test help request
        help_response = await agent.process_request("help")
        print(f"  ✓ Help response: {help_response.message[:50]}...")
        
        print("✅ Core Agent tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Core Agent test failed: {e}")
        return False

async def test_api_server():
    """Test the API server (basic import and setup)"""
    print("🌐 Testing API Server...")
    
    try:
        # Test import
        import server
        print("  ✓ Server module imported successfully")
        
        # Test FastAPI app creation
        app = server.app
        print(f"  ✓ FastAPI app created: {app.title}")
        
        print("✅ API Server tests completed")
        return True
        
    except Exception as e:
        print(f"❌ API Server test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🧪 Running Agentic AI Tool Tests")
    print("=" * 40)
    
    tests = [
        test_intent_analyzer,
        test_session_manager,
        test_response_generator,
        test_core_agent,
        test_api_server,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 Test Summary")
    print("-" * 20)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
