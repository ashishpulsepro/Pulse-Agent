#!/usr/bin/env python3
"""
Agentic AI Tool Launcher
Simple launcher script to start the AI agent server
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / 'backend'))

from config import config

def setup_logging(level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import requests
        import pydantic
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment variables and configuration"""
    issues = []
    
    if not config.REFRESH_TOKEN:
        issues.append("REFRESH_TOKEN environment variable not set")
    
    if issues:
        print("⚠️  Environment issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nSome features may not work properly.")
    else:
        print("✓ Environment configuration looks good")
    
    return len(issues) == 0

def start_server(host=None, port=None, reload=True):
    """Start the AI agent server"""
    host = host or config.SERVER_HOST
    port = port or config.SERVER_PORT
    
    print(f"🚀 Starting Agentic AI Tool server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print(f"   Environment: {'Production' if not reload else 'Development'}")
    print()
    
    try:
        import uvicorn
        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            reload=reload,
            log_level=config.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def run_health_check():
    """Run a health check on the server"""
    import requests
    import time
    
    print("🔍 Running health check...")
    
    try:
        # Wait a moment for server to start
        time.sleep(2)
        
        response = requests.get(f"http://{config.SERVER_HOST}:{config.SERVER_PORT}/health")
        if response.status_code == 200:
            data = response.json()
            print("✓ Server is healthy")
            print(f"   Status: {data.get('status')}")
            print(f"   Agent Available: {data.get('agent_available')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Agentic AI Tool Launcher")
    parser.add_argument("--host", default=None, help="Server host (default: from config)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: from config)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--check-only", action="store_true", help="Only run checks, don't start server")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🤖 PulsePro Agentic AI Tool Launcher")
    print("=" * 40)
    
    # Run checks
    print("Running pre-flight checks...")
    deps_ok = check_dependencies()
    env_ok = check_environment()
    
    if not deps_ok:
        print("\n❌ Dependency check failed. Please fix the issues above.")
        return 1
    
    print()
    
    if args.check_only:
        print("✅ Checks completed. Use without --check-only to start the server.")
        return 0
    
    # Validate configuration
    config.validate()
    
    # Start server
    success = start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
