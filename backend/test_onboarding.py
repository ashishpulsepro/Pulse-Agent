"""
test_onboarding.py - Test script for the onboarding flow
Demonstrates the complete 3-step onboarding process
"""

import requests
import json
import time
from typing import Dict, Any

class OnboardingTester:
    """Test class for the onboarding flow"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session_id = None
    
    def start_onboarding(self) -> Dict[str, Any]:
        """Start the onboarding process"""
        print("🚀 Starting Onboarding Process...")
        
        url = f"{self.base_url}/onboarding/start"
        response = requests.post(url, json={})
        
        if response.status_code == 200:
            data = response.json()
            self.session_id = data['session_id']
            print(f"✅ Onboarding started! Session ID: {self.session_id}")
            print(f"📝 Message: {data['message']}")
            return data
        else:
            print(f"❌ Failed to start onboarding: {response.text}")
            return {}
    
    def perform_action(self, action: str) -> Dict[str, Any]:
        """Perform an onboarding action"""
        if not self.session_id:
            print("❌ No active session. Start onboarding first.")
            return {}
        
        print(f"🎯 Performing action: {action}")
        
        url = f"{self.base_url}/onboarding/action"
        payload = {
            "action": action,
            "session_id": self.session_id
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Action successful!")
            print(f"📝 Message: {data['message']}")
            return data
        else:
            print(f"❌ Action failed: {response.text}")
            return {}
    
    def create_site(self, site_name: str) -> Dict[str, Any]:
        """Create a site during onboarding"""
        if not self.session_id:
            print("❌ No active session. Start onboarding first.")
            return {}
        
        print(f"🏢 Creating site: {site_name}")
        
        url = f"{self.base_url}/onboarding/create-site"
        payload = {
            "site_name": site_name,
            "session_id": self.session_id
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Site created successfully!")
            print(f"📝 Message: {data['message']}")
            return data
        else:
            print(f"❌ Site creation failed: {response.text}")
            return {}
    
    def create_user(self, first_name: str, last_name: str, email: str, permission_set: str = "Admin") -> Dict[str, Any]:
        """Create a user during onboarding"""
        if not self.session_id:
            print("❌ No active session. Start onboarding first.")
            return {}
        
        print(f"👥 Creating user: {first_name} {last_name} ({email})")
        
        url = f"{self.base_url}/onboarding/create-user"
        payload = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "permission_set_name": permission_set,
            "session_id": self.session_id
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ User created successfully!")
            print(f"📝 Message: {data['message']}")
            return data
        else:
            print(f"❌ User creation failed: {response.text}")
            return {}
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        if not self.session_id:
            print("❌ No active session.")
            return {}
        
        url = f"{self.base_url}/onboarding/session/{self.session_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get session info: {response.text}")
            return {}
    
    def check_health(self) -> bool:
        """Check if the onboarding API is healthy"""
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Health: {data['status']}")
                return data['status'] == 'healthy'
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

def run_complete_onboarding_demo():
    """Run a complete onboarding demonstration"""
    print("🎯 PulsePro Onboarding Flow Demonstration")
    print("=" * 50)
    
    tester = OnboardingTester()
    
    # Check health first
    print("\n1. Health Check...")
    if not tester.check_health():
        print("❌ API is not healthy. Please check the server.")
        return
    
    # Start onboarding
    print("\n2. Starting Onboarding...")
    start_result = tester.start_onboarding()
    if not start_result:
        return
    
    time.sleep(1)
    
    # Step 1: Create additional sites
    print("\n3. Creating Additional Sites...")
    
    # Create first site
    tester.perform_action("create_site")
    time.sleep(0.5)
    tester.create_site("Mumbai Office")
    time.sleep(1)
    
    # Create second site
    tester.perform_action("create_another_site")
    time.sleep(0.5)
    tester.create_site("Delhi Branch")
    time.sleep(1)
    
    # Move to users
    tester.perform_action("move_to_users")
    time.sleep(1)
    
    # Step 2: Add users
    print("\n4. Adding Users...")
    
    # Add first user
    tester.perform_action("add_user")
    time.sleep(0.5)
    tester.create_user("John", "Doe", "john.doe@company.com", "Admin")
    time.sleep(1)
    
    # Add second user
    tester.perform_action("add_another_user")
    time.sleep(0.5)
    tester.create_user("Jane", "Smith", "jane.smith@company.com", "Manager")
    time.sleep(1)
    
    # Move to templates
    tester.perform_action("move_to_templates")
    time.sleep(1)
    
    # Step 3: Templates (placeholder)
    print("\n5. Template Creation (Placeholder)...")
    tester.perform_action("finish_onboarding")
    
    # Get final session info
    print("\n6. Final Session Summary...")
    session_info = tester.get_session_info()
    if session_info and session_info.get('success'):
        session = session_info['session']
        print(f"📊 Onboarding Summary:")
        print(f"   - Sites Created: {len(session.get('created_sites', [])) + (1 if session.get('demo_site_created') else 0)}")
        print(f"   - Users Added: {len(session.get('created_users', []))}")
        print(f"   - Status: {session.get('current_step', 'unknown')}")
        print(f"   - Session ID: {session.get('session_id', 'unknown')}")
    
    print("\n🎉 Onboarding Demo Completed!")
    print("=" * 50)

def run_simple_onboarding_test():
    """Run a simple onboarding test"""
    print("🧪 Simple Onboarding Test")
    print("=" * 30)
    
    tester = OnboardingTester()
    
    # Check health
    if not tester.check_health():
        print("❌ API is not healthy.")
        return
    
    # Start onboarding
    tester.start_onboarding()
    
    # View demo site
    tester.perform_action("view_demo_site")
    
    # Create one site
    tester.perform_action("create_site")
    tester.create_site("Test Site")
    
    # Move to users
    tester.perform_action("move_to_users")
    
    # Add one user
    tester.perform_action("add_user")
    tester.create_user("Test", "User", "test@example.com")
    
    # Finish onboarding
    tester.perform_action("move_to_templates")
    tester.perform_action("finish_onboarding")
    
    print("\n✅ Simple test completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        run_simple_onboarding_test()
    else:
        run_complete_onboarding_demo()
        
    print("\n💡 Tips:")
    print("- Run 'python test_onboarding.py simple' for a quick test")
    print("- Check http://localhost:8001/docs for API documentation")
    print("- View http://localhost:8001/onboarding/stats for statistics")
