# PulsePro User Onboarding System

A comprehensive 3-step user onboarding flow for PulsePro that guides new users through setting up their account with sites, users, and templates.

## 🎯 Overview

The onboarding system implements a structured flow:

1. **Welcome & Demo Site Creation** - Automatic demo site setup
2. **Step 1: Create Sites** - User creates additional sites/locations
3. **Step 2: Add Users** - User adds team members with permissions
4. **Step 3: Create Templates** - Template creation (placeholder for future implementation)

## 🚀 Features

### ✅ Completed Features

- **Guided 3-Step Flow**: Structured onboarding process
- **Demo Site Creation**: Automatic demo site for new users
- **Site Management**: Create multiple sites with simple name-only creation
- **User Management**: Add users with permission sets via API integration
- **Session Management**: Persistent session state across the flow
- **Error Handling**: Comprehensive error handling and recovery
- **API Integration**: Full integration with existing PulsePro APIs
- **Statistics**: Track onboarding completion rates and metrics

### 🔄 Template System (Placeholder)

Template creation is currently a placeholder that will be implemented with the full template logic in future updates.

## 📁 File Structure

```
backend/
├── main1.py              # Main onboarding API (NEW)
├── test_onboarding.py    # Test script for onboarding flow (NEW)
├── site_manager.py       # Site management logic (REFERENCE)
├── tree_system.py        # Tree-based intent system (REFERENCE)
├── ai_agent.py          # AI agent logic (REFERENCE)
└── main.py              # Original main API (REFERENCE)
```

## 🛠️ Setup & Installation

### Prerequisites

1. Python 3.8+
2. Required packages (install from requirements.txt)
3. PulsePro API access with valid refresh token
4. Environment variables configured

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export refresh="your_refresh_token_here"

# Run the onboarding API
python main1.py
```

The API will start on `http://localhost:8001`

## 📖 API Documentation

### Core Endpoints

#### Start Onboarding
```http
POST /onboarding/start
Content-Type: application/json

{
  "user_id": "optional_user_id",
  "session_id": "optional_session_id"
}
```

#### Perform Actions
```http
POST /onboarding/action
Content-Type: application/json

{
  "action": "create_site|add_users|move_to_users|etc",
  "session_id": "session_id",
  "data": {}
}
```

#### Create Site
```http
POST /onboarding/create-site
Content-Type: application/json

{
  "site_name": "Mumbai Office",
  "session_id": "session_id"
}
```

#### Create User
```http
POST /onboarding/create-user
Content-Type: application/json

{
  "first_name": "John",
  "last_name": "Doe", 
  "email": "john@company.com",
  "permission_set_name": "Admin",
  "session_id": "session_id"
}
```

### Utility Endpoints

- `GET /onboarding/session/{session_id}` - Get session details
- `GET /onboarding/stats` - Get onboarding statistics
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## 🔄 Onboarding Flow

### Step 1: Welcome
```json
{
  "message": "Welcome to PulsePro! Let's set you up with these 3 easy steps...",
  "current_step": "welcome",
  "options": ["create_site", "add_users", "view_demo_site"],
  "data": {"demo_site_created": true}
}
```

### Step 2: Site Creation
```json
{
  "message": "Great! Let's create a new site. What would you like to name it?",
  "current_step": "site_creation_input",
  "data": {"awaiting_input": "site_name"}
}
```

### Step 3: User Addition
```json
{
  "message": "Let's add a new user! I'll need a few details...",
  "current_step": "user_creation_input", 
  "data": {"awaiting_input": "first_name", "permission_sets": [...]}
}
```

### Step 4: Templates (Placeholder)
```json
{
  "message": "Templates allow you to create reusable checklists...",
  "current_step": "templates_intro",
  "data": {"templates_available": false}
}
```

### Step 5: Completion
```json
{
  "message": "Congratulations! Onboarding Complete!",
  "current_step": "completed",
  "data": {
    "onboarding_completed": true,
    "summary": {
      "sites_created": 3,
      "users_added": 2,
      "templates_created": 0
    }
  }
}
```

## 🧪 Testing

### Run Complete Demo
```bash
python test_onboarding.py
```

### Run Simple Test
```bash
python test_onboarding.py simple
```

### Test Output Example
```
🎯 PulsePro Onboarding Flow Demonstration
==================================================

1. Health Check...
✅ API Health: healthy

2. Starting Onboarding...
🚀 Starting Onboarding Process...
✅ Onboarding started! Session ID: 12345...

3. Creating Additional Sites...
🏢 Creating site: Mumbai Office
✅ Site created successfully!

4. Adding Users...
👥 Creating user: John Doe (john.doe@company.com)
✅ User created successfully!

🎉 Onboarding Demo Completed!
```

## 📊 Session Management

The system maintains session state including:

```json
{
  "session_id": "uuid",
  "current_step": "welcome|site_creation|users_intro|etc", 
  "steps_completed": ["welcome", "site_creation"],
  "created_sites": [
    {
      "name": "Mumbai Office",
      "id": 123,
      "created_at": "2024-01-01T10:00:00"
    }
  ],
  "created_users": [
    {
      "first_name": "John",
      "last_name": "Doe",
      "email": "john@company.com",
      "permission_set": "Admin",
      "created_at": "2024-01-01T10:05:00"
    }
  ],
  "demo_site_created": true,
  "demo_site": {
    "name": "Demo Site - 20240101",
    "id": 122,
    "created_at": "2024-01-01T09:55:00"
  }
}
```

## 🔗 API Integration

### Site Creation
Uses `site_manager.create_site_by_name_only()` for simple site creation with just a name.

### User Creation  
Uses `user_manager.create_user()` with permission sets from `permission_manager.get_all_permission_sets()`.

### API Endpoints Used
- `POST /customer/save_loc_by_only_name/` - Create site by name
- `POST /customer/add_team_member/` - Create user
- `GET /customer/get_all_permission_bundles/` - Get permission sets

## 📈 Statistics & Monitoring

Access onboarding statistics:
```bash
curl http://localhost:8001/onboarding/stats
```

Response:
```json
{
  "success": true,
  "stats": {
    "total_sessions": 10,
    "completed_sessions": 8,
    "completion_rate": 80.0,
    "total_sites_created": 25,
    "total_users_created": 18,
    "active_sessions": 2
  }
}
```

## 🚦 Error Handling

The system includes comprehensive error handling:

- **API Failures**: Graceful degradation when external APIs fail
- **Validation Errors**: Clear error messages for invalid input
- **Session Recovery**: Ability to resume interrupted sessions
- **Timeout Handling**: Session cleanup for inactive sessions

## 🎨 Frontend Integration

The API is designed to work with any frontend framework. Example React integration:

```javascript
// Start onboarding
const response = await fetch('/onboarding/start', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({})
});

const data = await response.json();
// Display data.message to user
// Render options based on data.options
```

## 🔧 Configuration

### Environment Variables
```bash
refresh=your_refresh_token_here
OLLAMA_HOST=http://localhost:11434  # Optional for AI features
```

### API Configuration
- **Port**: 8001 (configurable)
- **Host**: 0.0.0.0 (configurable)
- **Reload**: True (development)

## 🛣️ Future Enhancements

1. **Template System**: Full implementation of template creation
2. **Progress Tracking**: Visual progress indicators
3. **Customizable Flows**: Admin-configurable onboarding steps
4. **Integration Testing**: Automated integration test suite
5. **Analytics Dashboard**: Real-time onboarding analytics
6. **Multi-language Support**: Internationalization
7. **Email Notifications**: Welcome emails and confirmations

## 📞 Support

For questions or issues:
1. Check the API documentation at `/docs`
2. Run health checks at `/health`
3. Review session state at `/onboarding/session/{session_id}`
4. Check logs for detailed error information

## 📄 License

This onboarding system is part of the PulsePro platform.

---

**Built with ❤️ for seamless user onboarding**
