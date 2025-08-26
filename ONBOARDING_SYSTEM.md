# PulsePro Complete Onboarding System

## Overview

This is a comprehensive onboarding flow system for PulsePro that guides users through setting up their workspace in 3 easy steps:

1. **Create Sites** 🏢 - Set up business locations
2. **Add Team Members** 👥 - Invite users and set permissions
3. **Create Templates** 📋 - Setup checklists and workflows

## Architecture

### Backend Components

#### 1. `main1.py` - Enhanced API Server

- **FastAPI application** with complete onboarding flow
- **Unified chat endpoint** supporting both normal operations and onboarding
- **Site management** endpoints for CRUD operations
- **User management** endpoints with permission handling
- **System status** and health monitoring

#### 2. `onboarding_system.py` - Core Onboarding Logic

- **OnboardingSystem class** - Main orchestrator
- **UserManager class** - Handles user creation and permissions
- **Step-by-step flow** with state management
- **Progress tracking** and session management
- **LLM integration** for natural language processing

#### 3. `site_manager.py` - Site Operations (Enhanced)

- Site CRUD operations
- Simple site creation by name only
- API integration with PulsePro backend
- Authentication and token management

#### 4. `tree_system.py` - Normal Operations (Existing)

- Tree-based intent processing
- Site management through conversation
- CREATE, READ, DELETE operations

### Frontend Components

#### 1. `OnboardingInterface.jsx` - Onboarding UI

- **Step-by-step guided interface** with progress tracking
- **Interactive chat** for onboarding conversations
- **Progress visualization** with step indicators
- **Summary panels** showing created sites and users
- **Responsive design** with modern UI

#### 2. `ChatInterface.jsx` - Normal Operations (Enhanced)

- Standard conversational interface
- Site management operations
- Clean, modern design

#### 3. `App.jsx` - Mode Switcher

- Toggle between normal chat and onboarding modes
- Unified interface management

#### 4. Enhanced API Service

- Complete API integration
- Onboarding-specific endpoints
- Site and user management methods

## Onboarding Flow Details

### Step 1: Greeting & Demo Site

```
🎉 Welcome to PulsePro!

Let's get you set up with these 3 easy steps:
- Step 1: Create a Site 🏢
- Step 2: Add Team Members 👥
- Step 3: Create Templates 📋

Creating demo site...
✅ Demo site "Demo Site - Welcome" created!
```

### Step 2: Site Creation Loop

```
Would you like to create a new site? (Yes/Skip)
→ User: "Yes"
What would you like to name your new site?
→ User: "Mumbai Office"
Perfect! I'll create "Mumbai Office". Proceed? (yes/no)
→ User: "yes"
✅ Site "Mumbai Office" created successfully!

Would you like to create another site? (Yes/Next)
```

### Step 3: User Creation Loop

```
Step 2: Add Team Members 👥
Would you like to add a team member? (Yes/Skip)
→ User: "Yes"
What's their first name?
→ User: "John"
What's their last name?
→ User: "Doe"
What's their email address?
→ User: "john@company.com"

User Details:
• Name: John Doe
• Email: john@company.com
• Permissions: Default User

Should I create this user? (yes/no)
→ User: "yes"
✅ User created successfully!

Would you like to add another team member? (Yes/Next)
```

### Step 4: Template Creation (Placeholder)

```
Step 3: Templates & Checklists 📋
Would you like to create a new template? (Yes/Finish)
→ User: "Finish"

🎉 Congratulations! Onboarding Complete!

Sites Created: 2
• Demo Site - Welcome
• Mumbai Office

Team Members Added: 1
• John Doe (john@company.com)

Welcome to PulsePro! 🚀
```

## API Endpoints

### Onboarding Endpoints

- `POST /onboarding/start` - Start onboarding process
- `POST /onboarding/chat` - Handle onboarding conversations
- `GET /onboarding/status/{session_id}` - Get onboarding status
- `DELETE /onboarding/session/{session_id}` - Clear onboarding session

### Unified Chat

- `POST /chat` - Unified endpoint supporting both modes
  - `mode: "normal"` - Standard operations
  - `mode: "onboarding"` - Onboarding flow

### Site Management

- `GET /sites` - List all sites
- `POST /sites` - Create site with full details
- `POST /sites/simple` - Create site with name only
- `DELETE /sites/{site_id}` - Delete site

### User Management

- `GET /users/permissions` - Get permission bundles
- `POST /users` - Create new user

### System

- `GET /health` - System health check
- `GET /status` - Detailed system status
- `GET /sessions` - Active sessions info

## Key Features

### 1. **Intelligent Conversation Flow**

- Natural language processing with LLM integration
- Fallback keyword matching when LLM unavailable
- Context-aware responses and data collection

### 2. **Progressive Onboarding**

- Step-by-step guidance with clear progress indicators
- Ability to skip steps or restart process
- Loops for creating multiple sites/users

### 3. **Real API Integration**

- Actual site creation using PulsePro staging API
- Real user creation with permission management
- Token-based authentication

### 4. **Modern UI/UX**

- Responsive design with gradient backgrounds
- Progress tracking with visual indicators
- Interactive chat interface with status badges
- Dark mode support

### 5. **Session Management**

- Individual session tracking
- State persistence during onboarding
- Ability to clear sessions and restart

### 6. **Error Handling**

- Comprehensive error handling and user feedback
- Graceful degradation when services unavailable
- Clear error messages and recovery options

## Configuration

### Environment Variables

```bash
# Required for API access
refresh=<your_refresh_token>
```

### API Configuration

- **Base URL**: `https://staging-api.pulsepro.ai`
- **Authentication**: JWT token-based
- **Default Permissions**: 1107 (configurable)

## Usage Instructions

### Running the System

1. **Start Backend**:

   ```bash
   cd backend
   python main1.py
   ```

2. **Start Frontend**:

   ```bash
   cd frontend
   npm start
   ```

3. **Access Application**:
   - Normal Chat: `http://localhost:3000` (default mode)
   - Onboarding: Click "🎉 Start Onboarding" button
   - API Docs: `http://localhost:8000/docs`

### Testing the Onboarding Flow

1. Open the application
2. Click "🎉 Start Onboarding"
3. Follow the guided conversation
4. Create sites and users as prompted
5. Complete the full onboarding process

### API Testing

```bash
# Start onboarding
curl -X POST http://localhost:8000/onboarding/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-session"}'

# Send onboarding message
curl -X POST http://localhost:8000/onboarding/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Yes", "session_id": "test-session"}'
```

## Future Enhancements

1. **Template Creation**: Full implementation of checklist/template creation
2. **Advanced Permissions**: More granular permission management
3. **Site Assignment**: Assign users to specific sites during onboarding
4. **Email Invitations**: Send actual email invites to new users
5. **Onboarding Analytics**: Track completion rates and user behavior
6. **Customizable Flow**: Admin-configurable onboarding steps
7. **Multi-language Support**: Internationalization for global users

## Technical Notes

### LLM Integration

- Uses Ollama with llama3.1:8b model
- Fallback to keyword matching if LLM unavailable
- Temperature set to 0.0 for consistent responses

### State Management

- Session-based state tracking
- Progress calculation based on completed steps
- Clean session cleanup after completion

### API Design

- RESTful endpoints with consistent response formats
- Comprehensive error handling and status codes
- OpenAPI documentation available at `/docs`

This complete onboarding system provides a smooth, guided experience for new PulsePro users while maintaining the flexibility of the existing chat-based site management system.
