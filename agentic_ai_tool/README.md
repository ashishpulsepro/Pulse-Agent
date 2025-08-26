# PulsePro Agentic AI Tool

A structured, intelligent AI agent system for site and user management that understands natural language and performs complex operations through a clean, conversational interface.

## Features

### 🤖 Intelligent Agent Core

- **Natural Language Understanding**: Processes user requests in plain English
- **Intent Analysis**: Accurately identifies what users want to accomplish
- **Context Awareness**: Maintains conversation context across interactions
- **Action Execution**: Performs complex operations through structured tools

### 🏢 Site Management

- **Create Sites**: "Create a site called Mumbai Office in India"
- **List Sites**: "Show me all my sites"
- **Site Details**: "Get details for the Delhi office"
- **Update Sites**: "Update the address for Mumbai office"
- **Delete Sites**: "Remove the Bangalore office"

### 👥 User Management

- **Assign Users**: "Assign John Smith to the Mumbai office"
- **Remove Users**: "Remove Jane from the Delhi office"
- **List Site Users**: "Who has access to the Mumbai office?"

### 💬 Conversational Interface

- **Smart Chat UI**: Modern, responsive chat interface
- **Status Indicators**: Real-time action status and progress
- **Session Management**: Persistent conversations with context
- **Error Handling**: Graceful error recovery and user guidance

## Architecture

### Core Components

```
agentic_ai_tool/
├── core_agent.py          # Main AI agent orchestrator
├── intent_analyzer.py     # Natural language understanding
├── action_executor.py     # Action coordination and execution
├── response_generator.py  # Natural language response generation
├── session_manager.py     # Conversation context management
├── tools/                 # Specialized operation tools
│   ├── site_tool.py      # Site management operations
│   └── user_tool.py      # User management operations
├── frontend/             # React chat interface
├── server.py            # FastAPI backend server
├── config.py           # Configuration management
└── launch.py           # Launcher script
```

### Design Principles

1. **Modularity**: Each component has a specific responsibility
2. **Extensibility**: Easy to add new tools and capabilities
3. **Reliability**: Robust error handling and recovery
4. **User-Friendly**: Natural language interface with clear feedback
5. **Maintainability**: Clean, well-documented code structure

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- Access to PulsePro API

### Backend Setup

1. **Install Python dependencies**:

```bash
cd agentic_ai_tool
pip install -r requirements.txt
```

2. **Set up environment variables**:

```bash
# Create .env file in the project root
echo "refresh=your_refresh_token_here" >> ../.env
```

3. **Start the AI agent server**:

```bash
python launch.py
```

### Frontend Integration

The frontend component can be integrated into your existing React application:

```jsx
import AgenticChatInterface from "./agentic_ai_tool/frontend/AgenticChatInterface.jsx";

function App() {
  return (
    <div className="App">
      <AgenticChatInterface />
    </div>
  );
}
```

## Usage Examples

### Site Operations

```
User: "Create a new site called Tech Park Mumbai"
AI: "Great! I've successfully created the site 'Tech Park Mumbai'. The site ID is 123."

User: "Show me all my sites"
AI: "Here are your sites:

1. **Tech Park Mumbai**
   Address: Address in Mumbai | ID: 123

2. **Delhi Office**
   Address: Connaught Place, Delhi | ID: 124"

User: "Delete the Mumbai site"
AI: "I've successfully deleted the site 'Tech Park Mumbai'."
```

### User Management

```
User: "Assign John to the Delhi office"
AI: "I've successfully assigned John to the site 'Delhi Office'."

User: "Who has access to Delhi office?"
AI: "Here are the users assigned to 'Delhi Office':

1. **John Smith**
   Email: john@company.com | Role: Manager

2. **Sarah Johnson**
   Email: sarah@company.com | Role: Employee"
```

### Smart Assistance

```
User: "Help me manage my sites"
AI: "I can help you with the following:

**Site Management:**
• Create a new site: 'Create a site called Office Mumbai'
• List all sites: 'Show me all sites'
• Get site details: 'Show me details of Mumbai office'
• Update a site: 'Update the Delhi office site'
• Delete a site: 'Delete the Bangalore office'

**User Management:**
• Assign user to site: 'Assign John to Mumbai office'
• Remove user from site: 'Remove Jane from Delhi office'
• List site users: 'Show users for Mumbai office'

Just tell me what you'd like to do in natural language!"
```

## API Endpoints

### Chat Interface

- `POST /chat` - Send message to AI agent
- `GET /chat/sessions/{session_id}/history` - Get conversation history
- `DELETE /chat/sessions/{session_id}` - Clear session

### Health & Status

- `GET /health` - Server health check
- `GET /agent/health` - Agent status and capabilities
- `GET /agent/capabilities` - Available actions and features

### Development

- `POST /agent/reinitialize` - Restart agent (development)
- `GET /debug/sessions` - View active sessions (development)

## Configuration

### Environment Variables

```bash
# API Configuration
PULSE_PRO_API_BASE_URL=https://staging-api.pulsepro.ai
PULSE_PRO_FRONTEND_URL=https://staging.pulsepro.ai

# Authentication
refresh=your_refresh_token

# Server Configuration
HOST=127.0.0.1
PORT=8000

# Session Management
SESSION_TIMEOUT_HOURS=24
MAX_SESSIONS=1000

# Development
LOG_LEVEL=INFO
ENABLE_DEBUG_ENDPOINTS=true
```

### Launcher Options

```bash
# Start with default settings
python launch.py

# Custom host/port
python launch.py --host 0.0.0.0 --port 9000

# Production mode (no auto-reload)
python launch.py --no-reload

# Run health checks only
python launch.py --check-only

# Debug mode
python launch.py --log-level DEBUG
```

## Development

### Adding New Tools

1. Create a new tool class in `tools/`:

```python
class NewTool:
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager

    async def perform_action(self, parameters):
        # Implementation
        return ExecutionResult(success=True, message="Done!")

    def get_available_actions(self):
        return ['new_action']
```

2. Register the tool in `core_agent.py`:

```python
self.new_tool = NewTool(auth_manager)
self.action_executor.register_tool('new', self.new_tool)
```

3. Add intent patterns in `intent_analyzer.py`:

```python
IntentType.NEW_ACTION: [
    r"do something new with (.+)",
    r"perform new action (.+)"
]
```

### Testing

```bash
# Run health checks
python launch.py --check-only

# Test specific endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

## Error Handling

The system includes comprehensive error handling:

- **API Errors**: Graceful handling of backend API failures
- **Authentication Issues**: Clear messaging for auth problems
- **Network Errors**: Retry logic and user-friendly error messages
- **Invalid Requests**: Input validation and helpful suggestions
- **Server Errors**: Detailed logging and fallback responses

## Security Considerations

- Environment-based configuration for sensitive data
- Request validation and sanitization
- Session management with timeouts
- CORS protection for frontend integration
- Error messages that don't expose internal details

## Performance

- Efficient session management with automatic cleanup
- Async/await pattern for non-blocking operations
- Configurable limits for sessions and message history
- Optimized API calls with proper error handling

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive error handling
3. Include docstrings and type hints
4. Test new features thoroughly
5. Update documentation as needed

## License

This project is part of the PulsePro system and follows the same licensing terms.
