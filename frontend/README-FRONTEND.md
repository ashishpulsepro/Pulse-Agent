# PulsePro AI Chat Interface

A clean, minimalistic chat interface built with React and Tailwind CSS for interacting with the PulsePro AI Assistant.

## Features

- 🤖 **AI-Powered Conversations**: Natural language interface for site management
- 💬 **Real-time Chat**: Instant responses with typing indicators
- 🎨 **Clean Design**: Minimalistic UI using Tailwind CSS
- 📱 **Responsive**: Works on desktop, tablet, and mobile
- 🔄 **Auto-scroll**: Automatically scrolls to latest messages
- 📊 **Status Indicators**: Visual feedback for operation status
- 🔍 **Health Monitoring**: Real-time backend connection status

## Quick Start

1. **Install Dependencies**

   ```bash
   npm install
   ```

2. **Install Tailwind CSS** (if not already done)

   ```bash
   npm install -D tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```

3. **Start Development Server**

   ```bash
   npm run dev
   ```

4. **Open Browser**
   Navigate to `http://localhost:3000`

## Usage

### Basic Commands

- "Create a site in Mumbai"
- "Show me all sites"
- "Delete Delhi office"
- "Proceed" (to execute operations)

### Chat Flow

1. Type your request in natural language
2. AI will ask for any missing information
3. When ready, AI will ask for confirmation
4. Type "Proceed" to execute the operation

## Architecture

```
src/
├── components/
│   ├── ChatInterface.jsx     # Main chat component
│   └── StatusIndicator.jsx   # Backend health indicator
├── services/
│   └── api.js               # API communication layer
├── utils/
│   └── chatUtils.js         # Utility functions
└── App.jsx                  # Main application component
```

## API Integration

The frontend connects to the backend API at `http://localhost:8000` with these endpoints:

- `POST /chat` - Send messages to AI
- `GET /chat/health` - Check chat system status
- `GET /health` - General health check

## Styling

Built with Tailwind CSS for:

- Consistent design system
- Responsive layouts
- Clean, professional appearance
- Easy customization

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Environment Setup

Make sure your backend is running on `http://localhost:8000` before starting the frontend.

## Backend Connection

The interface automatically monitors backend health and displays connection status in the header. Green indicates all systems are operational.

## Contributing

1. Follow React best practices
2. Use Tailwind CSS for styling
3. Keep components small and focused
4. Add proper error handling
5. Test with the backend API
