import React, { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import OnboardingInterface from './components/OnboardingInterface';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('onboarding'); // Default to onboarding

  return (
    <div className="App">
      {/* Navigation Bar */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">PulsePro</h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">Site Management Platform</p>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="flex items-center space-x-1">
              <button
                onClick={() => setCurrentView('onboarding')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentView === 'onboarding'
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  <span>Onboarding</span>
                </div>
              </button>
              
              <button
                onClick={() => setCurrentView('chat')}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  currentView === 'chat'
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  <span>AI Chat</span>
                </div>
              </button>
            </div>

            {/* View Info */}
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {currentView === 'onboarding' ? (
                <span>🚀 3-Step Setup</span>
              ) : (
                <span>💬 AI Assistant</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1">
        {currentView === 'onboarding' ? (
          <OnboardingInterface />
        ) : (
          <ChatInterface />
        )}
      </div>

      {/* Footer Info */}
      <div className="bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 py-2">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-4">
              <span>
                {currentView === 'onboarding' ? '🎯 Onboarding Flow' : '🤖 AI-Powered Chat'}
              </span>
              <span>•</span>
              <span>
                {currentView === 'onboarding' ? 'API: localhost:8001' : 'API: localhost:8000'}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span>Connected</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
