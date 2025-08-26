import React, { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import OnboardingInterface from './components/OnboardingInterface'
import './App.css'

function App() {
  const [currentMode, setCurrentMode] = useState('normal') // 'normal' or 'onboarding'

  const toggleMode = () => {
    setCurrentMode(currentMode === 'normal' ? 'onboarding' : 'normal')
  }

  return (
    <div className="App">
      {/* Mode Toggle Button */}
      <div className="fixed top-4 right-4 z-50">
        <button
          onClick={toggleMode}
          className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-medium py-2 px-4 rounded-lg shadow-lg transition-all duration-200 transform hover:scale-105"
        >
          {currentMode === 'normal' ? '🎉 Start Onboarding' : '💬 Normal Chat'}
        </button>
      </div>

      {/* Render appropriate interface */}
      {currentMode === 'onboarding' ? (
        <OnboardingInterface />
      ) : (
        <ChatInterface />
      )}
    </div>
  )
}

export default App
