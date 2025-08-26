import React, { useState, useEffect, useRef } from 'react';
import { generateSessionId, scrollToBottom, autoResizeTextarea } from '../utils/chatUtils';

const API_BASE_URL = "http://localhost:8000";

const OnboardingInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [currentStep, setCurrentStep] = useState('greeting');
  const [progress, setProgress] = useState(0);
  const [onboardingData, setOnboardingData] = useState({
    sites: [],
    users: [],
    completed: false
  });
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Generate session ID on component mount
  useEffect(() => {
    setSessionId(generateSessionId());
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom(messagesEndRef);
  }, [messages]);

  // Auto-resize textarea when content changes
  useEffect(() => {
    if (textareaRef.current) {
      autoResizeTextarea(textareaRef.current);
    }
  }, [inputMessage]);

  const startOnboarding = async () => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/onboarding/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId })
      });
      
      const data = await response.json();
      
      // Add welcome message to chat
      setMessages([{
        id: Date.now(),
        text: data.message,
        sender: 'ai',
        status: data.status,
        timestamp: new Date(),
        step: data.current_step,
        progress: data.progress
      }]);
      
      setCurrentStep(data.current_step);
      setProgress(data.progress);
      
    } catch (error) {
      console.error('Error starting onboarding:', error);
      setMessages([{
        id: Date.now(),
        text: 'Sorry, I encountered an error starting the onboarding. Please try again.',
        sender: 'ai',
        status: 'error',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setIsLoading(true);

    // Add user message to chat
    setMessages(prev => [...prev, {
      id: Date.now(),
      text: userMessage,
      sender: 'user',
      timestamp: new Date()
    }]);

    try {
      const response = await fetch(`${API_BASE_URL}/onboarding/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId
        })
      });
      
      const data = await response.json();
      
      // Add AI response to chat
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: data.message,
        sender: 'ai',
        status: data.status,
        timestamp: new Date(),
        step: data.current_step,
        progress: data.progress,
        data: data.data
      }]);

      // Update onboarding state
      setCurrentStep(data.current_step);
      setProgress(data.progress);
      
      // Update onboarding data if available
      if (data.data) {
        if (data.data.sites_created) {
          setOnboardingData(prev => ({
            ...prev,
            sites: data.data.sites_created
          }));
        }
        if (data.data.users_created) {
          setOnboardingData(prev => ({
            ...prev,
            users: data.data.users_created
          }));
        }
        if (data.data.summary) {
          setOnboardingData(prev => ({
            ...prev,
            sites: data.data.summary.sites_created || [],
            users: data.data.summary.users_created || [],
            completed: data.status === 'onboarding_completed'
          }));
        }
      }

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'ai',
        status: 'error',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const resetOnboarding = () => {
    setMessages([]);
    setSessionId(generateSessionId());
    setCurrentStep('greeting');
    setProgress(0);
    setOnboardingData({ sites: [], users: [], completed: false });
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getStepIcon = (step) => {
    switch (step) {
      case 'site_creation':
      case 'site_creation_loop':
        return '🏢';
      case 'user_creation':
      case 'user_creation_loop':
        return '👥';
      case 'template_creation':
        return '📋';
      case 'completed':
        return '🎉';
      default:
        return '✨';
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header with Progress */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 shadow-sm">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
              <span className="text-xl">{getStepIcon(currentStep)}</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">PulsePro Onboarding</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">Let's get you set up in 3 easy steps</p>
            </div>
          </div>
          <button
            onClick={resetOnboarding}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            title="Restart onboarding"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
        
        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-2">
          <span>Step 1: Sites 🏢</span>
          <span>Step 2: Users 👥</span>
          <span>Step 3: Templates 📋</span>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          {messages.length === 0 && (
            <div className="text-center py-16">
              <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl flex items-center justify-center mx-auto mb-8">
                <span className="text-3xl">🎉</span>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Welcome to PulsePro!</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-lg mx-auto">
                I'll guide you through setting up your workspace in just 3 simple steps. Ready to get started?
              </p>
              <button
                onClick={startOnboarding}
                className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 px-8 rounded-xl transition-all duration-200 transform hover:scale-105 shadow-lg"
              >
                Start Onboarding 🚀
              </button>
              
              {/* Onboarding Steps Preview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto mt-12">
                <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-sm">
                  <div className="text-3xl mb-4">🏢</div>
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Create Sites</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Set up your business locations and workspaces</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-sm">
                  <div className="text-3xl mb-4">👥</div>
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Add Team Members</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Invite your team and set their permissions</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-sm">
                  <div className="text-3xl mb-4">📋</div>
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Setup Templates</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Create checklists and workflows</p>
                </div>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`mb-6 ${message.sender === 'user' ? 'message-user' : 'message-ai'}`}>
              {message.sender === 'user' ? (
                // User message - right aligned
                <div className="flex items-start justify-end space-x-4">
                  <div className="flex-1 max-w-2xl text-right">
                    <div className="inline-block p-4 rounded-2xl rounded-br-md bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg">
                      <p className="text-sm leading-relaxed whitespace-pre-wrap text-left">{message.text}</p>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {formatTimestamp(message.timestamp)}
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                </div>
              ) : (
                // AI message - left aligned  
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-green-500 to-teal-500 flex items-center justify-center shadow-lg">
                    <span className="text-sm">{getStepIcon(message.step || currentStep)}</span>
                  </div>
                  <div className="flex-1 max-w-3xl">
                    <div className="inline-block p-4 rounded-2xl rounded-bl-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700 shadow-lg">
                      <div className="prose prose-sm max-w-none">
                        {message.text.split('\n').map((line, index) => {
                          if (line.trim() === '') return <br key={index} />;
                          
                          // Handle markdown-style formatting
                          if (line.startsWith('**') && line.endsWith('**')) {
                            return <h3 key={index} className="font-bold text-lg mb-2 text-gray-900 dark:text-white">{line.slice(2, -2)}</h3>;
                          }
                          if (line.startsWith('• ')) {
                            return <li key={index} className="ml-4 text-gray-700 dark:text-gray-300">{line.slice(2)}</li>;
                          }
                          return <p key={index} className="mb-2 text-gray-700 dark:text-gray-300">{line}</p>;
                        })}
                      </div>
                      
                      {/* Status Badge for AI messages */}
                      {message.status && (
                        <div className="mt-3 flex items-center justify-between">
                          <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                            message.status === 'onboarding_completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                            message.status === 'site_created' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                            message.status === 'user_created' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200' :
                            message.status === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                            'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                          }`}>
                            {message.status === 'onboarding_completed' && '🎉 '}
                            {message.status === 'site_created' && '🏢 '}
                            {message.status === 'user_created' && '👥 '}
                            {message.status === 'error' && '❌ '}
                            {message.status.replace('_', ' ')}
                          </span>
                          
                          {message.progress !== undefined && (
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {message.progress}% complete
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {formatTimestamp(message.timestamp)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="mb-6">
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-green-500 to-teal-500 flex items-center justify-center shadow-lg">
                  <span className="text-sm animate-pulse">{getStepIcon(currentStep)}</span>
                </div>
                <div className="flex-1">
                  <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl rounded-bl-md p-4 max-w-xs shadow-lg">
                    <div className="flex items-center space-x-3">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                      <span className="text-sm text-gray-500 dark:text-gray-400">Setting things up...</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Summary Panel (if onboarding completed) */}
      {onboardingData.completed && (
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900 dark:to-emerald-900 border-t border-green-200 dark:border-green-700 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">🎉</span>
                <div>
                  <h3 className="font-semibold text-green-900 dark:text-green-100">Onboarding Complete!</h3>
                  <p className="text-sm text-green-700 dark:text-green-300">
                    {onboardingData.sites.length} sites and {onboardingData.users.length} users created
                  </p>
                </div>
              </div>
              <button
                onClick={() => window.location.href = '/dashboard'}
                className="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
              >
                Go to Dashboard
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="relative flex items-end space-x-3">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={messages.length === 0 ? "Click 'Start Onboarding' above to begin..." : "Type your response..."}
                className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                rows="1"
                disabled={isLoading || messages.length === 0}
                style={{ minHeight: '44px', maxHeight: '120px' }}
              />
              <button
                onClick={sendMessage}
                disabled={!inputMessage.trim() || isLoading || messages.length === 0}
                className="absolute right-2 bottom-2 p-2 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
            Press Enter to send, Shift + Enter for new line
          </div>
        </div>
      </div>
    </div>
  );
};

export default OnboardingInterface;
