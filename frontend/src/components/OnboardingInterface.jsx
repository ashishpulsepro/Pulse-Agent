import React, { useState, useEffect, useRef } from 'react';
import onboardingApiService from '../services/onboardingApi';
import { generateSessionId, scrollToBottom, autoResizeTextarea } from '../utils/chatUtils';

const OnboardingInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [currentStep, setCurrentStep] = useState('welcome');
  const [awaitingInput, setAwaitingInput] = useState(null);
  const [userFormData, setUserFormData] = useState({});
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

  // Start onboarding when session ID is ready
  useEffect(() => {
    if (sessionId) {
      startOnboarding();
    }
  }, [sessionId]);

  const startOnboarding = async () => {
    try {
      setIsLoading(true);
      
      const response = await onboardingApiService.startOnboarding(null, sessionId);
      
      if (response.success) {
        // Format the welcome message properly
        const welcomeMessage = `🎉 Welcome to PulsePro! Let's set you up with these 3 easy steps:

**Step 1:** Create Sites 🏢
**Step 2:** Add Users 👥  
**Step 3:** Create Templates 📋

Good news! We've already created a demo site for you to get started.

Let's begin with Step 1. Would you like to create a new site?`;

        addMessage(welcomeMessage, 'ai');
        setCurrentStep('ask_create_site');
        setAwaitingInput('yes_no_create_site');
      }
    } catch (err) {
      addMessage(`Failed to start onboarding: ${err.message}`, 'ai', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const addMessage = (text, sender, status = null) => {
    setMessages(prev => [...prev, {
      id: Date.now(),
      text,
      sender,
      status,
      timestamp: new Date()
    }]);
  };

  const interpretUserIntent = (message, expectedType) => {
    const msg = message.toLowerCase().trim();
    
    if (expectedType === 'yes_no_create_site') {
      // Check for positive responses
      if (msg.includes('yes') || msg.includes('create') || msg.includes('site') || 
          msg.includes('sure') || msg.includes('ok') || msg.includes('yeah') ||
          msg === 'y' || msg.includes('let\'s go') || msg.includes('proceed')) {
        return 'yes';
      }
      // Check for negative responses  
      if (msg.includes('no') || msg.includes('skip') || msg.includes('next') ||
          msg.includes('move') || msg.includes('forward') || msg === 'n' ||
          msg.includes('continue') || msg.includes('users')) {
        return 'no';
      }
    }
    
    if (expectedType === 'yes_no_another_site') {
      if (msg.includes('yes') || msg.includes('another') || msg.includes('more') ||
          msg.includes('create') || msg === 'y' || msg.includes('sure')) {
        return 'yes';
      }
      if (msg.includes('no') || msg.includes('next') || msg.includes('users') ||
          msg.includes('move') || msg.includes('forward') || msg === 'n' ||
          msg.includes('continue') || msg.includes('step 2')) {
        return 'no';
      }
    }

    if (expectedType === 'yes_no_add_user') {
      if (msg.includes('yes') || msg.includes('add') || msg.includes('user') ||
          msg.includes('sure') || msg === 'y' || msg.includes('create')) {
        return 'yes';
      }
      if (msg.includes('no') || msg.includes('skip') || msg.includes('next') ||
          msg.includes('templates') || msg === 'n' || msg.includes('forward') ||
          msg.includes('step 3')) {
        return 'no';
      }
    }

    if (expectedType === 'yes_no_another_user') {
      if (msg.includes('yes') || msg.includes('another') || msg.includes('more') ||
          msg.includes('add') || msg === 'y' || msg.includes('user')) {
        return 'yes';
      }
      if (msg.includes('no') || msg.includes('next') || msg.includes('templates') ||
          msg.includes('move') || msg.includes('forward') || msg === 'n' ||
          msg.includes('step 3')) {
        return 'no';
      }
    }

    return message; // Return original if no intent detected
  };

  const handleUserMessage = async (message) => {
    if (!message.trim() || isLoading) return;

    const userMessage = message.trim();
    setInputMessage('');
    setIsLoading(true);

    // Add user message to chat
    addMessage(userMessage, 'user');

    try {
      if (awaitingInput) {
        const intent = interpretUserIntent(userMessage, awaitingInput);
        
        if (awaitingInput === 'yes_no_create_site') {
          if (intent === 'yes') {
            addMessage("Great! What would you like to name your new site?", 'ai');
            setAwaitingInput('site_name');
          } else if (intent === 'no') {
            addMessage("No problem! Let's move to Step 2 - Adding Users. Would you like to add a user to your team?", 'ai');
            setCurrentStep('ask_add_user');
            setAwaitingInput('yes_no_add_user');
          } else {
            addMessage("I didn't quite understand. Please answer with 'yes' if you want to create a site, or 'no' to move to adding users.", 'ai');
          }
        }
        
        else if (awaitingInput === 'site_name') {
          try {
            const response = await onboardingApiService.createSite(userMessage, sessionId);
            if (response.success) {
              addMessage(`✅ Excellent! Site '${userMessage}' has been created successfully! Would you like to create another site?`, 'ai');
              setAwaitingInput('yes_no_another_site');
            } else {
              addMessage(`Failed to create site: ${response.message}`, 'ai', 'error');
            }
          } catch (err) {
            addMessage(`Failed to create site: ${err.message}`, 'ai', 'error');
          }
        }
        
        else if (awaitingInput === 'yes_no_another_site') {
          if (intent === 'yes') {
            addMessage("Perfect! What would you like to name this next site?", 'ai');
            setAwaitingInput('site_name');
          } else if (intent === 'no') {
            addMessage("Great! Now let's move to Step 2 - Adding Users. Would you like to add a user to your team?", 'ai');
            setCurrentStep('ask_add_user');
            setAwaitingInput('yes_no_add_user');
          } else {
            addMessage("Please answer with 'yes' to create another site, or 'no' to move to adding users.", 'ai');
          }
        }
        
        else if (awaitingInput === 'yes_no_add_user') {
          if (intent === 'yes') {
            addMessage("Excellent! Let's add a new user. What's their first name?", 'ai');
            setAwaitingInput('user_first_name');
            setUserFormData({});
          } else if (intent === 'no') {
            addMessage("That's fine! Let's move to Step 3 - Templates. Templates allow you to create reusable checklists and workflows. This feature will be available soon. Your onboarding is now complete! 🎉", 'ai');
            setCurrentStep('completed');
            setAwaitingInput(null);
          } else {
            addMessage("Please answer with 'yes' to add a user, or 'no' to skip to templates.", 'ai');
          }
        }
        
        else if (awaitingInput === 'user_first_name') {
          setUserFormData(prev => ({ ...prev, first_name: userMessage }));
          addMessage("Got it! What's their last name?", 'ai');
          setAwaitingInput('user_last_name');
        }
        
        else if (awaitingInput === 'user_last_name') {
          setUserFormData(prev => ({ ...prev, last_name: userMessage }));
          addMessage("Perfect! What's their email address?", 'ai');
          setAwaitingInput('user_email');
        }
        
        else if (awaitingInput === 'user_email') {
          setUserFormData(prev => ({ ...prev, email: userMessage }));
          addMessage("Great! What permission level should they have? (Admin, Manager, or User)", 'ai');
          setAwaitingInput('user_permission');
        }
        
        else if (awaitingInput === 'user_permission') {
          try {
            const response = await onboardingApiService.createUser(
              userFormData.first_name,
              userFormData.last_name,
              userFormData.email,
              userMessage,
              sessionId
            );
            if (response.success) {
              addMessage(`✅ Perfect! User '${userFormData.first_name} ${userFormData.last_name}' has been added successfully! Would you like to add another user?`, 'ai');
              setAwaitingInput('yes_no_another_user');
              setUserFormData({});
            } else {
              addMessage(`Failed to create user: ${response.message}`, 'ai', 'error');
            }
          } catch (err) {
            addMessage(`Failed to create user: ${err.message}`, 'ai', 'error');
          }
        }
        
        else if (awaitingInput === 'yes_no_another_user') {
          if (intent === 'yes') {
            addMessage("Great! Let's add another user. What's their first name?", 'ai');
            setAwaitingInput('user_first_name');
            setUserFormData({});
          } else if (intent === 'no') {
            addMessage("Excellent! Now let's move to Step 3 - Templates. Templates allow you to create reusable checklists and workflows. This feature will be available soon. Your onboarding is now complete! 🎉", 'ai');
            setCurrentStep('completed');
            setAwaitingInput(null);
          } else {
            addMessage("Please answer with 'yes' to add another user, or 'no' to finish onboarding.", 'ai');
          }
        }
      }
    } catch (error) {
      console.error('Error handling user message:', error);
      addMessage('Sorry, I encountered an error. Please try again.', 'ai', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleUserMessage(inputMessage);
    }
  };

  const clearOnboarding = () => {
    setMessages([]);
    setSessionId(generateSessionId());
    setCurrentStep('welcome');
    setAwaitingInput(null);
    setUserFormData({});
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getCurrentStepDisplay = () => {
    if (currentStep === 'welcome' || currentStep.includes('site')) return 'Step 1: Create Sites';
    if (currentStep.includes('user')) return 'Step 2: Add Users';
    if (currentStep.includes('template') || currentStep === 'completed') return 'Step 3: Templates';
    return 'Onboarding';
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex justify-between items-center shadow-sm">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">PulsePro Onboarding</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">{getCurrentStepDisplay()}</p>
          </div>
        </div>
        <button
          onClick={clearOnboarding}
          className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          title="Restart onboarding"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6">
          {messages.length === 0 && !isLoading && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">Welcome to PulsePro Onboarding</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-md mx-auto">
                Let's get you set up in 3 easy steps. I'll guide you through creating sites, adding users, and setting up templates.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`mb-6 ${message.sender === 'user' ? 'message-user' : 'message-ai'}`}>
              {message.sender === 'user' ? (
                // User message - right aligned
                <div className="flex items-start justify-end space-x-4">
                  <div className="flex-1 max-w-2xl text-right">
                    <div className="inline-block p-4 rounded-2xl rounded-br-md bg-blue-500 text-white">
                      <p className="text-sm leading-relaxed whitespace-pre-wrap text-left">{message.text}</p>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {formatTimestamp(message.timestamp)}
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                </div>
              ) : (
                // AI message - left aligned  
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                  </div>
                  <div className="flex-1 max-w-2xl">
                    <div className="inline-block p-4 rounded-2xl rounded-bl-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700">
                      <div className="text-sm leading-relaxed whitespace-pre-wrap" dangerouslySetInnerHTML={{
                        __html: message.text
                          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          .replace(/\n/g, '<br/>')
                      }} />
                      
                      {/* Status Badge for AI messages */}
                      {message.status && (
                        <div className="mt-2 flex items-center">
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            message.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                            message.status === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                            'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                          }`}>
                            {message.status === 'completed' && (
                              <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                            )}
                            {message.status === 'error' && (
                              <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            )}
                            {message.status.replace('_', ' ')}
                          </span>
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
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                </div>
                <div className="flex-1">
                  <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl rounded-bl-md p-4 max-w-xs">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                      <span className="text-sm text-gray-500 dark:text-gray-400">Setting up...</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-4">
        <div className="max-w-3xl mx-auto">
          <div className="relative flex items-end space-x-3">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  awaitingInput === 'yes_no_create_site' ? "Type 'yes' to create a site or 'no' to move to users..." :
                  awaitingInput === 'site_name' ? "Enter the site name..." :
                  awaitingInput === 'yes_no_another_site' ? "Type 'yes' for another site or 'no' to continue..." :
                  awaitingInput === 'yes_no_add_user' ? "Type 'yes' to add a user or 'no' to skip..." :
                  awaitingInput === 'user_first_name' ? "Enter the user's first name..." :
                  awaitingInput === 'user_last_name' ? "Enter the user's last name..." :
                  awaitingInput === 'user_email' ? "Enter the user's email..." :
                  awaitingInput === 'user_permission' ? "Enter permission level (Admin, Manager, User)..." :
                  awaitingInput === 'yes_no_another_user' ? "Type 'yes' for another user or 'no' to finish..." :
                  "Type your response..."
                }
                className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                rows="1"
                disabled={isLoading || currentStep === 'completed'}
                style={{ minHeight: '44px', maxHeight: '120px' }}
              />
              <button
                onClick={() => handleUserMessage(inputMessage)}
                disabled={!inputMessage.trim() || isLoading || currentStep === 'completed'}
                className="absolute right-2 bottom-2 p-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
            {currentStep === 'completed' ? 'Onboarding completed! Click restart to begin again.' : 'Press Enter to send, Shift + Enter for new line'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OnboardingInterface;
