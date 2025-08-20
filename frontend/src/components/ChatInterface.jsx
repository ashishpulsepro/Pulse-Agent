import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, MoreVertical, Phone, Video, Search } from 'lucide-react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! How can I help you today?",
      sender: 'ai',
      timestamp: new Date(Date.now() - 300000)
    },
    {
      id: 2,
      text: "I need help creating a React application with a chat interface",
      sender: 'user',
      timestamp: new Date(Date.now() - 240000)
    },
    {
      id: 3,
      text: "I'd be happy to help you create a React chat interface! What specific features are you looking for?",
      sender: 'ai',
      timestamp: new Date(Date.now() - 180000)
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (inputText.trim()) {
      const newMessage = {
        id: messages.length + 1,
        text: inputText,
        sender: 'user',
        timestamp: new Date()
      };
      
      setMessages([...messages, newMessage]);
      setInputText('');
      setIsTyping(true);
      
      // Simulate AI response
      setTimeout(() => {
        const aiResponse = {
          id: messages.length + 2,
          text: "Thanks for your message! I'm here to help you with your React development needs.",
          sender: 'ai',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, aiResponse]);
        setIsTyping(false);
      }, 1500);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="vh-100 d-flex flex-column" style={{ backgroundColor: '#f8f9fa' }}>
      {/* Header */}
      <div className="bg-white border-bottom px-4 py-3 d-flex align-items-center justify-content-between shadow-sm">
        <div className="d-flex align-items-center">
          <div className="rounded-circle bg-primary d-flex align-items-center justify-content-center me-3" style={{ width: '40px', height: '40px' }}>
            <span className="text-white fw-bold">AI</span>
          </div>
          <div>
            <h6 className="mb-0 fw-semibold">AI Assistant</h6>
            <small className="text-muted">Online</small>
          </div>
        </div>
        <div className="d-flex align-items-center gap-3">
          <button className="btn btn-outline-secondary btn-sm rounded-circle p-2">
            <Phone size={16} />
          </button>
          <button className="btn btn-outline-secondary btn-sm rounded-circle p-2">
            <Video size={16} />
          </button>
          <button className="btn btn-outline-secondary btn-sm rounded-circle p-2">
            <Search size={16} />
          </button>
          <button className="btn btn-outline-secondary btn-sm rounded-circle p-2">
            <MoreVertical size={16} />
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-grow-1 overflow-auto p-4" style={{ backgroundColor: '#f8f9fa' }}>
        <div className="container-fluid">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`d-flex mb-3 ${message.sender === 'user' ? 'justify-content-end' : 'justify-content-start'}`}
            >
              <div
                className={`rounded-3 px-3 py-2 shadow-sm position-relative ${
                  message.sender === 'user'
                    ? 'bg-primary text-white'
                    : 'bg-white text-dark border'
                }`}
                style={{ maxWidth: '70%' }}
              >
                <p className="mb-1">{message.text}</p>
                <small className={`d-block text-end ${message.sender === 'user' ? 'text-white-50' : 'text-muted'}`}>
                  {formatTime(message.timestamp)}
                </small>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="d-flex justify-content-start mb-3">
              <div className="bg-white rounded-3 px-3 py-2 shadow-sm border">
                <div className="d-flex align-items-center">
                  <div className="spinner-grow spinner-grow-sm text-primary me-2" role="status" style={{ width: '0.5rem', height: '0.5rem' }}>
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  <div className="spinner-grow spinner-grow-sm text-primary me-2" role="status" style={{ width: '0.5rem', height: '0.5rem', animationDelay: '0.15s' }}>
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  <div className="spinner-grow spinner-grow-sm text-primary" role="status" style={{ width: '0.5rem', height: '0.5rem', animationDelay: '0.3s' }}>
                    <span className="visually-hidden">Loading...</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-top p-4">
        <div className="container-fluid">
          <div className="input-group">
            <button className="btn btn-outline-secondary" type="button">
              <Paperclip size={18} />
            </button>
            <textarea
              className="form-control border-0 shadow-none"
              placeholder="Type your message..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              rows="1"
              style={{ 
                resize: 'none',
                minHeight: '40px',
                backgroundColor: '#f8f9fa'
              }}
            />
            <button 
              className="btn btn-primary"
              type="button"
              onClick={handleSendMessage}
              disabled={!inputText.trim()}
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Bootstrap CSS */}
      <link 
        href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" 
        rel="stylesheet" 
      />
      
      <style jsx>{`
        .input-group textarea:focus {
          box-shadow: none !important;
          border-color: transparent !important;
        }
        
        .spinner-grow {
          animation-duration: 1s;
        }
        
        .btn:focus {
          box-shadow: none !important;
        }
        
        .container-fluid {
          max-width: 800px;
        }
        
        /* Custom scrollbar */
        .overflow-auto::-webkit-scrollbar {
          width: 6px;
        }
        
        .overflow-auto::-webkit-scrollbar-track {
          background: #f1f1f1;
        }
        
        .overflow-auto::-webkit-scrollbar-thumb {
          background: #c1c1c1;
          border-radius: 3px;
        }
        
        .overflow-auto::-webkit-scrollbar-thumb:hover {
          background: #a8a8a8;
        }
        
        /* Message animations */
        .d-flex.mb-3 {
          animation: fadeInUp 0.3s ease-out;
        }
        
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
};

export default ChatInterface;