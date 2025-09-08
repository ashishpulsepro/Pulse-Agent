import React, { useEffect, useRef, useState } from 'react'
import { Spinner } from 'reactstrap'
import { apiClient } from '../services/apiClient'
import { generateSessionId, scrollToBottom, autoResizeTextarea, renderMarkdown } from '../utils/chatUtils'

export default function OnboardingChat(){
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)
  const headerRef = useRef(null)
  const [headerHeight, setHeaderHeight] = useState(0)

  useEffect(() => { setSessionId(generateSessionId()) }, [])
  useEffect(() => { scrollToBottom(messagesEndRef) }, [messages])
  useEffect(() => {
    const measure = () => {
      if(headerRef.current){
        const h = headerRef.current.offsetHeight || 0
        setHeaderHeight(h)
      }
    }
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [])
  useEffect(() => { if(textareaRef.current) autoResizeTextarea(textareaRef.current) }, [input])

  const sendMessage = async () => {
    if(!input.trim() || loading) return

    const userMsg = { role: 'user', content: input.trim(), timestamp: new Date() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setError('')

    try{
      const res = await apiClient.sendOnboardingMessage(userMsg.content, sessionId)
      const reply = res?.message || res?.reply || '...'
      setMessages(prev => [...prev, { role: 'assistant', content: reply, status: res?.status, timestamp: new Date() }])
    }catch(err){
      setError(err?.message || 'Failed to send')
    }finally{
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendMessage()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
    setSessionId(generateSessionId())
  }

  return (
    <>
      {/* Global header */}
  <header ref={headerRef} className="pulse-header w-100 px-4 md:px-6 py-3 flex items-center justify-between bg-white/80 backdrop-blur border-b border-gray-200">
        <div className="flex items-center gap-3 select-none">
          <div className="h-10 w-10 rounded-xl bg-gray-900 flex items-center justify-center text-white shadow">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          </div>
          <div className="leading-tight">
            <div className="font-semibold text-gray-900 text-sm md:text-base">Pulse Assistant</div>
            <div className="text-[11px] md:text-xs text-gray-500 flex items-center gap-1">
              <span className="inline-flex h-2 w-2 rounded-full bg-green-500" /> Online
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="group flex items-center gap-2 text-xs md:text-sm text-gray-500 hover:text-red-600 font-medium px-3 py-2 rounded-lg hover:bg-red-50 transition-colors"
              title="Clear chat"
              type="button"
            >
              <i className="fa-solid fa-trash-can text-[13px] group-hover:scale-110 transition-transform" />
              <span>Clear</span>
            </button>
          )}
        </div>
      </header>
    <div className="px-3 pt-3 onboarding-container">
      <div className="centered-content">
      {/* Header */}

      {/* Empty state hero and feature cards */}
      {messages.length === 0 && (
        <section
          id="ai-assistant-hero"
          className="relative app-section bg-gradient-to-br from-gray-50 via-white to-blue-50 flex flex-col items-center px-6 pt-14 pb-20 w-full rounded-3xl shadow-sm border border-gray-200 overflow-visible"
          style={headerHeight ? {minHeight:`calc(100vh - ${headerHeight}px)`}: {minHeight:'100vh'}}
        >
          <div className="w-full max-w-5xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 bg-blue-100 text-blue-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
              <i className="fa-solid fa-sparkles text-blue-600" />
              AI-Powered Assistant
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl fw-bold text-gray-900 mb-6 leading-tight">
              Meet Your <span className="text-blue-600">AI Assistant</span>
            </h1>
            <p className="text-lg md:text-xl text-gray-600 mb-10 max-w-3xl mx-auto leading-relaxed px-2">
              Tired of products and their complex workflows . Don't worry we've got you covered! Just Try out our new AI Assistant to make your onboarding process a breeze.
            </p>
            {/* Primary prompt input styled with Tailwind but still using same handlers */}
            <div className="max-w-3xl mx-auto mb-12 px-1">
              <div className="relative">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e)=>setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  placeholder="What would you like me to help you with today?"
                  disabled={loading}
                  className="w-full px-6 py-5 text-base md:text-lg border-2 border-gray-200 rounded-2xl shadow-[0_8px_28px_-8px_rgba(0,0,0,0.15)] focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white resize-none leading-relaxed"
                  style={{minHeight:'64px',maxHeight:'140px'}}
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || loading}
                  className="absolute right-3 top-1/2 -translate-y-1/2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white p-3 rounded-xl transition-colors duration-200 shadow"
                >
                  {loading ? <Spinner size="sm" /> : <i className="fa-solid fa-paper-plane" />}
                </button>
              </div>
              <div className="flex items-center justify-center gap-2 mt-4 text-sm text-gray-500">
                <i className="fa-solid fa-lightbulb text-yellow-500" />
                <span>Press Enter to send, Shift + Enter for a new line</span>
              </div>
            </div>

            {/* Suggested prompts */}
            <div className="space-y-4 max-w-5xl mx-auto px-1">
              <p className="text-gray-600 font-medium mb-1 md:mb-2">Let's get you started with the onboarding procedure:</p>
              <div className="flex flex-wrap justify-center gap-3 md:gap-4">
                <button onClick={()=>setInput('Create a site')} className="prompt-pill bg-white border border-gray-200 hover:border-blue-300 hover:bg-blue-50 px-6 py-3 rounded-full text-gray-700 hover:text-blue-700 transition-all duration-200 shadow-sm hover:shadow-md">
                  <i className="fa-solid fa-building mr-2 text-blue-600" />
                  Add a new site called Downtown Store
                </button>
                <button onClick={()=>setInput('Create a User')} className="prompt-pill bg-white border border-gray-200 hover:border-green-300 hover:bg-green-50 px-6 py-3 rounded-full text-gray-700 hover:text-green-700 transition-all duration-200 shadow-sm hover:shadow-md">
                  <i className="fa-solid fa-user-plus mr-2 text-green-600" />
                  Invite John as an inspector
                </button>
                <button onClick={()=>setInput('Create a template')} className="prompt-pill bg-white border border-gray-200 hover:border-purple-300 hover:bg-purple-50 px-6 py-3 rounded-full text-gray-700 hover:text-purple-700 transition-all duration-200 shadow-sm hover:shadow-md">
                  <i className="fa-solid fa-calendar-check mr-2 text-purple-600" />
                  Create an audit schedule for the NewYork Site
                </button>
                <button onClick={()=>setInput('Share the mobile app download link with my team')} className="prompt-pill bg-white border border-gray-200 hover:border-indigo-300 hover:bg-indigo-50 px-6 py-3 rounded-full text-gray-700 hover:text-indigo-700 transition-all duration-200 shadow-sm hover:shadow-md">
                  <i className="fa-solid fa-share mr-2 text-indigo-600" />
                  Share the mobile app download link with my team
                </button>
              </div>
            </div>

            {/* Feature trio */}
            {/* <div className="mt-20 grid md:grid-cols-3 gap-10 max-w-4xl mx-auto">
              <div className="text-center p-6">
                <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <i className="fa-solid fa-wand-magic-sparkles text-2xl text-blue-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Natural Language</h3>
                <p className="text-gray-600 text-sm">Speak naturally - no need to learn complex commands or navigate menus</p>
              </div>
              <div className="text-center p-6">
                <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <i className="fa-solid fa-bolt text-2xl text-green-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Instant Actions</h3>
                <p className="text-gray-600 text-sm">Watch your requests come to life immediately with smart automation</p>
              </div>
              <div className="text-center p-6">
                <div className="w-16 h-16 bg-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <i className="fa-solid fa-brain text-2xl text-purple-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Context Aware</h3>
                <p className="text-gray-600 text-sm">Understands your business needs and suggests relevant next steps</p>
              </div>
            </div> */}
          </div>
        </section>
      )}

      {/* Unified chat section when messages exist */}
      {messages.length > 0 && (
        <section
          id="ai-chat-section"
          className="relative app-section bg-gradient-to-br from-gray-50 via-white to-blue-50 flex flex-col items-center px-4 md:px-6 pt-14 pb-24 w-full rounded-3xl shadow-sm border border-gray-200"
          style={headerHeight ? {minHeight:`calc(100vh - ${headerHeight}px)`}: {minHeight:'100vh'}}
        >
          <div className="w-full max-w-4xl mx-auto flex flex-col flex-1">
            {/* Chat header with status + clear */}
            

            {/* Messages list */}
            <div className="space-y-4 pr-1">
              {messages.map((m,i)=> (
                <div key={i} className={`flex ${m.role==='user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`rounded-2xl px-4 py-3 text-sm md:text-base leading-relaxed shadow-sm max-w-[85%] whitespace-pre-wrap ${m.role==='user' ? 'bg-blue-600 text-white' : 'bg-white/70 backdrop-blur border border-gray-200 text-gray-900'}`}
                  >
                    {m.role==='assistant' ? (
                      <div className="markdown-content text-gray-900">{renderMarkdown(m.content)}</div>
                    ) : m.content}
                  </div>
                </div>
              ))}
              {error && (
                <div className="flex justify-start">
                  <div className="rounded-2xl px-4 py-3 bg-red-50 border border-red-300 text-red-700 text-sm flex items-start gap-2 max-w-[85%]">
                    <i className="fa-solid fa-triangle-exclamation mt-0.5" />
                    <span className="flex-1">{error}</span>
                    <button onClick={()=>setError('')} className="text-xs underline decoration-dotted hover:text-red-800">dismiss</button>
                  </div>
                </div>
              )}
              {loading && (
                <div className="flex justify-start">
                  <div className="rounded-2xl px-4 py-3 bg-white/70 backdrop-blur border border-gray-200 text-gray-500 text-sm flex items-center gap-2">
                    <span className="spinner-grow spinner-grow-sm text-secondary" />
                    AI is thinking...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input area */}
            <div className="mt-6">
              <form onSubmit={handleSubmit} className="relative">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e)=>setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  placeholder="Type your message..."
                  disabled={loading}
                  className="w-full px-5 py-5 text-sm md:text-base border-2 border-gray-200 rounded-2xl shadow-[0_6px_24px_-10px_rgba(0,0,0,0.25)] focus:outline-none focus:border-blue-500 focus:ring-4 focus:ring-blue-100 transition-all duration-300 bg-white resize-none leading-relaxed"
                  style={{minHeight:'64px',maxHeight:'180px'}}
                />
                <button
                  type="submit"
                  disabled={!input.trim() || loading}
                  className="absolute right-3 top-1/2 -translate-y-1/2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white p-3 rounded-xl transition-colors duration-200 shadow"
                >
                  {loading ? <Spinner size="sm" /> : <i className="fa-solid fa-paper-plane" />}
                </button>
              </form>
              <div className="text-xs md:text-sm text-gray-500 mt-3 text-center">Press Enter to send · Shift + Enter for new line</div>
            </div>
          </div>
        </section>
      )}

  {/* Removed external Alert; errors now appear inline inside chat stream */}
      </div>
    </div>
    </>
  )
}
