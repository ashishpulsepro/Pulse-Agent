import React, { useEffect, useRef, useState } from 'react'
import { Alert, Button, Card, CardBody, Form, Input, InputGroup, InputGroupText, Spinner, Row, Col } from 'reactstrap'
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

  useEffect(() => { setSessionId(generateSessionId()) }, [])
  useEffect(() => { scrollToBottom(messagesEndRef) }, [messages])
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
      <div className="d-flex justify-content-between align-items-center mb-3 p-3 bg-white shadow-sm rounded-3 chat-header">
        <div className="d-flex align-items-center gap-3" role="button" onClick={clearChat} title="Go to start" style={{cursor:'pointer'}}>
          <div className="rounded-3 d-flex align-items-center justify-content-center" style={{width:40,height:40,background:'#111'}}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          </div>
          <div>
            <div className="fw-semibold">PulsePro AI</div>
            <div className="text-muted small">User Assistant</div>
          </div>
        </div>
        <Button color="link" className="text-muted" onClick={clearChat} title="Clear chat">Clear</Button>
      </div>
    <div className="px-3 pt-3 onboarding-container">
      <div className="centered-content">
      {/* Header */}

      {/* Empty state hero and feature cards */}
      {messages.length === 0 && (
        <div className="empty-layout">
          <div className="text-center mb-4">
            <div className="mx-auto mb-3 rounded-4 d-flex align-items-center justify-content-center" style={{width:64,height:64,background:'#111'}}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            </div>
            <h2 className="fw-bold mb-2">Become Friction-less with <span>⚡ PulsePro</span></h2>
            <p className="text-muted">Make your onboarding process seamless, simply by chatting with Pulse Agent</p>
          </div>

          {/* Wide input */}
          <Card className="border-0 shadow-sm">
            <CardBody className="p-2 p-md-3">
              <Form onSubmit={handleSubmit}>
                <InputGroup>
                  <InputGroupText className="bg-transparent border-0">
                    <div style={{color:'#777'}}>●</div>
                  </InputGroupText>
                  <Input
                    type="textarea"
                    innerRef={textareaRef}
                    value={input}
                    onChange={(e)=>setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask PulsePro to help you with anything..."
                    disabled={loading}
                    rows="1"
                    style={{ minHeight: '30px', maxHeight: '30px', resize: 'none' }}
                    className="border-0"
                  />
                  <Button type="submit" disabled={!input.trim() || loading}
                    style={{
                      background: '#111',
                      border: 'none',
                      padding: '12px 16px',
                      borderRadius: '12px',
                      marginLeft: '8px'
                    }}
                  >
                    {loading ? <Spinner size="sm" /> : (
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>
                    )}
                  </Button>
                </InputGroup>
              </Form>
            </CardBody>
          </Card>

          {/* Feature Cards at bottom */}
          <Row className="g-3 cards-bottom">
            <Col md="4">
              <Card className="h-100 feature-card-light" onClick={()=>setInput('Hey Pulse help me create a new site.')}>
                <CardBody className="text-center">
                  <div className="mx-auto mb-2 rounded-3 d-inline-flex align-items-center justify-content-center" style={{width:48,height:48,background:'#f0f0f0'}}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#111" strokeWidth="2"><path d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"/></svg>
                  </div>
                  <div className="fw-semibold">Create Site</div>
                  <div className="text-muted small">Create a new site</div>
                </CardBody>
              </Card>
            </Col>
            <Col md="4">
              <Card className="h-100 feature-card-light" onClick={()=>setInput('Hey Pulse, add a new user account ')}>
                <CardBody className="text-center">
                  <div className="mx-auto mb-2 rounded-3 d-inline-flex align-items-center justify-content-center" style={{width:48,height:48,background:'#f0f0f0'}}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#111" strokeWidth="2"><path d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"/></svg>
                  </div>
                  <div className="fw-semibold">Create User</div>
                  <div className="text-muted small">Add new user account</div>
                </CardBody>
              </Card>
            </Col>
            <Col md="4">
              <Card className="h-100 feature-card-light" onClick={()=>setInput('Hey Pulse, prepare a checklist ')}>
                <CardBody className="text-center">
                  <div className="mx-auto mb-2 rounded-3 d-inline-flex align-items-center justify-content-center" style={{width:48,height:48,background:'#f0f0f0'}}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#111" strokeWidth="2"><path d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"/></svg>
                  </div>
                  <div className="fw-semibold">Create Template</div>
                  <div className="text-muted small">Build your Checklist</div>
                </CardBody>
              </Card>
            </Col>
          </Row>
        </div>
      )}

      {/* Messages */}
  <div className="chat-scroll-area">
    <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`mb-3 ${m.role==='user' ? 'text-end' : 'text-start'}`}>
            <div className={`d-inline-block p-3 rounded-4 ${m.role==='user' ? '' : 'bg-white border'}`} style={m.role==='user' ? {background:'#111', color:'#fff'} : {}}>
              {m.role === 'assistant' ? (
                <div className="markdown-content">
                  {renderMarkdown(m.content)}
                </div>
              ) : (
                <div style={{whiteSpace:'pre-wrap'}}>{m.content}</div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-start mb-3">
            <div className="d-inline-flex align-items-center gap-2 p-3 rounded-4 bg-white border">
              <span className="spinner-grow spinner-grow-sm text-secondary" />
              <span className="text-muted small">AI is thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
  </div>
  </div>

      {/* Bottom input when there are messages */}
      {messages.length > 0 && (
        <Card className="border-0 shadow-sm bottom-input">
          <CardBody className="p-2 p-md-3">
            <Form onSubmit={handleSubmit}>
              <InputGroup>
                <Input
                  type="textarea"
                  innerRef={textareaRef}
                  value={input}
                  onChange={(e)=>setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message..."
                  disabled={loading}
                  rows="1"
                  style={{ minHeight: '70px', maxHeight: '140px', resize: 'none' }}
                  className="border-0"
                />
                <Button type="submit" disabled={!input.trim() || loading}
                  style={{
                    background: '#111',
                    border: 'none',
                    padding: '10px 14px',
                    borderRadius: '12px',
                    marginLeft: '8px'
                  }}
                >
                  {loading ? <Spinner size="sm" /> : (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>
                  )}
                </Button>
              </InputGroup>
              <div className="text-muted small mt-2 text-center">Press Enter to send, Shift + Enter for new line</div>
            </Form>
          </CardBody>
        </Card>
      )}

      {error && <Alert color="danger" className="mt-2">{error}</Alert>}
      </div>
    </div>
    </>
  )
}
