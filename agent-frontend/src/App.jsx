import React, { useState } from 'react'
import { Container } from 'reactstrap'
import TopBar from './components/TopBar'
import ModeToggle from './components/ModeToggle'
import OnboardingChat from './components/OnboardingChat'

export default function App() {
  const [mode, setMode] = useState('onboarding') // 'onboarding' | 'agent'

  return (
    <div className="app">
      {/* <TopBar /> */}
      <div className="main-content">
        <OnboardingChat />
      </div>
      {/* <Container className="py-3">
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h5 className="m-0 text-muted">Mode</h5>
          <ModeToggle mode={mode} onChange={setMode} />
        </div>

        {mode === 'onboarding' ? (
          <OnboardingChat />
        ) : (
          <div className="placeholder">
            <h4>Agent Chat (coming soon)</h4>
            <p className="text-muted small">
              No API yet. We will wire this once the endpoint is available.
            </p>
          </div>
        )}
      </Container> */}
    </div>
  )
}
