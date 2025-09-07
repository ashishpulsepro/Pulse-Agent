# Agent Frontend

React 18 + Vite + Reactstrap UI (black/white theme) for Pulse Agent.

## Dev

1. Install deps
2. Start dev server

```powershell
cd C:\Users\areeb\Pulse-Agent\agent-frontend
npm i
npm run dev
```

Optional env:

- Set API base (defaults to http://localhost:8000):

```powershell
$env:VITE_API_BASE_URL = 'http://localhost:8000'; npm run dev
```

## Auth (temporary)

- Put a refresh token in localStorage under any of these keys:
  - `PULSE_REFRESH_TOKEN` (preferred)
  - `refresh_token`
  - `access_token`
- Put a user email under `PULSE_USER_EMAIL` (optional, falls back to ashish@pulsepro.ai)

The client sends `Authorization: Bearer <token>` on all requests. We’ll replace this with real auth later.

## UI

- Toggle: Onboarding | Agent Chat (agent is placeholder)
- Onboarding chat:
  - 3 feature cards (Create Site, Create User, Create Checklist)
  - Messages and input using Reactstrap

## TODO / Unknowns

- Confirm response schema for onboarding (message vs reply)
- Real auth flow (refresh→access, or cookie). Currently uses localStorage token.
- Session management: sessionId persistence and list/reuse (backend supports history and /sessions/{email})
- Agent chat endpoint path and payload
- CORS: add Vite proxy if needed

```
# Example localStorage setup in DevTools console
localStorage.setItem('PULSE_REFRESH_TOKEN', '<YOUR_REFRESH_TOKEN>')
localStorage.setItem('PULSE_USER_EMAIL', 'you@domain.com')
```
