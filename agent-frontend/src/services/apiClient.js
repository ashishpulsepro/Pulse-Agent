// Mirrors the existing frontend/src/services/api.js patterns
// Default to local backend; override with VITE_API_BASE_URL when deploying the FastAPI backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// NOTE: For now, we read a refresh token from localStorage.
// Later, replace getAuthToken() with your real auth flow (access token or cookie).
function getAuthToken() {
  // Try a few common keys; primary is PULSE_REFRESH_TOKEN for clarity
  return (
    localStorage.getItem('PULSE_REFRESH_TOKEN') ||
    localStorage.getItem('refresh_token') ||
    localStorage.getItem('access_token') ||
    null
  )
}

class ApiClient {
  constructor(){
    this.baseURL = API_BASE_URL
  }

  async request(endpoint, options = {}){
    const url = `${this.baseURL}${endpoint}`
    const token = getAuthToken()
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(options.headers||{})
      },
      ...options,
    }

    const res = await fetch(url, config)
    if(!res.ok){
      const text = await res.text().catch(()=> '')
      throw new Error(`HTTP ${res.status}: ${text || res.statusText}`)
    }
    const contentType = res.headers.get('content-type') || ''
    if(contentType.includes('application/json')) return res.json()
    return res.text()
  }

  // Onboarding chat (existing backend)
  async sendOnboardingMessage(message, sessionId = null){
  const email = localStorage.getItem('PULSE_USER_EMAIL') || 'ashish@pulsepro.ai' // TODO: replace with real user profile
    return this.request('/chat/onboarding', {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId, email })
    })
  }

  // Placeholder for future agent chat
  async sendAgentMessage(message, sessionId = null){
    // TODO: update endpoint once available
    return this.request('/chat/agent', {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId })
    })
  }
}

export const apiClient = new ApiClient()
export default apiClient
