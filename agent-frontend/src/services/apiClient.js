// Mirrors the existing frontend/src/services/api.js patterns
// Default to local backend; override with VITE_API_BASE_URL when deploying the FastAPI backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// NOTE: For now, we read a refresh token from localStorage.
// Later, replace getAuthToken() with your real auth flow (access token or cookie).
function getAccessToken() {
  const user = JSON.parse(localStorage.getItem('user'))
  return (
    user && user.access_token
  )
}

function getRefreshToken() {
  const user = JSON.parse(localStorage.getItem('user'))
  return (
    user && user.refresh_token
  )
}

function getUserEmail() {
  const user = JSON.parse(localStorage.getItem('user'))
  return (
    user && user.email
  )
}

class ApiClient {
  constructor(){
    this.baseURL = API_BASE_URL
  }

  async request(endpoint, options = {}){
    const url = `${this.baseURL}${endpoint}`
    const token = getAccessToken() || 
    'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzU3MzE3NTIyLCJqdGkiOiI0OWJkNjg0NWZhOWE0YmE0OWIzZGU3OTk3ZTQ3Njc0MSIsInVzZXJfaWQiOjc1N30.J2RuwXxYApy5_1U3N0e1c6QxY1iG_A1tVeNVj2sLpm8'
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

    // const access_token=localStorage.getItem("user").get("access_token")
    const access_token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzU3MzE3NzY1LCJqdGkiOiIwODVjNmJjNDdlOWU0OTYyOTNlMjdlOWUxMjAyNDE3ZiIsInVzZXJfaWQiOjc1Nn0.foiP-eBvJDerIIGEQFvr4j5rxp615F9dfb8dxth8QO8'
    // const email_id=await this.getUserEmail(access_token)
    // console.log("email: "+ email_id)
  const email_id = localStorage.getItem('PULSE_USER_EMAIL') || 'ashish@pulsepro.ai' // TODO: replace with real user profile
    return this.request("/chat/onboarding", {
      method: "POST",
       headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${access_token}`, // ✅ add auth header
    },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        email:email_id
      }),
    });
  }

  // Placeholder for future agent chat
  async sendAgentMessage(message, sessionId = null){
    // TODO: update endpoint once available
    return this.request('/chat', {
      method: 'POST',
      body: JSON.stringify({ message, session_id: sessionId })
    })
  }
}

export const apiClient = new ApiClient()
export default apiClient
