// Mirrors the existing frontend/src/services/api.js patterns
// Default to local backend; override with VITE_API_BASE_URL when deploying the FastAPI backend
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// NOTE: For now, we read a refresh token from localStorage.
// Later, replace getAuthToken() with your real auth flow (access token or cookie).
// function getAccessToken() {
//   const raw = localStorage.getItem("user");
//   if (!raw) return null; // nothing stored
//   const user = JSON.parse(raw);
//   return user?.access_token || null;
// }

function getRefreshToken() {
  const raw = localStorage.getItem("user");
  if (!raw) return null;
  const user = JSON.parse(raw);
  return user?.refresh_token || null;
}

function getUserEmail() {
  const raw = localStorage.getItem("user");
  if (!raw) return null;
  const user = JSON.parse(raw);
  return user?.email || null;
}

class ApiClient {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const refreshToken = getRefreshToken() || 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTc1ODE3MTkyOCwianRpIjoiZjczZjhmMmUxNDMwNGRiZDkyMTNiOGEwNjMwOGJiMDciLCJ1c2VyX2lkIjo3NTZ9.UtjCTpxf9O-7RsibYam5-Bg6VL0Unr1mNOhRgiGk8Rk';

    // If caller passed a plain object body, we'll JSON stringify here and append refresh token if not already present.
    let body = options.body;
    if (body && typeof body === "object" && !(body instanceof FormData)) {
      body = { ...body };
      if (refreshToken && body.refresh_token === undefined) {
        body.refresh_token = refreshToken;
      }
      body = JSON.stringify(body);
    }

    const config = {
      method: options.method || "GET",
      body,
      headers: {
        "Content-Type": "application/json",
        ...(refreshToken ? { Authorization: `Bearer ${refreshToken}` } : {}), // send access token as 'Authorization'
        ...(options.headers || {}),
      },
    };

    const res = await fetch(url, config);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
    }
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) return res.json();
    return res.text();
  }

  // Onboarding chat (existing backend)
  async sendOnboardingMessage(message, sessionId = null) {
    const email_id =
      localStorage.getItem("PULSE_USER_EMAIL") ||
      getUserEmail() ||
      "ashish@pulsepro.ai";
    return this.request("/chat/onboarding", {
      method: "POST",
      body: {
        message,
        session_id: sessionId,
        email: email_id,
      },
    });
  }

  // Placeholder for future agent chat
  async sendAgentMessage(message, sessionId = null) {
    return this.request("/chat", {
      method: "POST",
      body: { message, session_id: sessionId },
    });
  }
}

export const apiClient = new ApiClient();
export default apiClient;
