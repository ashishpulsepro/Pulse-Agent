// Mirrors the existing frontend/src/services/api.js patterns
// Default to local backend; override with VITE_API_BASE_URL when deploying the FastAPI backend
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// NOTE: For now, we read a refresh token from localStorage.
// Later, replace getAuthToken() with your real auth flow (access token or cookie).
function getAccessToken() {
  const user = JSON.parse(localStorage.getItem("user"));
  return  user.access_token;
}

function getRefreshToken() {
  const user = JSON.parse(localStorage.getItem("user"));
  return user.refresh_token;
}

function getUserEmail() {
  const user = JSON.parse(localStorage.getItem("user"));
  return  user.email;
}

class ApiClient {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const accessToken = getAccessToken() || 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzU3MzI1NzIzLCJqdGkiOiIzZDIwYTllZTZkYTA0NDI5OTljM2RiNTBlM2U0NmZkYyIsInVzZXJfaWQiOjc1N30.3NV636FHHAuTJ0pk27ID-xNZCN4IbpYaUpfq8q6iBxk';
    const refreshToken = getRefreshToken() || 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTc1ODEwMTk1MCwianRpIjoiMTdlNzNhOGMyYTc2NDI2NGI3NDQzMDZhY2U2YjI2OTkiLCJ1c2VyX2lkIjo3NTd9.bgurTOG08Yz4HPqL77zYG9lBDscRfmkxhjZSDXWjvis';

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
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}), // send access token as 'Authorization'
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
      "areeb@pulsepro.ai";
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
