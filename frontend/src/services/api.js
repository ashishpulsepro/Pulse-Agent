// API configuration
const API_BASE_URL = "http://localhost:8000";

// API service class
class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Generic request method
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error for ${endpoint}:`, error);
      throw error;
    }
  }

  // Chat methods
  async sendMessage(message, sessionId = null) {
    // const access_token=localStorage.getItem("user").get("access_token")
    const access_token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzU3MTg2MTg3LCJqdGkiOiI2NDhhNGNiYTY0ODM0ZDE0YWNhOWIyMDY3NTE0OGFkZiIsInVzZXJfaWQiOjc1Nn0.Q3VvGcVrxOZ83UVQkXafGuRvV3jyNSO9Bdgruywb4z4'
    // const email_id=await this.getUserEmail(access_token)
    // console.log("email: "+ email_id)
    const email_id="ashish@pulsepro.ai"
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

  async getUserProfile(accessToken) {
  try {
    console.log("inside");
    const response = await fetch("/api/common/get_profile/", {
      method: "GET",
      headers: {
        "Accept": "application/json, text/plain, */*",
        "Authorization": `Bearer ${accessToken}`
      }
    });

    console.log("status:", response.status);

    // Read once
    const text = await response.text();
    console.log("raw response:", text);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Parse JSON from text
    const profile = JSON.parse(text);

    // return only selected fields
    return {
      email: profile.email,
      firstName: profile.first_name,
      lastName: profile.last_name,
      role: profile.role
        };
  } catch (error) {
    console.error("Failed to fetch profile:", error);
    return null;
  }
}

async getUserEmail(accessToken) {
  try {
    console.log("inside");
    const response = await fetch("/api/common/get_profile/", {
      method: "GET",
      headers: {
        "Accept": "application/json, text/plain, */*",
        "Authorization": `Bearer ${accessToken}`
      }
    });

    console.log("status:", response.status);

    // Read once
    const text = await response.text();
    console.log("raw response:", text);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Parse JSON from text
    const profile = JSON.parse(text);

    console.log("email:", profile.email);
    return profile.email;
  } catch (error) {
    console.error("Failed to fetch profile:", error);
    return null;
  }
}




  async getChatHistory(sessionId) {
    return this.request(`/chat/history/${sessionId}`);
  }

  async clearChatSession(sessionId) {
    return this.request(`/chat/sessions/${sessionId}`, {
      method: "DELETE",
    });
  }

  

  // Health check methods
  async checkHealth() {
    return this.request("/health");
  }

  async checkChatHealth() {
    return this.request("/chat/health");
  }

  async testOllamaConnection() {
    return this.request("/ollama/test");
  }

  // System status
  async getSystemStatus() {
    return this.request("/status");
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;
