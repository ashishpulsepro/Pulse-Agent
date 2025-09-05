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
    // const email_id=getUserProfile(access_token)
    const email_id="ashish@pulsepro.ai"
    return this.request("/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        session_id: sessionId,
        email:email_id
      }),
    });
  }

  async getUserProfile(accessToken) {
  try {
    const response = await fetch("https://staging-api.pulsepro.ai/common/get_profile/", {
      method: "GET",
      headers: {
        "Accept": "application/json, text/plain, */*",
        "Authorization": `Bearer ${accessToken}`
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const profile = await response.json();

    // return only selected fields
    return {
      email: profile.email,
      firstName: profile.first_name,
      lastName: profile.last_name,
      role: profile.role,
      imageUrl: profile.image_url
    };
  } catch (error) {
    console.error("Failed to fetch profile:", error);
    return null;
  }
}

async getUserEmail(accessToken) {
  try {
    const response = await fetch("https://staging-api.pulsepro.ai/common/get_profile/", {
      method: "GET",
      headers: {
        "Accept": "application/json, text/plain, */*",
        "Authorization": `Bearer ${accessToken}`
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const profile = await response.json();
    return profile.email;  // returns the "email" field
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
