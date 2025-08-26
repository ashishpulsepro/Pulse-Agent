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
    return this.request("/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        session_id: sessionId,
      }),
    });
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

  // Onboarding methods
  async startOnboarding(sessionId = null) {
    return this.request("/onboarding/start", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
      }),
    });
  }

  async sendOnboardingMessage(message, sessionId) {
    return this.request("/onboarding/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        session_id: sessionId,
      }),
    });
  }

  async getOnboardingStatus(sessionId) {
    return this.request(`/onboarding/status/${sessionId}`);
  }

  async clearOnboardingSession(sessionId) {
    return this.request(`/onboarding/session/${sessionId}`, {
      method: "DELETE",
    });
  }

  async clearAllOnboardingSessions() {
    return this.request("/onboarding/sessions", {
      method: "DELETE",
    });
  }

  // Unified chat method (supports both normal and onboarding modes)
  async sendUnifiedMessage(message, sessionId = null, mode = "normal") {
    return this.request("/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        session_id: sessionId,
        mode: mode,
      }),
    });
  }

  // Site management methods
  async getAllSites() {
    return this.request("/sites");
  }

  async createSite(siteData) {
    return this.request("/sites", {
      method: "POST",
      body: JSON.stringify(siteData),
    });
  }

  async createSimpleSite(locationName) {
    return this.request("/sites/simple", {
      method: "POST",
      body: JSON.stringify({ location_name: locationName }),
    });
  }

  async deleteSite(siteId) {
    return this.request(`/sites/${siteId}`, {
      method: "DELETE",
    });
  }

  // User management methods
  async getPermissionBundles() {
    return this.request("/users/permissions");
  }

  async createUser(userData) {
    return this.request("/users", {
      method: "POST",
      body: JSON.stringify(userData),
    });
  }

  // Session management
  async getAllSessions() {
    return this.request("/sessions");
  }

  async clearAllSessions() {
    return this.request("/sessions", {
      method: "DELETE",
    });
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;
