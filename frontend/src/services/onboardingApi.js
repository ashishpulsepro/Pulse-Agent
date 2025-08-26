// Onboarding API service
const ONBOARDING_API_BASE_URL = "http://localhost:8000";

class OnboardingApiService {
  constructor() {
    this.baseURL = ONBOARDING_API_BASE_URL;
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
      console.error(`Onboarding API Error for ${endpoint}:`, error);
      throw error;
    }
  }

  // Onboarding flow methods
  async startOnboarding(userId = null, sessionId = null) {
    return this.request("/onboarding/start", {
      method: "POST",
      body: JSON.stringify({
        user_id: userId,
        session_id: sessionId,
      }),
    });
  }

  async performAction(action, sessionId, data = null) {
    return this.request("/onboarding/action", {
      method: "POST",
      body: JSON.stringify({
        action,
        session_id: sessionId,
        data,
      }),
    });
  }

  async createSite(siteName, sessionId) {
    return this.request("/onboarding/create-site", {
      method: "POST",
      body: JSON.stringify({
        site_name: siteName,
        session_id: sessionId,
      }),
    });
  }

  async createUser(firstName, lastName, email, permissionSet, sessionId) {
    return this.request("/onboarding/create-user", {
      method: "POST",
      body: JSON.stringify({
        first_name: firstName,
        last_name: lastName,
        email: email,
        permission_set_name: permissionSet,
        session_id: sessionId,
      }),
    });
  }

  async getSession(sessionId) {
    return this.request(`/onboarding/session/${sessionId}`);
  }

  async clearSession(sessionId) {
    return this.request(`/onboarding/session/${sessionId}`, {
      method: "DELETE",
    });
  }

  async getStats() {
    return this.request("/onboarding/stats");
  }

  // Health check methods
  async checkHealth() {
    return this.request("/health");
  }

  async getSystemInfo() {
    return this.request("/");
  }
}

// Export singleton instance
export const onboardingApiService = new OnboardingApiService();
export default onboardingApiService;
