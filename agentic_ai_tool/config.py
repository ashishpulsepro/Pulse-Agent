"""
Configuration for Agentic AI Tool
"""

import os
from typing import Optional

class Config:
    """Configuration settings for the AI agent"""
    
    # API Configuration
    PULSE_PRO_API_BASE_URL: str = os.getenv("PULSE_PRO_API_BASE_URL", "https://staging-api.pulsepro.ai")
    PULSE_PRO_FRONTEND_URL: str = os.getenv("PULSE_PRO_FRONTEND_URL", "https://staging.pulsepro.ai")
    
    # Authentication
    REFRESH_TOKEN: Optional[str] = os.getenv("refresh")
    
    # Server Configuration
    SERVER_HOST: str = os.getenv("HOST", "127.0.0.1")
    SERVER_PORT: int = int(os.getenv("PORT", 8000))
    
    # Session Management
    SESSION_TIMEOUT_HOURS: int = int(os.getenv("SESSION_TIMEOUT_HOURS", 24))
    MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", 1000))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173"
    ]
    
    # Features
    ENABLE_DEBUG_ENDPOINTS: bool = os.getenv("ENABLE_DEBUG_ENDPOINTS", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.REFRESH_TOKEN:
            print("Warning: REFRESH_TOKEN not set. Some features may not work.")
            return False
        return True

# Create global config instance
config = Config()
