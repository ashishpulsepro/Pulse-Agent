"""
Session Manager
Manages conversation sessions and context
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a message in a conversation"""
    id: str
    session_id: str
    sender: str  # 'user' or 'agent'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'sender': self.sender,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }

@dataclass
class Session:
    """Represents a conversation session"""
    id: str
    created_at: datetime
    last_activity: datetime
    messages: List[Message]
    context: Dict[str, Any]
    
    def __post_init__(self):
        if not self.messages:
            self.messages = []
        if not self.context:
            self.context = {}
    
    def add_message(self, sender: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to the session"""
        message = Message(
            id=str(uuid.uuid4()),
            session_id=self.id,
            sender=sender,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
        return message
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the session context"""
        return {
            'session_id': self.id,
            'message_count': len(self.messages),
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'context': self.context
        }

class SessionManager:
    """
    Manages conversation sessions and maintains context
    """
    
    def __init__(self, session_timeout_hours: int = 24, max_sessions: int = 1000):
        """Initialize session manager"""
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.max_sessions = max_sessions
        
        # Context tracking
        self.global_context = {}
        
        logger.info(f"SessionManager initialized with {session_timeout_hours}h timeout")
    
    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        
        # Clean up old sessions if needed
        self._cleanup_sessions()
        
        session = Session(
            id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            messages=[],
            context={}
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        session = self.sessions.get(session_id)
        
        if session and self._is_session_expired(session):
            self.clear_session(session_id)
            return None
        
        return session
    
    def add_message(self, session_id: str, sender: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[Message]:
        """Add a message to a session"""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None
        
        message = session.add_message(sender, content, metadata)
        
        # Update context based on message
        self._update_session_context(session, message)
        
        logger.debug(f"Added message to session {session_id}: {sender}")
        return message
    
    def get_session_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from a session"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.messages
        if limit:
            messages = messages[-limit:]
        
        return [msg.to_dict() for msg in messages]
    
    def get_recent_context(self, session_id: str, message_count: int = 5) -> Dict[str, Any]:
        """Get recent conversation context"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        recent_messages = session.messages[-message_count:] if session.messages else []
        
        return {
            'session_id': session_id,
            'recent_messages': [msg.to_dict() for msg in recent_messages],
            'session_context': session.context,
            'last_activity': session.last_activity.isoformat()
        }
    
    def update_session_context(self, session_id: str, context_updates: Dict[str, Any]) -> bool:
        """Update session context"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.context.update(context_updates)
        session.last_activity = datetime.now()
        
        logger.debug(f"Updated context for session {session_id}")
        return True
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        active_sessions = []
        
        for session_id, session in self.sessions.items():
            if not self._is_session_expired(session):
                active_sessions.append(session.get_context_summary())
        
        return active_sessions
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about sessions"""
        total_sessions = len(self.sessions)
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        
        # Count active sessions (not expired)
        active_sessions = len([s for s in self.sessions.values() if not self._is_session_expired(s)])
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'total_messages': total_messages,
            'session_timeout_hours': self.session_timeout.total_seconds() / 3600,
            'max_sessions': self.max_sessions
        }
    
    def _is_session_expired(self, session: Session) -> bool:
        """Check if a session has expired"""
        return datetime.now() - session.last_activity > self.session_timeout
    
    def _cleanup_sessions(self) -> None:
        """Clean up expired sessions and enforce max session limit"""
        # Remove expired sessions
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if self._is_session_expired(session)
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        # Enforce max session limit
        if len(self.sessions) > self.max_sessions:
            # Remove oldest sessions
            sessions_by_age = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_activity
            )
            
            sessions_to_remove = len(self.sessions) - self.max_sessions
            for session_id, _ in sessions_by_age[:sessions_to_remove]:
                del self.sessions[session_id]
            
            logger.info(f"Removed {sessions_to_remove} old sessions to enforce limit")
    
    def _update_session_context(self, session: Session, message: Message) -> None:
        """Update session context based on new message"""
        # Extract and store relevant context from messages
        
        # Track mentioned entities
        if 'mentioned_sites' not in session.context:
            session.context['mentioned_sites'] = []
        if 'mentioned_users' not in session.context:
            session.context['mentioned_users'] = []
        
        # Simple entity extraction (can be enhanced)
        content_lower = message.content.lower()
        
        # Look for site mentions
        site_keywords = ['site', 'office', 'location', 'branch']
        for keyword in site_keywords:
            if keyword in content_lower and message.sender == 'user':
                # This is a simplified extraction - could be improved with NLP
                words = message.content.split()
                for i, word in enumerate(words):
                    if keyword.lower() in word.lower() and i + 1 < len(words):
                        potential_site = words[i + 1].strip('.,!?')
                        if potential_site not in session.context['mentioned_sites']:
                            session.context['mentioned_sites'].append(potential_site)
        
        # Store last user intent if this is an agent response with metadata
        if message.sender == 'agent' and message.metadata:
            if 'intent' in message.metadata:
                session.context['last_intent'] = message.metadata['intent']
            if 'action_taken' in message.metadata:
                session.context['last_action'] = message.metadata['action_taken']
        
        # Update conversation flow
        session.context['message_count'] = len(session.messages)
        session.context['last_sender'] = message.sender
