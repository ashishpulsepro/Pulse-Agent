from pymongo import MongoClient
from datetime import datetime
import uuid
import urllib.parse



import logging
import json


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# MongoDB connection setup----------------------------------------------------------------------------------------
username = "ashish"
password = urllib.parse.quote_plus("Radhey@123")  # URL encode the password
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.3uxl669.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client_mongo = MongoClient(MONGO_URI)
db = client_mongo.Conversations
conversations_collection = db.conversations






#services------------------------------------------------------------------------------------------------------------------

def save_conversation_to_db(session_id: str, role: str, message: str,email:str=None, intent: str = None):
    """Save message to MongoDB"""
    try:
        conversations_collection.insert_one({
            "session_id": session_id,
            "email":email,
            "role": role,
            "message": message,
            "timestamp": datetime.now(),
            "intent": intent
        })
    except Exception as e:
        logger.error(f"Failed to save to MongoDB: {e}")

def get_conversation_from_db(session_id: str) -> list:
    """Get conversation history from MongoDB"""
    try:
        messages = conversations_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1)
        return list(messages)
    except Exception as e:
        logger.error(f"Failed to get from MongoDB: {e}")
        return []
    

def get_all_session_ids(email:str) -> list:
    """Fetch all unique session IDs from MongoDB"""
    try:
        session_ids = conversations_collection.distinct(
            "session_id",  # field to get unique values of
            {"email": email}  # filter condition
        )
        return session_ids
    except Exception as e:
        logger.error(f"Failed to fetch session_ids from MongoDB: {e}")
        return []



def clear_conversation_from_db(session_id: str):
    """Clear conversation history from MongoDB"""
    try:
        conversations_collection.delete_many({"session_id": session_id})
    except Exception as e:
        logger.error(f"Failed to clear from MongoDB: {e}")

def get_session_intent(session_id):
    """Get the intent for a session from MongoDB"""
    try:
        # Find any document with the session_id that has an intent field
        session = conversations_collection.find_one(
            {"session_id": session_id, "intent": {"$exists": True}},
            sort=[("timestamp", -1)]  # Get the most recent one
        )
        print("session intent inside get_session_intent : ", session)
        return session['intent'] if session else None
    except Exception as e:
        logger.error(f"Failed to get intent from MongoDB: {e}")
        return None


def store_session_intent(session_id, session_intent):
    """Update the intent for all documents in a session"""
    try:
        # Update intent for all documents with the given session_id
        result = conversations_collection.update_many(
            {"session_id": session_id},  # Match all documents with this session_id
            {"$set": {"intent": session_intent, "timestamp": datetime.utcnow()}},
            upsert=False  # do NOT create new documents
        )

        if result.matched_count == 0:
            logger.warning(f"No documents found for session {session_id}, nothing updated")
            return False

        logger.info(f"Intent '{session_intent}' updated for {result.modified_count} documents in session {session_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to update intent in MongoDB: {e}")
        return False


def safe_extract_text(response):
    try:
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
        # fallback
        return ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
