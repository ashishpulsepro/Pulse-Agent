from pymongo import MongoClient
from datetime import datetime
import urllib.parse
import logging
import os
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db():
    """get db initialized"""
    try:
        # username = os.getenv('username')
        # password = urllib.parse.quote_plus(os.getenv('password'))  # URL encode the password
        MONGO_URI = os.getenv('MONGO_URI')
        client_mongo = MongoClient(MONGO_URI)
        db = client_mongo.Conversations
        conversations_collection = db.conversations
        
        return conversations_collection
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None