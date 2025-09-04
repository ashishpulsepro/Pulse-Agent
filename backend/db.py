from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

# Your Django-style DB settings
DB_CONFIG = {
    "ENGINE": "django.contrib.gis.db.backends.postgis",
    "NAME": "pulse_staging",
    "USER": "postgres",
    "PASSWORD": "Pulse@123!",
    "HOST": "152.42.156.131",
    "PORT": "5432",
}

from urllib.parse import quote_plus

password = quote_plus("Pulse@123!")  # encodes special chars if any
DATABASE_URL = f"postgresql+psycopg2://postgres:{password}@152.42.156.131:5432/pulse_staging"


# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def initialize_db():
    """Return engine and sessionmaker for external usage."""
    return engine, SessionLocal


def get_db_session():
    """FastAPI dependency - yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()