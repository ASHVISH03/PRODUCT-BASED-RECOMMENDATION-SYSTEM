from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config.configuration import ConfigurationManager
import os
from pathlib import Path

# Define the SQLite database path
# Using a local sqlite file in the data/external directory
project_root = Path(__file__).resolve().parent.parent
DB_DIR = project_root / "data/external"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "ecommerce.db"

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# create_engine in SQLite requires connect_args={"check_same_thread": False}
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    Dependency to get a SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
