"""
SQLite database via SQLAlchemy.
Tables are created automatically on first startup — no migration scripts needed.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings

# SQLite URL — file is created automatically if it doesn't exist
_db_url = f"sqlite:///{settings.sqlite_db_path}"

engine = create_engine(
    _db_url,
    connect_args={"check_same_thread": False},  # needed for FastAPI threading
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session and closes it afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Called once at startup to create all tables if they don't exist yet."""
    from app.models import Project, Document  # noqa: F401 — registers models
    Base.metadata.create_all(bind=engine)
