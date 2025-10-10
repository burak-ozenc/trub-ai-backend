from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.sql import func
from app.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Profile fields
    bio = Column(Text, nullable=True)
    skill_level = Column(String(50), nullable=True)  # beginner, intermediate, advanced, professional


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Recording metadata
    filename = Column(String(255), nullable=False)
    guidance = Column(Text, nullable=False)
    analysis_type = Column(String(50), default="full")
    duration = Column(Float, nullable=True)  # Duration in seconds

    # Analysis results (stored as JSON)
    analysis_results = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    audio_file_path = Column(String(50), nullable=False)
    # Note: Audio files stored separately (file storage or S3), not in DB
