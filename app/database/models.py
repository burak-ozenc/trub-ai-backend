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

class Exercise(Base):
    __tablename__ = "exercises"

    id = Column(Integer, primary_key=True, index=True)

    # Exercise identification
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    technique = Column(String(50), nullable=False)  # breathing, tone, rhythm, etc.
    difficulty = Column(String(50), nullable=False)  # beginner, intermediate, advanced

    # Exercise content
    instructions = Column(Text, nullable=False)  # Step-by-step instructions
    duration_minutes = Column(Integer, nullable=True)  # Recommended duration
    sheet_music_url = Column(String(500), nullable=True)  # PDF URL or path

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
    order_index = Column(Integer, default=0)  # For ordering exercises