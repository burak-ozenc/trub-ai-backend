import datetime

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


class PracticeSession(Base):
    __tablename__ = "practice_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    exercise_id = Column(Integer, ForeignKey("exercises.id"), nullable=False, index=True)

    # Session data
    duration_seconds = Column(Integer, nullable=True)  # Actual duration
    completed = Column(Boolean, default=False)

    # Analysis results (if they recorded during session)
    recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=True)
    simplified_feedback = Column(Text, nullable=True)  # Simple, actionable feedback

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

class CalendarEntry(Base):
    __tablename__ = "calendar_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    exercise_id = Column(Integer, ForeignKey("exercises.id"), nullable=False, index=True)

    # Schedule info
    scheduled_date = Column(DateTime(timezone=True), nullable=False, index=True)  # Date for practice
    scheduled_time = Column(String(10), nullable=True)  # Optional time like "14:00"
    duration_minutes = Column(Integer, nullable=True)  # Planned duration

    # Completion tracking
    completed = Column(Boolean, default=False)
    practice_session_id = Column(Integer, ForeignKey("practice_sessions.id"), nullable=True)  # Link to actual session

    # Notes
    notes = Column(Text, nullable=True)  # User's notes for this scheduled practice

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Song(Base):
    __tablename__ = "songs"

    id = Column(Integer, primary_key=True, index=True)

    # Song identification
    title = Column(String(255), nullable=False)
    composer = Column(String(255), nullable=True)
    artist = Column(String(255), nullable=True)
    genre = Column(String(50), nullable=False)  # classical, folk, christmas, jazz

    # File paths (relative to data/songs/)
    beginner_midi_path = Column(String(500), nullable=True)
    intermediate_midi_path = Column(String(500), nullable=True)
    advanced_midi_path = Column(String(500), nullable=True)

    beginner_sheet_music_path = Column(String(500), nullable=True)  # PDF path
    intermediate_sheet_music_path = Column(String(500), nullable=True)
    advanced_sheet_music_path = Column(String(500), nullable=True)

    backing_track_path = Column(String(500), nullable=True)  # WAV path

    # Musical metadata
    tempo = Column(Integer, nullable=True)  # BPM
    key_signature = Column(String(10), nullable=True)  # "Bb", "C", "F", etc.
    time_signature = Column(String(10), nullable=True)  # "4/4", "3/4", "6/8"
    duration_seconds = Column(Integer, nullable=True)

    # Legal & content management
    is_public_domain = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    order_index = Column(Integer, default=0)  # For sorting display

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class PlayAlongSession(Base):
    __tablename__ = "play_along_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    song_id = Column(Integer, ForeignKey("songs.id"), nullable=False, index=True)

    difficulty = Column(String(50), nullable=False)  # beginner, intermediate, advanced

    # Performance metrics
    pitch_accuracy = Column(Float, nullable=True)  # 0-100
    rhythm_accuracy = Column(Float, nullable=True)  # 0-100
    total_score = Column(Float, nullable=True)  # 0-100

    # Session data
    completed = Column(Boolean, default=False)
    duration_seconds = Column(Integer, nullable=True)
    recording_path = Column(String(500), nullable=True)  # Optional recording file

    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)