from sqlalchemy.orm import Session
from typing import Optional, List
from app.database.models import User, Recording
from app.core.security import get_password_hash, verify_password
from app.database.models import Exercise, PracticeSession
from app.database.models import CalendarEntry
from datetime import datetime, date, timedelta

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()

def create_user(db: Session, email: str, username: str, password: str,
                full_name: Optional[str] = None) -> User:
    """Create new user"""
    hashed_password = get_password_hash(password)
    db_user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        full_name=full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def update_user(db: Session, user_id: int, **kwargs) -> Optional[User]:
    """Update user fields"""
    user = get_user_by_id(db, user_id)
    if not user:
        return None

    for key, value in kwargs.items():
        if hasattr(user, key) and value is not None:
            setattr(user, key, value)

    db.commit()
    db.refresh(user)
    return user

# Recording CRUD operations
def create_recording(db: Session, user_id: int, filename: str, guidance: str,
                     analysis_results: dict, analysis_type: str = "full",
                     duration: float = None, audio_file_path: str = None) -> Recording:
    """Create new recording for a user"""
    recording = Recording(
        user_id=user_id,
        filename=filename,
        audio_file_path=audio_file_path,  # NEW - Store audio file path
        guidance=guidance,
        analysis_type=analysis_type,
        duration=duration,
        analysis_results=analysis_results
    )
    db.add(recording)
    db.commit()
    db.refresh(recording)
    return recording

def get_user_recordings(db: Session, user_id: int, skip: int = 0, limit: int = 50) -> List[Recording]:
    """Get all recordings for a user"""
    return db.query(Recording).filter(
        Recording.user_id == user_id
    ).order_by(Recording.created_at.desc()).offset(skip).limit(limit).all()

def get_recording_by_id(db: Session, recording_id: int, user_id: int) -> Optional[Recording]:
    """Get a specific recording (with user verification)"""
    return db.query(Recording).filter(
        Recording.id == recording_id,
        Recording.user_id == user_id
    ).first()

def delete_recording(db: Session, recording_id: int, user_id: int) -> bool:
    """Delete a recording"""
    recording = get_recording_by_id(db, recording_id, user_id)
    if not recording:
        return False

    db.delete(recording)
    db.commit()
    return True

def get_recording_count(db: Session, user_id: int) -> int:
    """Get total recording count for user"""
    return db.query(Recording).filter(Recording.user_id == user_id).count()

# Exercise CRUD
def get_exercises(db: Session, technique: Optional[str] = None,
                  difficulty: Optional[str] = None,
                  skip: int = 0, limit: int = 50) -> List[Exercise]:
    """Get exercises with optional filters"""
    query = db.query(Exercise).filter(Exercise.is_active == True)

    if technique:
        query = query.filter(Exercise.technique == technique)
    if difficulty:
        query = query.filter(Exercise.difficulty == difficulty)

    return query.order_by(Exercise.order_index).offset(skip).limit(limit).all()


def get_exercise_by_id(db: Session, exercise_id: int) -> Optional[Exercise]:
    """Get exercise by ID"""
    return db.query(Exercise).filter(
        Exercise.id == exercise_id,
        Exercise.is_active == True
    ).first()


def create_exercise(db: Session, title: str, technique: str, difficulty: str,
                    instructions: str, description: str = None,
                    duration_minutes: int = None, sheet_music_url: str = None,
                    order_index: int = 0) -> Exercise:
    """Create new exercise"""
    exercise = Exercise(
        title=title,
        description=description,
        technique=technique,
        difficulty=difficulty,
        instructions=instructions,
        duration_minutes=duration_minutes,
        sheet_music_url=sheet_music_url,
        order_index=order_index
    )
    db.add(exercise)
    db.commit()
    db.refresh(exercise)
    return exercise


# Practice Session CRUD
def create_practice_session(db: Session, user_id: int, exercise_id: int) -> PracticeSession:
    """Create new practice session"""
    session = PracticeSession(
        user_id=user_id,
        exercise_id=exercise_id
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def complete_practice_session(db: Session, session_id: int, user_id: int,
                              duration_seconds: int = None,
                              recording_id: int = None,
                              simplified_feedback: str = None) -> Optional[PracticeSession]:
    """Mark practice session as complete"""
    session = db.query(PracticeSession).filter(
        PracticeSession.id == session_id,
        PracticeSession.user_id == user_id
    ).first()

    if not session:
        return None

    session.completed = True
    session.completed_at = datetime.utcnow()
    session.duration_seconds = duration_seconds
    session.recording_id = recording_id
    session.simplified_feedback = simplified_feedback

    db.commit()
    db.refresh(session)
    return session


def get_user_practice_sessions(db: Session, user_id: int,
                               skip: int = 0, limit: int = 50) -> List[PracticeSession]:
    """Get user's practice sessions"""
    return db.query(PracticeSession).filter(
        PracticeSession.user_id == user_id
    ).order_by(PracticeSession.started_at.desc()).offset(skip).limit(limit).all()

# Calendar CRUD
def create_calendar_entry(
        db: Session,
        user_id: int,
        exercise_id: int,
        scheduled_date: datetime,
        scheduled_time: str = None,
        duration_minutes: int = None,
        notes: str = None
) -> CalendarEntry:
    """Create a calendar entry (scheduled practice)"""
    entry = CalendarEntry(
        user_id=user_id,
        exercise_id=exercise_id,
        scheduled_date=scheduled_date,
        scheduled_time=scheduled_time,
        duration_minutes=duration_minutes,
        notes=notes
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_calendar_entries_by_date_range(
        db: Session,
        user_id: int,
        start_date: datetime,
        end_date: datetime
) -> List[CalendarEntry]:
    """Get calendar entries for a date range"""
    return db.query(CalendarEntry).filter(
        CalendarEntry.user_id == user_id,
        CalendarEntry.scheduled_date >= start_date,
        CalendarEntry.scheduled_date <= end_date
    ).order_by(CalendarEntry.scheduled_date).all()


def get_calendar_entries_by_date(
        db: Session,
        user_id: int,
        target_date: date
) -> List[CalendarEntry]:
    """Get calendar entries for a specific date"""
    start_of_day = datetime.combine(target_date, datetime.min.time())
    end_of_day = datetime.combine(target_date, datetime.max.time())

    return db.query(CalendarEntry).filter(
        CalendarEntry.user_id == user_id,
        CalendarEntry.scheduled_date >= start_of_day,
        CalendarEntry.scheduled_date <= end_of_day
    ).order_by(CalendarEntry.scheduled_date).all()


def get_calendar_entry_by_id(
        db: Session,
        entry_id: int,
        user_id: int
) -> Optional[CalendarEntry]:
    """Get a specific calendar entry"""
    return db.query(CalendarEntry).filter(
        CalendarEntry.id == entry_id,
        CalendarEntry.user_id == user_id
    ).first()


def update_calendar_entry(
        db: Session,
        entry_id: int,
        user_id: int,
        **kwargs
) -> Optional[CalendarEntry]:
    """Update calendar entry"""
    entry = get_calendar_entry_by_id(db, entry_id, user_id)
    if not entry:
        return None

    for key, value in kwargs.items():
        if hasattr(entry, key) and value is not None:
            setattr(entry, key, value)

    db.commit()
    db.refresh(entry)
    return entry


def mark_calendar_entry_complete(
        db: Session,
        entry_id: int,
        user_id: int,
        practice_session_id: int = None
) -> Optional[CalendarEntry]:
    """Mark a calendar entry as completed"""
    entry = get_calendar_entry_by_id(db, entry_id, user_id)
    if not entry:
        return None

    entry.completed = True
    entry.practice_session_id = practice_session_id

    db.commit()
    db.refresh(entry)
    return entry


def delete_calendar_entry(
        db: Session,
        entry_id: int,
        user_id: int
) -> bool:
    """Delete a calendar entry"""
    entry = get_calendar_entry_by_id(db, entry_id, user_id)
    if not entry:
        return False

    db.delete(entry)
    db.commit()
    return True


def get_upcoming_practices(
        db: Session,
        user_id: int,
        limit: int = 10
) -> List[CalendarEntry]:
    """Get upcoming scheduled practices"""
    now = datetime.utcnow()
    return db.query(CalendarEntry).filter(
        CalendarEntry.user_id == user_id,
        CalendarEntry.scheduled_date >= now,
        CalendarEntry.completed == False
    ).order_by(CalendarEntry.scheduled_date).limit(limit).all()