from sqlalchemy.orm import Session
from typing import Optional, List
from app.database.models import User, Recording
from app.core.security import get_password_hash, verify_password

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
                     duration: float = None) -> Recording:
    """Create new recording for a user"""
    recording = Recording(
        user_id=user_id,
        filename=filename,
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