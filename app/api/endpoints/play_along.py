"""
Play-Along API Endpoints

Handles play-along session management:
- Start new play-along session
- Submit performance data
- Get session history
- Get session feedback
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database.connection import get_db
from app.database import crud
from app.database.models import User
from app.api.endpoints.auth import get_current_user
from pydantic import BaseModel, Field
from datetime import datetime

router = APIRouter(prefix="/play-along", tags=["play-along"])


# Pydantic Models
class PlayAlongSessionCreate(BaseModel):
    """Request to start play-along session"""
    song_id: int = Field(..., description="Song ID to play")
    difficulty: str = Field(..., description="Difficulty: beginner, intermediate, advanced")


class PerformanceSubmit(BaseModel):
    """Performance data submission"""
    session_id: int
    pitch_accuracy: Optional[float] = Field(None, ge=0, le=100)
    rhythm_accuracy: Optional[float] = Field(None, ge=0, le=100)
    total_score: Optional[float] = Field(None, ge=0, le=100)
    duration_seconds: Optional[int] = Field(None, ge=0)


class PlayAlongSessionResponse(BaseModel):
    """Play-along session response"""
    id: int
    user_id: int
    song_id: int
    difficulty: str
    pitch_accuracy: Optional[float] = None
    rhythm_accuracy: Optional[float] = None
    total_score: Optional[float] = None
    completed: bool
    duration_seconds: Optional[int] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PlayAlongSessionWithSong(BaseModel):
    """Session response with song details"""
    id: int
    song_id: int
    song_title: str
    song_composer: Optional[str]
    difficulty: str
    pitch_accuracy: Optional[float]
    rhythm_accuracy: Optional[float]
    total_score: Optional[float]
    completed: bool
    started_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


@router.post("/start", response_model=dict)
async def start_play_along_session(
        session_data: PlayAlongSessionCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Start a new play-along session
    
    Creates session record and returns song details
    """
    # Verify song exists
    song = crud.get_song_by_id(db, session_data.song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    # Validate difficulty
    if session_data.difficulty not in ['beginner', 'intermediate', 'advanced']:
        raise HTTPException(status_code=400, detail="Invalid difficulty level. Must be: beginner, intermediate, or advanced")

    # Create session
    try:
        session = crud.create_play_along_session(
            db=db,
            user_id=current_user.id,
            song_id=session_data.song_id,
            difficulty=session_data.difficulty
        )

        return {
            "session_id": session.id,
            "song": {
                "id": song.id,
                "title": song.title,
                "composer": song.composer,
                "artist": song.artist,
                "genre": song.genre,
                "tempo": song.tempo,
                "key_signature": song.key_signature,
                "time_signature": song.time_signature,
                "duration_seconds": song.duration_seconds
            },
            "difficulty": session.difficulty,
            "started_at": session.started_at,
            "message": "Session started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


@router.post("/submit-performance")
async def submit_performance(
        performance: PerformanceSubmit,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Submit performance data for a play-along session
    
    Marks session as complete and saves scores
    """
    # Complete the session
    session = crud.complete_play_along_session(
        db=db,
        session_id=performance.session_id,
        user_id=current_user.id,
        pitch_accuracy=performance.pitch_accuracy,
        rhythm_accuracy=performance.rhythm_accuracy,
        total_score=performance.total_score,
        duration_seconds=performance.duration_seconds
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or you don't have permission to access it"
        )

    return {
        "message": "Performance submitted successfully",
        "session_id": session.id,
        "total_score": session.total_score,
        "pitch_accuracy": session.pitch_accuracy,
        "rhythm_accuracy": session.rhythm_accuracy,
        "completed_at": session.completed_at
    }


@router.get("/sessions", response_model=List[PlayAlongSessionResponse])
async def get_play_along_sessions(
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
        completed_only: bool = Query(False, description="Only show completed sessions"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get user's play-along session history
    
    Returns list of sessions with performance data
    """
    sessions = crud.get_user_play_along_sessions(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )

    # Filter by completed if requested
    if completed_only:
        sessions = [s for s in sessions if s.completed]

    return sessions


@router.get("/sessions/{session_id}", response_model=dict)
async def get_session_details(
        session_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific session
    
    Includes song details and performance metrics
    """
    session = crud.get_play_along_session_by_id(db, session_id, current_user.id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get song details
    song = crud.get_song_by_id(db, session.song_id)

    return {
        "session": {
            "id": session.id,
            "difficulty": session.difficulty,
            "pitch_accuracy": session.pitch_accuracy,
            "rhythm_accuracy": session.rhythm_accuracy,
            "total_score": session.total_score,
            "completed": session.completed,
            "duration_seconds": session.duration_seconds,
            "started_at": session.started_at,
            "completed_at": session.completed_at
        },
        "song": {
            "id": song.id,
            "title": song.title,
            "composer": song.composer,
            "artist": song.artist,
            "genre": song.genre,
            "tempo": song.tempo
        } if song else None
    }


@router.get("/stats", response_model=dict)
async def get_play_along_stats(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get user's play-along statistics
    
    Returns aggregate performance metrics
    """
    sessions = crud.get_user_play_along_sessions(
        db=db,
        user_id=current_user.id,
        skip=0,
        limit=1000  # Get all sessions for stats
    )

    completed_sessions = [s for s in sessions if s.completed]

    if not completed_sessions:
        return {
            "total_sessions": len(sessions),
            "completed_sessions": 0,
            "average_score": 0,
            "average_pitch_accuracy": 0,
            "average_rhythm_accuracy": 0,
            "total_practice_time_minutes": 0
        }

    # Calculate averages
    avg_score = sum(s.total_score for s in completed_sessions if s.total_score) / len(completed_sessions)
    avg_pitch = sum(s.pitch_accuracy for s in completed_sessions if s.pitch_accuracy) / len(completed_sessions)
    avg_rhythm = sum(s.rhythm_accuracy for s in completed_sessions if s.rhythm_accuracy) / len(completed_sessions)
    total_time = sum(s.duration_seconds or 0 for s in completed_sessions) / 60  # Convert to minutes

    return {
        "total_sessions": len(sessions),
        "completed_sessions": len(completed_sessions),
        "average_score": round(avg_score, 2),
        "average_pitch_accuracy": round(avg_pitch, 2),
        "average_rhythm_accuracy": round(avg_rhythm, 2),
        "total_practice_time_minutes": round(total_time, 2)
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
        session_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Delete a play-along session
    
    Only allows deleting own sessions
    """
    session = crud.get_play_along_session_by_id(db, session_id, current_user.id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        db.delete(session)
        db.commit()
        return {"message": "Session deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")