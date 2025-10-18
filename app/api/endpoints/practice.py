from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database.connection import get_db
from app.database import crud
from app.core.models import (
    PracticeSessionCreate,
    PracticeSessionComplete,
    PracticeSessionResponse,
    SimpleFeedbackResult
)
from app.api.endpoints.auth import get_current_user
from app.database.models import User
from app.services.practice_service import PracticeService

router = APIRouter(prefix="/practice", tags=["practice"])


@router.post("/sessions", response_model=PracticeSessionResponse)
def start_practice_session(
        session_data: PracticeSessionCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Start a new practice session"""
    practice_service = PracticeService()

    # Verify exercise exists
    exercise = crud.get_exercise_by_id(db, session_data.exercise_id)
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")

    session = practice_service.start_practice_session(
        db,
        current_user.id,
        session_data.exercise_id
    )
    return session


@router.put("/sessions/{session_id}/complete", response_model=PracticeSessionResponse)
def complete_practice_session(
        session_id: int,
        completion_data: PracticeSessionComplete,
        calendar_entry_id: int = None,  # NEW: Accept calendar_entry_id
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Mark a practice session as complete"""
    practice_service = PracticeService()

    # Get the session to find exercise
    session = db.query(crud.PracticeSession).filter(
        crud.PracticeSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    exercise = crud.get_exercise_by_id(db, session.exercise_id)

    # If calendar_entry_id provided, use the calendar-aware completion
    if calendar_entry_id:
        completed_session = practice_service.complete_practice_session_with_calendar(
            db,
            session_id,
            current_user.id,
            calendar_entry_id,
            completion_data.duration_seconds,
            completion_data.recording_id,
            None,  # analysis_result - would be fetched from recording
            exercise.technique
        )
    else:
        # Regular completion without calendar
        completed_session = practice_service.complete_practice_session(
            db,
            session_id,
            current_user.id,
            completion_data.duration_seconds,
            completion_data.recording_id,
            None,
            exercise.technique
        )

    if not completed_session:
        raise HTTPException(status_code=404, detail="Failed to complete session")

    return completed_session


@router.get("/sessions", response_model=List[PracticeSessionResponse])
def get_practice_sessions(
        skip: int = 0,
        limit: int = 50,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get user's practice session history"""
    sessions = crud.get_user_practice_sessions(db, current_user.id, skip, limit)
    return sessions


@router.get("/sessions/{session_id}/feedback")
def get_session_feedback(
        session_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get simplified feedback for a practice session"""
    practice_service = PracticeService()

    session = db.query(crud.PracticeSession).filter(
        crud.PracticeSession.id == session_id,
        crud.PracticeSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Practice session not found")

    if not session.simplified_feedback:
        return {"message": "No feedback available for this session"}

    feedback = practice_service.parse_simplified_feedback(session.simplified_feedback)
    return feedback

@router.post("/sessions/from-calendar/{calendar_entry_id}", response_model=PracticeSessionResponse)
def start_practice_from_calendar(
        calendar_entry_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Start a practice session from a calendar entry"""
    from app.services.practice_service import PracticeService

    practice_service = PracticeService()
    session, calendar_entry = practice_service.start_practice_session_from_calendar(
        db,
        current_user.id,
        calendar_entry_id
    )

    if not session:
        raise HTTPException(status_code=404, detail="Calendar entry not found")

    return session