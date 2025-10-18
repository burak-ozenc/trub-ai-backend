from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, date, timedelta
from app.database.connection import get_db
from app.database import crud
from app.core.models import (
    CalendarEntryCreate,
    CalendarEntryUpdate,
    CalendarEntryResponse,
    CalendarEntryWithExercise
)
from app.api.endpoints.auth import get_current_user
from app.database.models import User

router = APIRouter(prefix="/calendar", tags=["calendar"])


@router.post("/entries", response_model=CalendarEntryResponse)
def create_calendar_entry(
        entry_data: CalendarEntryCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Schedule a practice session"""
    # Verify exercise exists
    exercise = crud.get_exercise_by_id(db, entry_data.exercise_id)
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")

    entry = crud.create_calendar_entry(
        db,
        current_user.id,
        entry_data.exercise_id,
        entry_data.scheduled_date,
        entry_data.scheduled_time,
        entry_data.duration_minutes,
        entry_data.notes
    )
    return entry


@router.get("/entries", response_model=List[CalendarEntryResponse])
def get_calendar_entries(
        start_date: datetime = Query(..., description="Start of date range"),
        end_date: datetime = Query(..., description="End of date range"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get calendar entries for a date range"""
    entries = crud.get_calendar_entries_by_date_range(
        db,
        current_user.id,
        start_date,
        end_date
    )
    return entries


@router.get("/entries/date/{target_date}")
def get_entries_by_date(
        target_date: date,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get calendar entries for a specific date with exercise details"""
    entries = crud.get_calendar_entries_by_date(db, current_user.id, target_date)

    # Manually load exercise data for each entry
    result = []
    for entry in entries:
        exercise = crud.get_exercise_by_id(db, entry.exercise_id)
        result.append({
            **entry.__dict__,
            "exercise": exercise.__dict__ if exercise else None
        })

    return result


@router.get("/entries/upcoming", response_model=List[CalendarEntryResponse])
def get_upcoming_practices(
        limit: int = Query(10, ge=1, le=50),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get upcoming scheduled practices"""
    entries = crud.get_upcoming_practices(db, current_user.id, limit)
    return entries


@router.get("/entries/{entry_id}", response_model=CalendarEntryResponse)
def get_calendar_entry(
        entry_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get a specific calendar entry"""
    entry = crud.get_calendar_entry_by_id(db, entry_id, current_user.id)
    if not entry:
        raise HTTPException(status_code=404, detail="Calendar entry not found")
    return entry


@router.put("/entries/{entry_id}", response_model=CalendarEntryResponse)
def update_calendar_entry(
        entry_id: int,
        update_data: CalendarEntryUpdate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Update a calendar entry"""
    entry = crud.update_calendar_entry(
        db,
        entry_id,
        current_user.id,
        **update_data.dict(exclude_unset=True)
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Calendar entry not found")
    return entry


@router.post("/entries/{entry_id}/complete", response_model=CalendarEntryResponse)
def complete_calendar_entry(
        entry_id: int,
        practice_session_id: int = None,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Mark a calendar entry as completed"""
    entry = crud.mark_calendar_entry_complete(
        db,
        entry_id,
        current_user.id,
        practice_session_id
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Calendar entry not found")
    return entry


@router.delete("/entries/{entry_id}")
def delete_calendar_entry(
        entry_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Delete a calendar entry"""
    success = crud.delete_calendar_entry(db, entry_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Calendar entry not found")
    return {"message": "Calendar entry deleted successfully"}