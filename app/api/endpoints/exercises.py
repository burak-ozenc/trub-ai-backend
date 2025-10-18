from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database.connection import get_db
from app.database import crud
from app.core.models import ExerciseResponse, ExerciseCreate
from app.api.endpoints.auth import get_current_user
from app.database.models import User

router = APIRouter(prefix="/exercises", tags=["exercises"])


@router.get("/", response_model=List[ExerciseResponse])
def get_exercises(
        technique: Optional[str] = None,
        difficulty: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get exercises with optional filters
    
    Query params:
    - technique: Filter by technique (breathing, tone, rhythm, etc.)
    - difficulty: Filter by difficulty (beginner, intermediate, advanced)
    """
    exercises = crud.get_exercises(db, technique, difficulty, skip, limit)
    return exercises


@router.get("/recommended", response_model=List[ExerciseResponse])
def get_recommended_exercises(
        technique: Optional[str] = None,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get recommended exercises based on user's skill level
    """
    from app.services.practice_service import PracticeService

    practice_service = PracticeService()
    exercises = practice_service.get_recommended_exercises(
        db,
        current_user.skill_level or "intermediate",
        technique
    )
    return exercises


@router.get("/{exercise_id}", response_model=ExerciseResponse)
def get_exercise(
        exercise_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get a specific exercise by ID"""
    exercise = crud.get_exercise_by_id(db, exercise_id)
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")
    return exercise


@router.post("/", response_model=ExerciseResponse)
def create_exercise(
        exercise: ExerciseCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Create a new exercise (admin only for now)
    TODO: Add admin check
    """
    new_exercise = crud.create_exercise(
        db,
        title=exercise.title,
        technique=exercise.technique,
        difficulty=exercise.difficulty,
        instructions=exercise.instructions,
        description=exercise.description,
        duration_minutes=exercise.duration_minutes,
        sheet_music_url=exercise.sheet_music_url,
        order_index=exercise.order_index
    )
    return new_exercise