from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.database.connection import get_db
from app.database import crud
from app.core.models import UserResponse
from app.api.endpoints.auth import get_current_user

router = APIRouter(prefix="/users", tags=["users"])

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    bio: Optional[str] = None
    skill_level: Optional[str] = None

@router.get("/me", response_model=UserResponse)
async def read_user_me(current_user = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_user_me(
        user_update: UserUpdate,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Update current user profile"""
    updated_user = crud.update_user(
        db=db,
        user_id=current_user.id,
        full_name=user_update.full_name,
        bio=user_update.bio,
        skill_level=user_update.skill_level
    )

    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return updated_user