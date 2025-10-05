from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import os

from app.database.connection import get_db
from app.database import crud
from app.core.models import RecordingCreate, RecordingResponse
from app.api.endpoints.auth import get_current_user
from app.services.file_service import FileService

router = APIRouter(prefix="/recordings", tags=["recordings"])


def get_file_service() -> FileService:
    return FileService()


@router.post("/", response_model=RecordingResponse, status_code=status.HTTP_201_CREATED)
async def create_recording(
        recording: RecordingCreate,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Save a new recording with analysis results"""
    db_recording = crud.create_recording(
        db=db,
        user_id=current_user.id,
        filename=recording.filename,
        guidance=recording.guidance,
        analysis_results=recording.analysis_results,
        analysis_type=recording.analysis_type,
        duration=recording.duration,
        audio_file_path=recording.audio_file_path  # NEW - Save audio file path
    )
    return db_recording


@router.get("/", response_model=List[RecordingResponse])
async def get_my_recordings(
        skip: int = 0,
        limit: int = 50,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get all recordings for current user"""
    recordings = crud.get_user_recordings(db, current_user.id, skip, limit)
    return recordings


@router.get("/{recording_id}", response_model=RecordingResponse)
async def get_recording(
        recording_id: int,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get a specific recording"""
    recording = crud.get_recording_by_id(db, recording_id, current_user.id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found"
        )
    return recording


@router.get("/{recording_id}/audio")
async def get_recording_audio(
        recording_id: int,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db),
        file_service: FileService = Depends(get_file_service)
):
    """Get audio file for a recording"""
    # Verify recording belongs to user
    recording = crud.get_recording_by_id(db, recording_id, current_user.id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found"
        )

    # Check if audio file path exists
    if not recording.audio_file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found for this recording"
        )

    # Check if file exists on disk
    if not file_service.file_exists(recording.audio_file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found on server"
        )

    # Return audio file
    return FileResponse(
        path=recording.audio_file_path,
        media_type="audio/wav",
        filename=recording.filename
    )


@router.delete("/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recording(
        recording_id: int,
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db),
        file_service: FileService = Depends(get_file_service)
):
    """Delete a recording and its audio file"""
    # Get recording first to get audio file path
    recording = crud.get_recording_by_id(db, recording_id, current_user.id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found"
        )

    # Delete audio file if exists
    if recording.audio_file_path:
        file_service.cleanup_file(recording.audio_file_path)

    # Delete database record
    success = crud.delete_recording(db, recording_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found"
        )


@router.get("/stats/count")
async def get_recording_stats(
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get recording statistics for current user"""
    count = crud.get_recording_count(db, current_user.id)
    return {"total_recordings": count}