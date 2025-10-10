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


@router.get("/stats/progress")
async def get_progress_stats(
        current_user = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get progress statistics and trends for current user"""
    recordings = crud.get_user_recordings(db, current_user.id, skip=0, limit=100)

    if not recordings:
        return {
            "total_recordings": 0,
            "total_practice_time": 0,
            "trends": [],
            "averages": {},
            "latest_scores": {}
        }

    # Calculate trends over time
    trends = []
    for recording in recordings:
        analysis = recording.analysis_results or {}

        # Extract scores from analysis
        breath = analysis.get('breath_control', {})
        tone = analysis.get('tone_quality', {})
        rhythm = analysis.get('rhythm_timing', {})
        expression = analysis.get('expression', {})
        flexibility = analysis.get('flexibility', {})

        trend_point = {
            "date": recording.created_at.isoformat(),
            "breath_score": _extract_score(breath),
            "tone_score": _extract_score(tone),
            "rhythm_score": rhythm.get('timing_deviation', 0) if rhythm else 0,
            "expression_score": expression.get('dynamic_range', 0) if expression else 0,
            "flexibility_score": flexibility.get('transition_smoothness', 0) if flexibility else 0,
        }
        trends.append(trend_point)

    # Calculate averages
    total = len(recordings)
    avg_breath = sum(t['breath_score'] for t in trends) / total if total > 0 else 0
    avg_tone = sum(t['tone_score'] for t in trends) / total if total > 0 else 0
    avg_rhythm = sum(t['rhythm_score'] for t in trends) / total if total > 0 else 0

    # Get latest scores
    latest = trends[0] if trends else {}

    # Calculate total practice time (estimate)
    total_time = sum(r.duration or 0 for r in recordings)

    return {
        "total_recordings": total,
        "total_practice_time": round(total_time, 2),
        "trends": list(reversed(trends)),  # Oldest to newest for charts
        "averages": {
            "breath": round(avg_breath, 2),
            "tone": round(avg_tone, 2),
            "rhythm": round(avg_rhythm, 2)
        },
        "latest_scores": latest
    }


def _extract_score(analysis_section: dict) -> float:
    """Extract a numeric score from analysis section"""
    if not analysis_section:
        return 0.0

    # Try to find numeric values
    if 'average_breath_length' in analysis_section:
        return min(analysis_section['average_breath_length'], 10.0)
    if 'harmonic_ratio' in analysis_section:
        return analysis_section['harmonic_ratio'] * 10

    return 0.0