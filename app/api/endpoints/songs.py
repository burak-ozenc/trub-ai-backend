"""
Songs API Endpoints

Handles:
- Song library listing
- Song details retrieval
- Sheet music serving
- Backing track serving
- MIDI file serving
- Play-along session management
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os

from app.database.connection import get_db
from app.database import crud
from app.database.models import User
from app.api.endpoints.auth import get_current_user
from app.core.models import SongResponse, SongListResponse, PlayAlongSessionCreate, PlayAlongSessionResponse
from app.config import settings

router = APIRouter(prefix="/songs", tags=["songs"])


@router.get("/library", response_model=SongListResponse)
async def get_song_library(
        genre: Optional[str] = Query(None, description="Filter by genre"),
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get song library with optional filters
    
    Returns list of all available songs for play-along
    """
    songs = crud.get_songs(db, genre=genre, skip=skip, limit=limit)
    total = crud.get_song_count(db)

    return {
        "songs": songs,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.get("/{song_id}", response_model=SongResponse)
async def get_song_details(
        song_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific song
    """
    song = crud.get_song_by_id(db, song_id)

    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    return song


@router.get("/{song_id}/sheet-music/{difficulty}")
async def get_sheet_music(
        song_id: int,
        difficulty: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get sheet music PDF for specific song and difficulty
    
    Args:
        song_id: Song ID
        difficulty: beginner, intermediate, or advanced
    
    Returns:
        PDF file
    """
    if difficulty not in ['beginner', 'intermediate', 'advanced']:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")

    song = crud.get_song_by_id(db, song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    # Get appropriate sheet music path
    sheet_music_path = getattr(song, f"{difficulty}_sheet_music_path")

    if not sheet_music_path:
        raise HTTPException(
            status_code=404,
            detail=f"Sheet music not available for {difficulty} difficulty"
        )

    # Check if file exists
    if not os.path.exists(sheet_music_path):
        raise HTTPException(status_code=404, detail="Sheet music file not found")

    return FileResponse(
        path=sheet_music_path,
        media_type="application/pdf",
        filename=f"{song.title}_{difficulty}.pdf"
    )


@router.get("/{song_id}/backing-track")
async def get_backing_track(
        song_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get backing track audio for song
    
    Returns:
        WAV audio file
    """
    song = crud.get_song_by_id(db, song_id)
    print('song', song.title)
    # if not song:
    #     raise HTTPException(status_code=404, detail="Song not found")

    if not song.backing_track_path:
        raise HTTPException(status_code=404, detail="Backing track not available")

    # Check if file exists
    if not os.path.exists(song.backing_track_path):
        raise HTTPException(status_code=404, detail="Backing track file not found")

    return FileResponse(
        path=song.backing_track_path,
        media_type="audio/wav",
        filename=f"{song.title}_backing.wav"
    )



@router.get("/{song_id}/midi/{difficulty}")
async def get_song_midi(
        song_id: int,
        difficulty: str,
        current_user: User = Depends(get_current_user),  # ✅ This should work
        db: Session = Depends(get_db)
):
    """
    Get MIDI file for specific song and difficulty
    
    FIXED: Proper authentication and error handling
    """
    # Validate difficulty
    if difficulty not in ['beginner', 'intermediate', 'advanced']:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")

    # Get song from database
    song = crud.get_song_by_id(db, song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    # Get MIDI file path based on difficulty
    midi_path_attr = f"{difficulty}_midi_path"
    midi_path = getattr(song, midi_path_attr, None)

    if not midi_path:
        raise HTTPException(
            status_code=404,
            detail=f"MIDI file not found for {difficulty} difficulty"
        )

    # Check if file exists
    if not os.path.exists(midi_path):
        raise HTTPException(
            status_code=404,
            detail=f"MIDI file not found on server: {midi_path}"
        )

    # Return file
    return FileResponse(
        midi_path,
        media_type="audio/midi",
        filename=f"{song.title}_{difficulty}.mid"
    )


@router.get("/{song_id}/sheet-music/{difficulty}")
async def get_song_sheet_music(
        song_id: int,
        difficulty: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get sheet music PDF for specific song and difficulty
    
    FIXED: Proper authentication
    """
    if difficulty not in ['beginner', 'intermediate', 'advanced']:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")

    song = crud.get_song_by_id(db, song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    sheet_music_attr = f"{difficulty}_sheet_music_path"
    sheet_music_path = getattr(song, sheet_music_attr, None)

    if not sheet_music_path or not os.path.exists(sheet_music_path):
        raise HTTPException(
            status_code=404,
            detail=f"Sheet music not found for {difficulty} difficulty"
        )

    return FileResponse(
        sheet_music_path,
        media_type="application/pdf",
        filename=f"{song.title}_{difficulty}.pdf"
    )


@router.get("/{song_id}/backing-track")
async def get_song_backing_track(
        song_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get backing track for song
    
    FIXED: Proper authentication
    """
    song = crud.get_song_by_id(db, song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    if not song.backing_track_path or not os.path.exists(song.backing_track_path):
        raise HTTPException(status_code=404, detail="Backing track not found")

    # Determine media type based on file extension
    file_ext = Path(song.backing_track_path).suffix.lower()
    media_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.mid': 'audio/midi',
        '.midi': 'audio/midi'
    }
    media_type = media_types.get(file_ext, 'audio/mpeg')

    return FileResponse(
        song.backing_track_path,
        media_type=media_type,
        filename=f"{song.title}_backing{file_ext}"
    )