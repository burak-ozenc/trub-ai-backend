from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any

from app.services.audio_processor import AudioProcessorService
from app.services.file_service import FileService
from app.core.models import AnalysisType
from app.core.exceptions import AudioProcessingError, FileProcessingError, AnalysisError

router = APIRouter(prefix="/audio", tags=["audio"])


def get_audio_processor() -> AudioProcessorService:
    return AudioProcessorService()


def get_file_service() -> FileService:
    return FileService()


@router.post("/analyze")
async def analyze_audio(
        audioData: UploadFile = File(...),
        analysis_type: str = Form(default="full"),
        file_service: FileService = Depends(get_file_service),
        audio_processor: AudioProcessorService = Depends(get_audio_processor)
) -> Dict[str, Any]:
    """
    Analyze audio file without LLM feedback
    
    Args:
        audioData: Audio file to analyze
        analysis_type: Type of analysis (full, breath, tone, etc.)
        
    Returns:
        Technical analysis results
    """
    try:
        # Validate analysis type
        try:
            analysis_enum = AnalysisType(analysis_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis type. Must be one of: {[t.value for t in AnalysisType]}"
            )

        # Save uploaded file
        file_path = await file_service.save_uploaded_file(audioData)

        try:
            # Analyze audio
            analysis_result = audio_processor.analyze_audio(file_path, analysis_enum)

            # Extract technical data
            technical_data = audio_processor.extract_technical_data(analysis_result)

            return {
                "analysis_type": analysis_type,
                "technical_analysis": technical_data,
                "raw_results": analysis_result.dict(exclude_none=True),
                "file_path": file_path
            }

        finally:
            # Optional: cleanup file after processing
            # file_service.cleanup_file(file_path)
            pass

    except FileProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (AudioProcessingError, AnalysisError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/analyze-breath")
async def analyze_breath_control(
        audioData: UploadFile = File(...),
        file_service: FileService = Depends(get_file_service),
        audio_processor: AudioProcessorService = Depends(get_audio_processor)
) -> Dict[str, Any]:
    """
    Analyze only breath control in audio
    
    Args:
        audioData: Audio file to analyze
        
    Returns:
        Breath control analysis results
    """
    try:
        # Save uploaded file
        file_path = await file_service.save_uploaded_file(audioData)

        try:
            # Analyze breath control only
            analysis_result = audio_processor.analyze_audio(file_path, AnalysisType.BREATH)

            return {
                "breath_analysis": analysis_result.breath_control.dict() if analysis_result.breath_control else None,
                "file_path": file_path
            }

        finally:
            # Optional: cleanup file after processing
            # file_service.cleanup_file(file_path)
            pass

    except FileProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (AudioProcessingError, AnalysisError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/analyze-tone")
async def analyze_tone_quality(
        audioData: UploadFile = File(...),
        file_service: FileService = Depends(get_file_service),
        audio_processor: AudioProcessorService = Depends(get_audio_processor)
) -> Dict[str, Any]:
    """
    Analyze only tone quality in audio
    
    Args:
        audioData: Audio file to analyze
        
    Returns:
        Tone quality analysis results
    """
    try:
        # Save uploaded file
        file_path = await file_service.save_uploaded_file(audioData)

        try:
            # Analyze tone quality only
            analysis_result = audio_processor.analyze_audio(file_path, AnalysisType.TONE)

            return {
                "tone_analysis": analysis_result.tone_quality.dict() if analysis_result.tone_quality else None,
                "file_path": file_path
            }

        finally:
            # Optional: cleanup file after processing
            # file_service.cleanup_file(file_path)
            pass

    except FileProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (AudioProcessingError, AnalysisError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
