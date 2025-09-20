from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Dict, Any

from app.services.audio_processor import AudioProcessorService
from app.services.llm_service import LLMService
from app.services.file_service import FileService
from app.core.models import AnalysisType, LLMResponse
from app.core.exceptions import AudioProcessingError, FileProcessingError, LLMServiceError, AnalysisError

router = APIRouter(prefix="/analysis", tags=["analysis"])

def get_audio_processor() -> AudioProcessorService:
    return AudioProcessorService()

def get_llm_service() -> LLMService:
    return LLMService()

def get_file_service() -> FileService:
    return FileService()

@router.post("/comprehensive")
async def comprehensive_analysis(
        audioData: UploadFile = File(...),
        guidance: str = Form(..., description="User's question or guidance text"),
        analysis_type: str = Form(default="full"),
        file_service: FileService = Depends(get_file_service),
        audio_processor: AudioProcessorService = Depends(get_audio_processor),
        llm_service: LLMService = Depends(get_llm_service)
) -> Dict[str, Any]:
    """
    Comprehensive analysis with LLM feedback
    
    Main endpoint that combines technical analysis with AI-powered feedback
    
    Args:
        audioData: Audio file to analyze
        guidance: User's question or guidance text
        analysis_type: Type of analysis to perform
        
    Returns:
        Complete analysis with LLM feedback and recommendations
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
            # Perform technical analysis
            analysis_result = audio_processor.analyze_audio(file_path, analysis_enum)

            # Get LLM feedback
            llm_response = await llm_service.get_comprehensive_feedback(analysis_result, guidance)

            return {
                "feedback": llm_response.feedback,
                "technical_analysis": llm_response.technical_analysis,
                "recommendations": llm_response.recommendations,
                "analysis_type": analysis_type,
                "user_question": guidance,
                "file_path": file_path
            }

        finally:
            # Optional: cleanup file after processing
            # file_service.cleanup_file(file_path)
            pass

    except FileProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (AudioProcessingError, AnalysisError) as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis error: {str(e)}")
    except LLMServiceError as e:
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# TODO-Idea endpoints for future implementation:
# @router.post("/compare-performance")
# async def compare_with_previous():
#     """Compare current performance with previous recordings"""
#     pass

# @router.post("/generate-practice-plan") 
# async def generate_practice_plan():
#     """Generate personalized practice plan based on weaknesses"""
#     pass

# @router.get("/progress-report/{user_id}")
# async def get_progress_report():
#     """Get user's progress report over time"""
#     pass