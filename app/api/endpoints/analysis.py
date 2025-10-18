from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Dict, Any

from app.services.audio_processor import AudioProcessorService
from app.services.llm_service import LLMService
from app.services.file_service import FileService
from app.services.feedback_simplifier import FeedbackSimplifier  # NEW IMPORT
from app.core.models import AnalysisType, LLMResponse
from app.core.exceptions import AudioProcessingError, FileProcessingError, LLMServiceError, AnalysisError

router = APIRouter(prefix="/analysis", tags=["analysis"])


def get_audio_processor() -> AudioProcessorService:
    return AudioProcessorService()


def get_llm_service() -> LLMService:
    return LLMService()


def get_file_service() -> FileService:
    return FileService()


def get_feedback_simplifier() -> FeedbackSimplifier:  # NEW DEPENDENCY
    return FeedbackSimplifier()


@router.post("/comprehensive")
async def comprehensive_analysis(
        audioData: UploadFile = File(...),
        guidance: str = Form(..., description="User's question or guidance text"),
        analysis_type: str = Form(default="full"),
        file_service: FileService = Depends(get_file_service),
        audio_processor: AudioProcessorService = Depends(get_audio_processor),
        llm_service: LLMService = Depends(get_llm_service),
        feedback_simplifier: FeedbackSimplifier = Depends(get_feedback_simplifier)  # NEW DEPENDENCY
) -> Dict[str, Any]:
    """
    Comprehensive analysis with LLM feedback
    
    Main endpoint that combines technical analysis with AI-powered feedback
    
    Args:
        audioData: Audio file to analyze
        guidance: User's question or guidance text
        analysis_type: Type of analysis to perform
        
    Returns:
        Complete analysis with LLM feedback, recommendations, and simplified feedback
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

        # Save uploaded file (KEEP THE FILE for playback later)
        file_path = await file_service.save_uploaded_file(audioData)
        print("Analyzing the file", file_path)

        # Perform technical analysis with trumpet detection
        analysis_result, trumpet_detection = audio_processor.analyze_audio(file_path, analysis_enum)
        print("trumpet_detection.is_trumpet", trumpet_detection.is_trumpet)

        # Check if trumpet was detected
        if not trumpet_detection.is_trumpet:
            # Clean up file if not a trumpet
            file_service.cleanup_file(file_path)
            return {
                "error": "No trumpet detected",
                "detection_result": trumpet_detection.dict(),
                "message": trumpet_detection.warning_message or "Please ensure you're playing a trumpet and try again.",
                "recommendations": trumpet_detection.recommendations,
                "file_path": None  # No file saved for non-trumpet audio
            }

        # Proceed with LLM feedback only if trumpet detected
        llm_response = await llm_service.get_comprehensive_feedback(analysis_result, guidance)

        # ===== NEW: GENERATE SIMPLIFIED FEEDBACK =====
        # Extract technical data for simplification
        technical_data = audio_processor.extract_technical_data(analysis_result)

        # Determine technique focus from guidance text
        technique_focus = determine_technique_focus(guidance, analysis_type)

        # Generate simplified feedback
        simplified_feedback = feedback_simplifier.simplify_analysis(
            technical_data,
            technique_focus
        )
        # ===== END NEW SECTION =====

        return {
            "feedback": llm_response.feedback,
            "technical_analysis": llm_response.technical_analysis,
            "recommendations": llm_response.recommendations,
            "detection_result": trumpet_detection.dict(),
            "analysis_type": analysis_type,
            "user_question": guidance,
            "file_path": file_path,  # Return file path for database storage
            # NEW: Add simplified feedback to response
            "simplified_feedback": {
                "overall_status": simplified_feedback.overall_status,
                "main_issue": simplified_feedback.main_issue,
                "quick_tip": simplified_feedback.quick_tip,
                "next_step": simplified_feedback.next_step
            }
        }

    except FileProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (AudioProcessingError, AnalysisError) as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis error: {str(e)}")
    except LLMServiceError as e:
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# NEW HELPER FUNCTION
def determine_technique_focus(guidance: str, analysis_type: str) -> str:
    """
    Determine which technique the user is focusing on based on guidance text
    
    Args:
        guidance: User's input text
        analysis_type: Type of analysis being performed
        
    Returns:
        Technique focus: breathing, tone, rhythm, articulation, or flexibility
    """
    guidance_lower = guidance.lower()

    # Check guidance text for keywords
    if any(word in guidance_lower for word in ["breath", "breathing", "air", "support"]):
        return "breathing"
    elif any(word in guidance_lower for word in ["tone", "sound", "quality", "timbre"]):
        return "tone"
    elif any(word in guidance_lower for word in ["rhythm", "timing", "tempo", "beat"]):
        return "rhythm"
    elif any(word in guidance_lower for word in ["articulation", "tongue", "attack", "staccato", "legato"]):
        return "articulation"
    elif any(word in guidance_lower for word in ["flexibility", "slur", "interval", "range"]):
        return "flexibility"

    # Fallback to analysis_type if no keywords found
    analysis_type_map = {
        "breath": "breathing",
        "tone": "tone",
        "rhythm": "rhythm",
        "expression": "tone",
        "flexibility": "flexibility"
    }

    return analysis_type_map.get(analysis_type, "breathing")  # Default to breathing


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