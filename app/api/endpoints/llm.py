from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Dict, Any

from app.services.llm_service import LLMService
from app.services.audio_processor import AudioProcessorService
from app.services.file_service import FileService
from app.core.models import QuestionRequest, QuestionResponse, AnalysisType
from app.core.exceptions import LLMServiceError, AudioProcessingError, FileProcessingError

router = APIRouter(prefix="/llm", tags=["llm"])


def get_llm_service() -> LLMService:
    return LLMService()


def get_audio_processor() -> AudioProcessorService:
    return AudioProcessorService()


def get_file_service() -> FileService:
    return FileService()


@router.post("/ask-question")
async def ask_question(
        question: str = Form(..., description="Trumpet technique question"),
        llm_service: LLMService = Depends(get_llm_service)
) -> QuestionResponse:
    """
    Ask trumpet technique questions without audio context
    
    Args:
        question: User's question about trumpet technique
        
    Returns:
        Answer from trumpet teaching expert AI
    """
    try:
        response = await llm_service.answer_question(question)
        return response

    except LLMServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/ask-with-context")
async def ask_question_with_audio_context(
        question: str = Form(..., description="Trumpet technique question"),
        audioData: UploadFile = File(..., description="Audio file for context"),
        analysis_type: str = Form(default="full", description="Type of analysis for context"),
        llm_service: LLMService = Depends(get_llm_service),
        audio_processor: AudioProcessorService = Depends(get_audio_processor),
        file_service: FileService = Depends(get_file_service)
) -> Dict[str, Any]:
    """
    Ask questions with audio context for personalized answers
    
    Args:
        question: User's question about trumpet technique
        audioData: Audio file to provide context
        analysis_type: Type of analysis to perform for context
        
    Returns:
        Personalized answer based on audio analysis
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
            # Analyze audio for context
            analysis_result = audio_processor.analyze_audio(file_path, analysis_enum)

            # Get personalized answer
            response = await llm_service.answer_question(question, analysis_result)

            # Extract technical context used
            technical_context = audio_processor.extract_technical_data(analysis_result)

            return {
                "question": question,
                "answer": response.answer,
                "context_used": response.context_used,
                "technical_context": technical_context,
                "file_path": file_path
            }

        finally:
            # Optional: cleanup file after processing
            # file_service.cleanup_file(file_path)
            pass

    except FileProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AudioProcessingError as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")
    except LLMServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# TODO-LLM endpoints for future implementation:
# @router.post("/generate-practice-exercises")
# async def generate_practice_exercises():
#     """Generate custom practice exercises based on analysis"""
#     pass

# @router.post("/explain-technique")
# async def explain_technique():
#     """Provide detailed explanation of specific trumpet techniques"""
#     pass

# @router.post("/compare-with-professionals")
# async def compare_with_professionals():
#     """Compare user's playing with professional recordings"""
#     pass
