from sqlalchemy.orm import Session
from typing import Optional
from app.database import crud
from app.services.feedback_simplifier import FeedbackSimplifier
from app.services.audio_processor import AudioProcessorService
from app.core.models import AudioAnalysisResult, SimpleFeedbackResult
from app.database import crud as db_crud


class PracticeService:
    """Service for managing practice sessions"""

    def __init__(self):
        self.feedback_simplifier = FeedbackSimplifier()
        self.audio_processor = AudioProcessorService()

    def start_practice_session(self, db: Session, user_id: int, exercise_id: int):
        """Start a new practice session"""
        return crud.create_practice_session(db, user_id, exercise_id)

    def complete_practice_session(
            self,
            db: Session,
            session_id: int,
            user_id: int,
            duration_seconds: int = None,
            recording_id: int = None,
            analysis_result: Optional[AudioAnalysisResult] = None,
            exercise_technique: str = "breathing"
    ):
        """
        Complete a practice session with optional analysis
        
        Args:
            db: Database session
            session_id: Practice session ID
            user_id: User ID
            duration_seconds: How long they practiced
            recording_id: If they recorded, the recording ID
            analysis_result: Analysis of their performance
            exercise_technique: What technique the exercise focuses on
        """

        simplified_feedback = None

        # If we have analysis results, simplify them
        if analysis_result:
            technical_data = self.audio_processor.extract_technical_data(analysis_result)
            simple_feedback = self.feedback_simplifier.simplify_analysis(
                technical_data,
                exercise_technique
            )
            # Convert to string for storage
            simplified_feedback = f"{simple_feedback.overall_status}|{simple_feedback.main_issue or 'None'}|{simple_feedback.quick_tip}|{simple_feedback.next_step}"

        return crud.complete_practice_session(
            db,
            session_id,
            user_id,
            duration_seconds,
            recording_id,
            simplified_feedback
        )

    def get_recommended_exercises(
            self,
            db: Session,
            user_skill_level: str,
            technique: Optional[str] = None
    ):
        """Get recommended exercises for a user based on their skill level"""

        # Map skill levels to difficulty
        difficulty_map = {
            "beginner": "beginner",
            "intermediate": "intermediate",
            "advanced": "advanced",
            "professional": "advanced"  # Use advanced for professionals
        }

        difficulty = difficulty_map.get(user_skill_level, "intermediate")

        return crud.get_exercises(db, technique=technique, difficulty=difficulty)

    def parse_simplified_feedback(self, feedback_string: str) -> dict:
        """Parse stored simplified feedback back into dict"""
        if not feedback_string:
            return None

        parts = feedback_string.split('|')
        if len(parts) != 4:
            return None

        return {
            "overall_status": parts[0],
            "main_issue": parts[1] if parts[1] != "None" else None,
            "quick_tip": parts[2],
            "next_step": parts[3]
        }

    def start_practice_session_from_calendar(
            self,
            db: Session,
            user_id: int,
            calendar_entry_id: int
    ) -> tuple:
        """Start practice session from a calendar entry"""
    
        # Get calendar entry
        calendar_entry = db_crud.get_calendar_entry_by_id(db, calendar_entry_id, user_id)
        if not calendar_entry:
            return None, None
    
        # Create practice session
        session = crud.create_practice_session(db, user_id, calendar_entry.exercise_id)
    
        return session, calendar_entry


    def complete_practice_session_with_calendar(
            self,
            db: Session,
            session_id: int,
            user_id: int,
            calendar_entry_id: int = None,
            duration_seconds: int = None,
            recording_id: int = None,
            analysis_result: Optional[AudioAnalysisResult] = None,
            exercise_technique: str = "breathing"
    ):
        """Complete practice session and update calendar entry"""
        from app.database import crud as db_crud
    
        # Complete practice session
        completed_session = self.complete_practice_session(
            db,
            session_id,
            user_id,
            duration_seconds,
            recording_id,
            analysis_result,
            exercise_technique
        )
    
        # If linked to calendar, mark calendar entry as complete
        if calendar_entry_id and completed_session:
            db_crud.mark_calendar_entry_complete(
                db,
                calendar_entry_id,
                user_id,
                completed_session.id
            )
    
        return completed_session