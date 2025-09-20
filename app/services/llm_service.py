import ollama
import json
from typing import List, Optional
from app.core.models import AudioAnalysisResult, LLMResponse, QuestionResponse
from app.core.exceptions import LLMServiceError
from app.config import settings

class LLMService:
    """Service for LLM integration and feedback generation"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.OLLAMA_MODEL

    async def get_comprehensive_feedback(
            self,
            analysis_result: AudioAnalysisResult,
            user_question: str
    ) -> LLMResponse:
        """
        Generate comprehensive feedback using LLM
        
        Args:
            analysis_result: Results from audio analysis
            user_question: User's question or guidance
            
        Returns:
            LLMResponse with feedback and recommendations
        """
        try:
            # Extract technical data
            from app.services.audio_processor import AudioProcessorService
            processor = AudioProcessorService()
            technical_data = processor.extract_technical_data(analysis_result)

            # Create structured prompt
            prompt = self._create_comprehensive_prompt(technical_data, user_question)

            # Get LLM response
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )

            feedback_text = response["response"]

            return LLMResponse(
                feedback=feedback_text,
                technical_analysis=technical_data,
                recommendations=self._extract_recommendations(feedback_text)
            )

        except Exception as e:
            raise LLMServiceError(f"Failed to generate LLM feedback: {str(e)}")

    async def answer_question(
            self,
            question: str,
            analysis_context: Optional[AudioAnalysisResult] = None
    ) -> QuestionResponse:
        """
        Answer trumpet technique questions with optional audio context
        
        Args:
            question: User's question
            analysis_context: Optional audio analysis for context
            
        Returns:
            QuestionResponse with answer
        """
        try:
            context_used = analysis_context is not None
            prompt = self._create_qa_prompt(question, analysis_context)

            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )

            return QuestionResponse(
                question=question,
                answer=response["response"],
                context_used=context_used
            )

        except Exception as e:
            raise LLMServiceError(f"Failed to answer question: {str(e)}")

    def _create_comprehensive_prompt(self, technical_data: dict, user_question: str) -> str:
        """Create structured prompt for comprehensive feedback"""
        return f"""You are an experienced trumpet teacher providing feedback to a student.

Technical Analysis Results:
{json.dumps(technical_data, indent=2)}

Student's Question:
{user_question}

Please provide:
1. Overall performance assessment
2. Specific technical feedback based on the analysis
3. Actionable practice recommendations
4. Answer to the student's specific question
5. Encouragement and next steps

Keep your response encouraging but honest, focusing on practical advice the student can implement immediately."""

    def _create_qa_prompt(self, question: str, analysis_context: Optional[AudioAnalysisResult]) -> str:
        """Create prompt for question answering"""
        context_section = ""
        if analysis_context:
            from app.services.audio_processor import AudioProcessorService
            processor = AudioProcessorService()
            technical_data = processor.extract_technical_data(analysis_context)
            context_section = f"\nBased on the student's recent performance analysis:\n{json.dumps(technical_data, indent=2)}"

        return f"""You are an expert trumpet teacher. A student has asked you a question about trumpet technique.

Question: {question}{context_section}

Provide a clear, practical answer that:
1. Directly addresses their question
2. Includes specific technique tips
3. Suggests practice exercises if relevant
4. Is encouraging and supportive

Keep your answer concise but comprehensive."""

    def _extract_recommendations(self, feedback_text: str) -> List[str]:
        """
        Extract actionable recommendations from feedback text
        
        Args:
            feedback_text: LLM-generated feedback
            
        Returns:
            List of extracted recommendations
        """
        lines = feedback_text.split('\n')
        recommendations = []

        for line in lines:
            line = line.strip()
            # Look for lines that contain action words
            if any(keyword in line.lower() for keyword in [
                'practice', 'try', 'work on', 'focus on', 'exercise',
                'improve', 'develop', 'strengthen', 'build'
            ]):
                # Ensure it's not just a header and has reasonable length
                if len(line) > 15 and not line.endswith(':') and '?' not in line:
                    # Clean up the line
                    cleaned_line = line.lstrip('•-*').strip()
                    if cleaned_line and len(cleaned_line) > 10:
                        recommendations.append(cleaned_line)

        # Limit to 5 most relevant recommendations
        return recommendations[:5]