from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class AnalysisType(str, Enum):
    FULL = "full"
    BREATH = "breath"
    TONE = "tone"
    RHYTHM = "rhythm"
    EXPRESSION = "expression"
    FLEXIBILITY = "flexibility"

class AudioAnalysisRequest(BaseModel):
    guidance: str = Field(..., description="User's question or guidance text")
    analysis_type: AnalysisType = Field(default=AnalysisType.FULL, description="Type of analysis to perform")

class BreathInterval(BaseModel):
    start_time: float
    end_time: float
    duration: float

class BreathAnalysisResult(BaseModel):
    breath_intervals: List[BreathInterval]
    average_breath_length: float
    breath_consistency: str
    recommendations: str
    breath_count: int

class ToneAnalysisResult(BaseModel):
    harmonic_ratio: float
    quality_score: str
    recommendations: str

class RhythmAnalysisResult(BaseModel):
    tempo: float
    consistency: str
    recommendations: str

class ExpressionAnalysisResult(BaseModel):
    dynamic_range: float
    expression_level: str
    recommendations: str

class FlexibilityAnalysisResult(BaseModel):
    transition_smoothness: float
    flexibility_level: str
    recommendations: str

class AudioAnalysisResult(BaseModel):
    breath_control: Optional[BreathAnalysisResult] = None
    tone_quality: Optional[ToneAnalysisResult] = None
    rhythm_timing: Optional[RhythmAnalysisResult] = None
    expression: Optional[ExpressionAnalysisResult] = None
    flexibility: Optional[FlexibilityAnalysisResult] = None

class LLMResponse(BaseModel):
    feedback: str
    technical_analysis: Dict[str, Any]
    recommendations: List[str]

class QuestionRequest(BaseModel):
    question: str = Field(..., description="User's question about trumpet technique")

class QuestionResponse(BaseModel):
    question: str
    answer: str
    context_used: bool = Field(default=False, description="Whether audio context was used")

class TrumpetDetectionResult(BaseModel):
    is_trumpet: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    detection_features: Dict[str, Any]
    warning_message: Optional[str] = None
    recommendations: List[str] = []