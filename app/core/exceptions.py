class TrumpetAnalyzerException(Exception):
    """Base exception for Trumpet Analyzer"""
    pass

class AudioProcessingError(TrumpetAnalyzerException):
    """Raised when audio processing fails"""
    pass

class LLMServiceError(TrumpetAnalyzerException):
    """Raised when LLM service fails"""
    pass

class FileProcessingError(TrumpetAnalyzerException):
    """Raised when file processing fails"""
    pass

class AnalysisError(TrumpetAnalyzerException):
    """Raised when analysis fails"""
    pass