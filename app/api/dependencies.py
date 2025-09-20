from functools import lru_cache
from app.services.audio_processor import AudioProcessorService
from app.services.llm_service import LLMService
from app.services.file_service import FileService
from app.config import Settings, settings

@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""
    return settings

def get_audio_processor() -> AudioProcessorService:
    """Dependency for AudioProcessorService"""
    return AudioProcessorService()

def get_llm_service() -> LLMService:
    """Dependency for LLMService"""
    return LLMService()

def get_file_service() -> FileService:
    """Dependency for FileService"""
    return FileService()

# Health check dependencies
def check_ollama_connection():
    """Check if Ollama service is available"""
    try:
        import ollama
        # Try to list models to check connection
        ollama.list()
        return True
    except Exception:
        return False

def check_upload_directory():
    """Check if upload directory is accessible"""
    import os
    return os.path.exists(settings.UPLOAD_DIR) and os.access(settings.UPLOAD_DIR, os.W_OK)