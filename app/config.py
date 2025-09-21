import os
from typing import Optional

class Settings:
    # File handling
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/recordings")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB

    # Audio processing
    AUDIO_SAMPLE_RATE: Optional[int] = None  # Let librosa decide
    TRUMPET_LOW_FREQ: float = 233.0
    TRUMPET_HIGH_FREQ: float = 2118.90

    # Breath analysis
    MIN_SILENCE_DURATION: float = 0.3
    SILENCE_THRESHOLD: float = 0.02

    # LLM configuration
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "tinyllama:1.1b")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

    # API configuration
    API_TITLE: str = "Trumpet Analyzer API"
    API_VERSION: str = "1.0.0"

    # CORS configuration
    CORS_ORIGINS: list = [
        "http://localhost:3000",      # React development server
        "https://localhost:3000",      # React development server
        "http://127.0.0.1:3000",     # Alternative localhost
        "http://localhost:3001",      # Alternative React port
        "http://127.0.0.1:3001",     # Alternative localhost
        "http://localhost:5173",      # Vite development server
        "http://127.0.0.1:5173",     # Alternative Vite
        "http://localhost:8080",      # Alternative development port
        "http://127.0.0.1:8080",     # Alternative localhost
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list = [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "User-Agent",
        "DNT",
        "Cache-Control",
        "X-Mx-ReqToken",
        "Keep-Alive",
        "X-Requested-With",
        "If-Modified-Since",
    ]

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def __init__(self):
        # Ensure upload directory exists
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

settings = Settings()