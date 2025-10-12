import os
from typing import Optional, List

class Settings:
    # File handling
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/recordings")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB

    # Cloudinary configuration
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "")
    CLOUDINARY_FOLDER: str = os.getenv("CLOUDINARY_FOLDER", "trumpet-analyzer")

    # Use Cloudinary if credentials are set, otherwise use local storage
    USE_CLOUDINARY: bool = bool(
        os.getenv("CLOUDINARY_CLOUD_NAME") and
        os.getenv("CLOUDINARY_API_KEY") and
        os.getenv("CLOUDINARY_API_SECRET")
    )

    # Audio processing
    AUDIO_SAMPLE_RATE: Optional[int] = None  # Let librosa decide
    TRUMPET_LOW_FREQ: float = 233.0
    TRUMPET_HIGH_FREQ: float = 2118.90

    # Breath analysis
    MIN_SILENCE_DURATION: float = 0.3
    SILENCE_THRESHOLD: float = 0.02

    # LLM configuration - Google AI Studio (Gemini)
    GOOGLE_AI_API_KEY: str = os.getenv("GOOGLE_AI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

    # Legacy Ollama settings (for backwards compatibility)
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "tinyllama:1.1b")

    # API configuration
    API_TITLE: str = "Trumpet Analyzer API"
    API_VERSION: str = "1.0.0"

    # CORS configuration - Support environment variable for production
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "https://provincial-avrit-at-it-8949d5c5.koyeb.app,http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"
    ).split(",") if isinstance(os.getenv("CORS_ORIGINS"), str) else [
        "https://provincial-avrit-at-it-8949d5c5.koyeb.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
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
        "If-Modified-Since",
    ]

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Database configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:admin@localhost:5432/trumpet_analyzer"
    )

    # Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # ML Model configuration
    ML_MODEL_PATH: str = "app/ml/models/trumpet_classifier.pkl"
    ML_FEATURE_CACHE_DIR: str = "data/ml_features"
    ML_TRAINING_DATA_DIR: str = "data/ml_training"
    ML_ENABLED: bool = False  # DISABLED temporarily until model is retrained
    ML_CONFIDENCE_WEIGHT: float = 0.4  # Not used when ML_ENABLED = False

    def __init__(self):
        # Ensure upload directory exists (for local storage fallback)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

settings = Settings()