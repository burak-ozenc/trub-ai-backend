from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os

from app.config import settings
from app.api.endpoints import audio, analysis, llm, auth, users, recordings
from app.api.dependencies import check_upload_directory

# Import database components
from app.database.connection import engine, Base
from app.database import models  # This imports all models

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="AI-powered trumpet performance analysis and coaching system"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Startup event to create tables
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")

# Include API routers
app.include_router(audio.router)
app.include_router(analysis.router)
app.include_router(llm.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(recordings.router)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": f"Welcome to {settings.API_TITLE}",
        "version": settings.API_VERSION,
        "description": "AI-powered trumpet performance analysis and coaching",
        "endpoints": {
            "audio_analysis": "/audio/",
            "comprehensive_analysis": "/analysis/",
            "llm_services": "/llm/",
            "health_check": "/health",
            "documentation": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "checks": {}
    }

    # Check upload directory
    health_status["checks"]["upload_directory"] = {
        "status": "healthy" if check_upload_directory() else "unhealthy",
        "path": settings.UPLOAD_DIR
    }

    # Check database connection
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_status["checks"]["database"] = {
            "status": "healthy"
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }

    # Determine overall status
    all_healthy = all(
        check["status"] == "healthy"
        for check in health_status["checks"].values()
    )

    if not all_healthy:
        health_status["status"] = "degraded"
        return JSONResponse(status_code=503, content=health_status)

    return health_status

@app.get("/config")
async def get_config():
    """Get non-sensitive configuration information"""
    return {
        "api_title": settings.API_TITLE,
        "api_version": settings.API_VERSION,
        "ollama_model": settings.OLLAMA_MODEL,
        "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
        "supported_analysis_types": ["full", "breath", "tone", "rhythm", "expression", "flexibility"],
        "features": {
            "breath_analysis": True,
            "tone_analysis": True,
            "llm_feedback": True,
            "question_answering": True,
            "rhythm_analysis": True,
            "expression_analysis": True,
            "flexibility_analysis": True,
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    return JSONResponse(
        status_code=500,
        content={
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.LOG_LEVEL == "DEBUG" else "Internal server error"
        }
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_level=settings.LOG_LEVEL.lower()
    )