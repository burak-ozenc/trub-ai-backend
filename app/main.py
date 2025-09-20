from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.config import settings
from app.api.endpoints import audio, analysis, llm
from app.api.dependencies import check_ollama_connection, check_upload_directory

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="AI-powered trumpet performance analysis and coaching system"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(audio.router)
app.include_router(analysis.router)
app.include_router(llm.router)

# TODO-Idea List (embedded in main for reference)
"""
✓ Breath Control Analysis - IMPLEMENTED
- Embouchure Stability Analysis
- Range Assessment 
- Articulation Analysis
- Scale/Exercise Recognition
- Progress Tracking
- Practice Session Analytics
"""

# TODO-LLM List (embedded in main for reference)
"""
✓ Conversational Feedback - IMPLEMENTED
✓ Question Answering - IMPLEMENTED
- Personalized Practice Plans
- Performance Comparison
"""

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
    """
    Health check endpoint
    
    Checks:
    - Basic API functionality
    - Ollama LLM service connection
    - Upload directory accessibility
    """
    health_status = {
        "status": "healthy",
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "checks": {}
    }

    # Check Ollama connection
    health_status["checks"]["ollama"] = {
        "status": "healthy" if check_ollama_connection() else "unhealthy",
        "model": settings.OLLAMA_MODEL
    }

    # Check upload directory
    health_status["checks"]["upload_directory"] = {
        "status": "healthy" if check_upload_directory() else "unhealthy",
        "path": settings.UPLOAD_DIR
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
    """
    Get non-sensitive configuration information
    """
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
            "rhythm_analysis": False,  # TODO: implement
            "expression_analysis": False,  # TODO: implement
            "flexibility_analysis": False,  # TODO: implement
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
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )