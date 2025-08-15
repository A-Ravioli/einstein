from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import structlog
from dotenv import load_dotenv
import os

from app.core.config import get_settings
from app.api.v1.api import api_router
from app.core.database import init_db
from app.services.ai_services import AIServicesManager

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AI Co-Scientist Platform...")
    
    # Initialize database
    await init_db()
    
    # Initialize AI services
    ai_manager = AIServicesManager()
    await ai_manager.initialize()
    app.state.ai_manager = ai_manager
    
    logger.info("AI Co-Scientist Platform started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Co-Scientist Platform...")
    if hasattr(app.state, 'ai_manager'):
        await app.state.ai_manager.cleanup()
    logger.info("AI Co-Scientist Platform shutdown complete")

# Create FastAPI application
settings = get_settings()

app = FastAPI(
    title="AI Co-Scientist Platform",
    description="Advanced AI-powered scientific research platform with multi-agent system for literature review, hypothesis generation, and experiment design",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Co-Scientist Platform API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "database": "connected",
            "ai_services": "active",
            "api": "running"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 