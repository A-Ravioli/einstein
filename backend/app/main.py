"""
Einstein Scientific Workflow Builder - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.api_v1.api import api_router
from app.core.database import engine, Base, AsyncSessionLocal
from app.core.logging import setup_logging
from app.core.error_handlers import setup_error_handlers

# Setup logging
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    try:
        # Create database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize default templates
        from app.services.template_service import initialize_templates
        async with AsyncSessionLocal() as db:
            await initialize_templates(db)
            
    except Exception as e:
        print(f"Database initialization error: {e}")
    
    yield
    
    # Shutdown
    pass

# Create FastAPI application
app = FastAPI(
    title="Einstein Scientific Workflow Builder",
    description="Multi-cloud scientific workflow management platform",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup error handlers
setup_error_handlers(app)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Einstein Scientific Workflow Builder API",
        "version": "0.1.0",
        "docs": f"{settings.API_V1_STR}/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
