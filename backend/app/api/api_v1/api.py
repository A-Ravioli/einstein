"""
API v1 router configuration
"""

from fastapi import APIRouter

from app.api.api_v1.endpoints import workflows, platforms, jobs, files

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
api_router.include_router(platforms.router, prefix="/platforms", tags=["platforms"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
