"""
Platform integration endpoints
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.platform_service import PlatformService

router = APIRouter()


@router.get("/")
async def list_platforms():
    """List all available platforms"""
    service = PlatformService()
    platforms = await service.list_available_platforms()
    return platforms


@router.get("/{platform_name}/status")
async def check_platform_status(platform_name: str):
    """Check if a platform is available and authenticated"""
    service = PlatformService()
    status = await service.check_platform_status(platform_name)
    return status


@router.post("/{platform_name}/authenticate")
async def authenticate_platform(
    platform_name: str,
    credentials: Dict[str, Any]
):
    """Authenticate with a platform"""
    service = PlatformService()
    result = await service.authenticate_platform(platform_name, credentials)
    return result


@router.get("/{platform_name}/tools")
async def list_platform_tools(platform_name: str):
    """List available tools on a platform"""
    service = PlatformService()
    tools = await service.list_platform_tools(platform_name)
    return tools
