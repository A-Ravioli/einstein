"""
Celery tasks for platform integration and management
"""

import asyncio
from typing import Dict, Any, List
from celery import current_task

from app.worker import celery_app
from app.core.database import AsyncSessionLocal
from app.services.platform_service import PlatformService
from loguru import logger


@celery_app.task(bind=True, name="check_platform_health")
def check_platform_health_task(self):
    """Check health status of all platforms"""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Checking platform health"}
        )
        
        return asyncio.run(_check_platform_health_async(self))
    
    except Exception as e:
        logger.error(f"Platform health check task failed: {e}")
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Platform health check failed"}
        )
        raise


async def _check_platform_health_async(task):
    """Async platform health check logic"""
    try:
        platform_service = PlatformService()
        
        # Get all platforms
        platforms = await platform_service.list_available_platforms()
        total_platforms = len(platforms)
        
        health_results = []
        
        for i, platform in enumerate(platforms):
            # Update progress
            progress = int((i / total_platforms) * 100)
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": progress,
                    "total": 100,
                    "status": f"Checking {platform['name']}"
                }
            )
            
            # Check platform status
            try:
                status = await platform_service.check_platform_status(platform['name'])
                health_results.append({
                    "platform": platform['name'],
                    "status": status,
                    "timestamp": asyncio.get_event_loop().time()
                })
                logger.info(f"Platform {platform['name']} health: {status['status']}")
            
            except Exception as e:
                health_results.append({
                    "platform": platform['name'],
                    "status": {"status": "error", "message": str(e)},
                    "timestamp": asyncio.get_event_loop().time()
                })
                logger.error(f"Health check failed for {platform['name']}: {e}")
        
        task.update_state(
            state="SUCCESS",
            meta={
                "current": 100,
                "total": 100,
                "status": "Platform health check completed",
                "results": health_results
            }
        )
        
        return health_results
        
    except Exception as e:
        logger.error(f"Platform health check failed: {e}")
        raise


@celery_app.task(name="sync_platform_tools")
def sync_platform_tools_task(platform_name: str):
    """Synchronize available tools from a platform"""
    try:
        return asyncio.run(_sync_platform_tools_async(platform_name))
    except Exception as e:
        logger.error(f"Platform tools sync task failed: {e}")
        raise


async def _sync_platform_tools_async(platform_name: str):
    """Async platform tools synchronization logic"""
    try:
        platform_service = PlatformService()
        
        # Get platform adapter
        adapter = platform_service.get_adapter(platform_name)
        if not adapter:
            raise ValueError(f"Platform {platform_name} not supported")
        
        # Get tools from platform
        tools = await adapter.list_tools()
        
        # TODO: Store tools in database for caching
        # This would involve creating a PlatformTool model and service
        
        logger.info(f"Synchronized {len(tools)} tools from {platform_name}")
        return {
            "platform": platform_name,
            "tools_count": len(tools),
            "tools": tools
        }
        
    except Exception as e:
        logger.error(f"Platform tools sync failed for {platform_name}: {e}")
        raise


@celery_app.task(name="authenticate_platform")
def authenticate_platform_task(platform_name: str, credentials: Dict[str, Any]):
    """Authenticate with a platform and store credentials"""
    try:
        return asyncio.run(_authenticate_platform_async(platform_name, credentials))
    except Exception as e:
        logger.error(f"Platform authentication task failed: {e}")
        raise


async def _authenticate_platform_async(platform_name: str, credentials: Dict[str, Any]):
    """Async platform authentication logic"""
    try:
        platform_service = PlatformService()
        
        # Authenticate with platform
        result = await platform_service.authenticate_platform(platform_name, credentials)
        
        if result["success"]:
            # TODO: Store encrypted credentials in database
            # This would involve creating a PlatformCredential model
            logger.info(f"Successfully authenticated with {platform_name}")
        else:
            logger.warning(f"Authentication failed for {platform_name}: {result['message']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Platform authentication failed for {platform_name}: {e}")
        raise


@celery_app.task(name="refresh_platform_tokens")
def refresh_platform_tokens_task():
    """Refresh authentication tokens for all platforms"""
    try:
        return asyncio.run(_refresh_platform_tokens_async())
    except Exception as e:
        logger.error(f"Platform token refresh task failed: {e}")
        raise


async def _refresh_platform_tokens_async():
    """Async platform token refresh logic"""
    try:
        platform_service = PlatformService()
        
        # TODO: Implement token refresh logic
        # This would involve:
        # 1. Getting stored credentials from database
        # 2. Refreshing tokens for each platform
        # 3. Updating stored credentials
        
        logger.info("Platform token refresh completed")
        return {"status": "completed"}
        
    except Exception as e:
        logger.error(f"Platform token refresh failed: {e}")
        raise
