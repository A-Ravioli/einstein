"""
Workflow execution endpoints
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.workflow_execution_service import WorkflowExecutionService

router = APIRouter()


@router.post("/execute")
async def execute_workflow(
    workflow_id: int,
    platform: str,
    parameters: Optional[Dict[str, Any]] = None,
    priority: Optional[int] = 0,
    db: AsyncSession = Depends(get_db)
):
    """Execute a workflow on the specified platform"""
    service = WorkflowExecutionService(db)
    
    try:
        result = await service.submit_workflow(
            workflow_id=workflow_id,
            platform=platform,
            parameters=parameters,
            priority=priority
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}"
        )


@router.get("/status/{task_id}")
async def get_execution_status(
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get execution status for a task"""
    service = WorkflowExecutionService(db)
    
    try:
        status_info = await service.get_execution_status(task_id)
        return status_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


@router.post("/cancel/{task_id}")
async def cancel_execution(
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Cancel a workflow execution"""
    service = WorkflowExecutionService(db)
    
    try:
        result = await service.cancel_execution(task_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cancellation failed: {str(e)}"
        )


@router.post("/retry")
async def retry_workflow(
    workflow_id: int,
    failed_job_id: int,
    platform: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Retry a failed workflow execution"""
    service = WorkflowExecutionService(db)
    
    try:
        result = await service.retry_workflow(
            workflow_id=workflow_id,
            failed_job_id=failed_job_id,
            platform=platform
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retry failed: {str(e)}"
        )


@router.post("/estimate-cost")
async def estimate_workflow_cost(
    workflow_id: int,
    platform: str,
    parameters: Optional[Dict[str, Any]] = None,
    db: AsyncSession = Depends(get_db)
):
    """Estimate the cost of running a workflow"""
    service = WorkflowExecutionService(db)
    
    try:
        estimate = await service.estimate_cost(
            workflow_id=workflow_id,
            platform=platform,
            parameters=parameters
        )
        return estimate
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cost estimation failed: {str(e)}"
        )
