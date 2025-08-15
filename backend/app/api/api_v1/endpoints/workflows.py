"""
Workflow management endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.workflow import WorkflowCreate, WorkflowUpdate, WorkflowResponse
from app.services.workflow_service import WorkflowService

router = APIRouter()


@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all workflows"""
    service = WorkflowService(db)
    workflows = await service.list_workflows(skip=skip, limit=limit)
    return workflows


@router.post("/", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    workflow_data: WorkflowCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new workflow"""
    service = WorkflowService(db)
    workflow = await service.create_workflow(workflow_data)
    return workflow


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific workflow"""
    service = WorkflowService(db)
    workflow = await service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    return workflow


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: int,
    workflow_data: WorkflowUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a workflow"""
    service = WorkflowService(db)
    workflow = await service.update_workflow(workflow_id, workflow_data)
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    return workflow


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a workflow"""
    service = WorkflowService(db)
    success = await service.delete_workflow(workflow_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )


@router.post("/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: int,
    platform: str,
    db: AsyncSession = Depends(get_db)
):
    """Execute a workflow on a specific platform"""
    service = WorkflowService(db)
    job = await service.execute_workflow(workflow_id, platform)
    return {"job_id": job.id, "status": job.status, "platform": platform}


@router.post("/{workflow_id}/validate")
async def validate_workflow(
    workflow_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Validate a workflow definition"""
    service = WorkflowService(db)
    validation_result = await service.validate_workflow(workflow_id)
    return validation_result
