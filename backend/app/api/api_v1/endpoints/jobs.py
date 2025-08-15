"""
Job monitoring and management endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.job import JobResponse
from app.services.job_service import JobService

router = APIRouter()


@router.get("/", response_model=List[JobResponse])
async def list_jobs(
    workflow_id: Optional[int] = None,
    status: Optional[str] = None,
    platform: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List jobs with optional filtering"""
    service = JobService(db)
    jobs = await service.list_jobs(
        workflow_id=workflow_id,
        status=status,
        platform=platform,
        skip=skip,
        limit=limit
    )
    return jobs


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get job details"""
    service = JobService(db)
    job = await service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    return job


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Cancel a running job"""
    service = JobService(db)
    result = await service.cancel_job(job_id)
    return result


@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get job execution logs"""
    service = JobService(db)
    logs = await service.get_job_logs(job_id)
    return {"logs": logs}


@router.get("/{job_id}/results")
async def get_job_results(
    job_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get job results and output files"""
    service = JobService(db)
    results = await service.get_job_results(job_id)
    return results
