"""
Celery tasks for workflow execution and monitoring
"""

import asyncio
from datetime import datetime
from typing import Dict, Any
from celery import current_task

from app.worker import celery_app
from app.core.database import AsyncSessionLocal
from app.services.workflow_service import WorkflowService
from app.services.job_service import JobService
from app.services.platform_service import PlatformService
from app.models.job import JobStatus
from app.schemas.job import JobUpdate
from app.websockets.monitoring_websocket import broadcast_job_update, broadcast_task_update
from loguru import logger


@celery_app.task(bind=True, name="execute_workflow")
def execute_workflow_task(self, workflow_id: int, platform: str, parameters: Dict[str, Any] = None):
    """Execute a workflow on the specified platform"""
    try:
        # Update task state
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting workflow execution"}
        )
        
        # Run async function in event loop
        return asyncio.run(_execute_workflow_async(
            self, workflow_id, platform, parameters or {}
        ))
    
    except Exception as e:
        logger.error(f"Workflow execution task failed: {e}")
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "status": "Workflow execution failed"}
        )
        raise


async def _execute_workflow_async(task, workflow_id: int, platform: str, parameters: Dict[str, Any]):
    """Async workflow execution logic"""
    async with AsyncSessionLocal() as db:
        try:
            workflow_service = WorkflowService(db)
            job_service = JobService(db)
            platform_service = PlatformService()
            
            # Update progress
            task.update_state(
                state="PROGRESS",
                meta={"current": 10, "total": 100, "status": "Validating workflow"}
            )
            
            # Get and validate workflow
            workflow = await workflow_service.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            validation = await workflow_service.validate_workflow(workflow_id)
            if not validation.is_valid:
                raise ValueError(f"Invalid workflow: {', '.join(validation.errors)}")
            
            # Update progress
            task.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Creating job record"}
            )
            
            # Create job record
            job = await job_service.create_job(
                workflow_id=workflow_id,
                platform=platform,
                parameters=parameters
            )
            
            # Update progress
            task.update_state(
                state="PROGRESS",
                meta={"current": 30, "total": 100, "status": f"Submitting to {platform}"}
            )
            
            # Get platform adapter
            adapter = platform_service.get_adapter(platform)
            if not adapter:
                raise ValueError(f"Platform {platform} not supported")
            
            # Submit workflow to platform
            external_job_id = await adapter.submit_workflow(workflow.definition)
            
            # Update job with external ID
            await job_service.update_job(job.id, JobUpdate(
                external_job_id=external_job_id,
                status=JobStatus.QUEUED,
                started_at=datetime.utcnow()
            ))
            
            # Broadcast job update
            await broadcast_job_update(
                job_id=job.id,
                status="queued",
                workflow_id=workflow_id,
                message=f"Job submitted to {platform}"
            )
            
            # Update progress
            task.update_state(
                state="PROGRESS",
                meta={"current": 50, "total": 100, "status": "Monitoring execution"}
            )
            
            # Start monitoring (this will be handled by another task)
            monitor_job_task.delay(job.id)
            
            # Final progress update
            task.update_state(
                state="SUCCESS",
                meta={
                    "current": 100, 
                    "total": 100, 
                    "status": "Workflow submitted successfully",
                    "job_id": job.id,
                    "external_job_id": external_job_id
                }
            )
            
            return {
                "job_id": job.id,
                "external_job_id": external_job_id,
                "status": "submitted"
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            # Update job status if job was created
            if 'job' in locals():
                await job_service.update_job(job.id, JobUpdate(
                    status=JobStatus.FAILED,
                    error_message=str(e),
                    completed_at=datetime.utcnow()
                ))
            raise


@celery_app.task(bind=True, name="monitor_job")
def monitor_job_task(self, job_id: int):
    """Monitor job execution status"""
    try:
        return asyncio.run(_monitor_job_async(self, job_id))
    except Exception as e:
        logger.error(f"Job monitoring task failed: {e}")
        raise


async def _monitor_job_async(task, job_id: int):
    """Async job monitoring logic"""
    async with AsyncSessionLocal() as db:
        try:
            job_service = JobService(db)
            platform_service = PlatformService()
            
            # Get job details
            job = await job_service.get_job(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            # Get platform adapter
            adapter = platform_service.get_adapter(job.platform)
            if not adapter:
                raise ValueError(f"Platform {job.platform} not supported")
            
            # Monitor until completion
            poll_count = 0
            max_polls = 720  # 6 hours with 30-second intervals
            
            while job.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING] and poll_count < max_polls:
                try:
                    # Get status from platform
                    platform_status = await adapter.get_job_status(job.external_job_id)
                    
                    # Map platform status to internal status
                    new_status = _map_platform_status(platform_status, job.platform)
                    
                    # Update job if status changed
                    if new_status != job.status:
                        update_data = JobUpdate(status=new_status)
                        
                        # Set completion time if job finished
                        if new_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                            update_data.completed_at = datetime.utcnow()
                            
                            # Get results if completed successfully
                            if new_status == JobStatus.COMPLETED:
                                try:
                                    results = platform_status.get('results', {})
                                    update_data.results = results
                                except Exception as e:
                                    logger.warning(f"Could not retrieve results for job {job_id}: {e}")
                        
                        await job_service.update_job(job_id, update_data)
                        
                        # Broadcast job update
                        await broadcast_job_update(
                            job_id=job_id,
                            status=new_status.value,
                            workflow_id=job.workflow_id,
                            message=f"Job status updated to {new_status.value}"
                        )
                        
                        logger.info(f"Job {job_id} status updated to {new_status}")
                    
                    # Update task progress
                    task.update_state(
                        state="PROGRESS",
                        meta={
                            "status": f"Job status: {new_status}",
                            "platform_status": platform_status,
                            "poll_count": poll_count
                        }
                    )
                    
                    # Break if job is finished
                    if new_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                        break
                    
                    # Wait before next poll
                    await asyncio.sleep(30)
                    poll_count += 1
                    
                    # Refresh job from database
                    job = await job_service.get_job(job_id)
                    
                except Exception as e:
                    logger.error(f"Error polling job {job_id}: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    poll_count += 1
            
            # Check if we exceeded max polls (timeout)
            if poll_count >= max_polls:
                await job_service.update_job(job_id, JobUpdate(
                    status=JobStatus.FAILED,
                    error_message="Job monitoring timeout exceeded",
                    completed_at=datetime.utcnow()
                ))
                logger.error(f"Job {job_id} monitoring timeout exceeded")
            
            return {"job_id": job_id, "final_status": job.status}
            
        except Exception as e:
            logger.error(f"Job monitoring failed: {e}")
            # Update job status to failed
            await job_service.update_job(job_id, JobUpdate(
                status=JobStatus.FAILED,
                error_message=f"Monitoring error: {str(e)}",
                completed_at=datetime.utcnow()
            ))
            raise


def _map_platform_status(platform_status: Dict[str, Any], platform: str) -> JobStatus:
    """Map platform-specific status to internal status"""
    if platform == "galaxy":
        galaxy_status = platform_status.get("state", "unknown")
        mapping = {
            "new": JobStatus.PENDING,
            "queued": JobStatus.QUEUED,
            "running": JobStatus.RUNNING,
            "ok": JobStatus.COMPLETED,
            "error": JobStatus.FAILED,
            "deleted": JobStatus.CANCELLED
        }
        return mapping.get(galaxy_status, JobStatus.PENDING)
    
    elif platform == "nextflow":
        nextflow_status = platform_status.get("status", "unknown")
        mapping = {
            "submitted": JobStatus.PENDING,
            "running": JobStatus.RUNNING,
            "succeeded": JobStatus.COMPLETED,
            "failed": JobStatus.FAILED,
            "cancelled": JobStatus.CANCELLED
        }
        return mapping.get(nextflow_status, JobStatus.PENDING)
    
    # Default mapping for unknown platforms
    return JobStatus.PENDING


@celery_app.task(name="cleanup_expired_jobs")
def cleanup_expired_jobs():
    """Clean up expired jobs and temporary files"""
    try:
        return asyncio.run(_cleanup_expired_jobs_async())
    except Exception as e:
        logger.error(f"Job cleanup task failed: {e}")
        raise


async def _cleanup_expired_jobs_async():
    """Async job cleanup logic"""
    async with AsyncSessionLocal() as db:
        try:
            job_service = JobService(db)
            
            # TODO: Implement cleanup logic
            # - Remove old completed jobs
            # - Clean up temporary files
            # - Cancel stuck jobs
            
            logger.info("Job cleanup completed")
            return {"status": "completed"}
            
        except Exception as e:
            logger.error(f"Job cleanup failed: {e}")
            raise
