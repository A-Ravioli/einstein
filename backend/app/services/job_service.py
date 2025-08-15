"""
Job service layer
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from app.models.job import Job, JobStep, JobStatus, PlatformType
from app.schemas.job import JobResponse, JobCreate, JobUpdate
from loguru import logger


class JobService:
    """Service for job management"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_jobs(
        self,
        workflow_id: Optional[int] = None,
        status: Optional[str] = None,
        platform: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[JobResponse]:
        """List jobs with optional filtering"""
        try:
            query = select(Job).options(selectinload(Job.job_steps)).offset(skip).limit(limit)
            
            filters = []
            if workflow_id:
                filters.append(Job.workflow_id == workflow_id)
            if status:
                filters.append(Job.status == status)
            if platform:
                filters.append(Job.platform == platform)
            
            if filters:
                query = query.where(and_(*filters))
            
            result = await self.db.execute(query)
            jobs = result.scalars().all()
            
            return [JobResponse.from_orm(job) for job in jobs]
        
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            raise
    
    async def get_job(self, job_id: int) -> Optional[JobResponse]:
        """Get a specific job by ID"""
        try:
            query = select(Job).options(selectinload(Job.job_steps)).where(Job.id == job_id)
            result = await self.db.execute(query)
            job = result.scalar_one_or_none()
            
            if job:
                return JobResponse.from_orm(job)
            return None
        
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            raise
    
    async def create_job(
        self,
        workflow_id: int,
        platform: str,
        parameters: Dict[str, Any] = None,
        platform_config: Dict[str, Any] = None
    ) -> Job:
        """Create a new job"""
        try:
            job = Job(
                workflow_id=workflow_id,
                platform=PlatformType(platform),
                parameters=parameters or {},
                platform_config=platform_config or {},
                status=JobStatus.PENDING
            )
            
            self.db.add(job)
            await self.db.commit()
            await self.db.refresh(job)
            
            logger.info(f"Created job: {job.id}")
            return job
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating job: {e}")
            raise
    
    async def update_job(
        self, 
        job_id: int, 
        job_data: JobUpdate
    ) -> Optional[JobResponse]:
        """Update an existing job"""
        try:
            query = select(Job).where(Job.id == job_id)
            result = await self.db.execute(query)
            job = result.scalar_one_or_none()
            
            if not job:
                return None
            
            # Update fields if provided
            if job_data.status is not None:
                job.status = job_data.status
            if job_data.external_job_id is not None:
                job.external_job_id = job_data.external_job_id
            if job_data.results is not None:
                job.results = job_data.results
            if job_data.logs is not None:
                job.logs = job_data.logs
            if job_data.error_message is not None:
                job.error_message = job_data.error_message
            if job_data.started_at is not None:
                job.started_at = job_data.started_at
            if job_data.completed_at is not None:
                job.completed_at = job_data.completed_at
            if job_data.duration_seconds is not None:
                job.duration_seconds = job_data.duration_seconds
            if job_data.cpu_hours is not None:
                job.cpu_hours = job_data.cpu_hours
            if job_data.memory_gb_hours is not None:
                job.memory_gb_hours = job_data.memory_gb_hours
            if job_data.cost_estimate is not None:
                job.cost_estimate = job_data.cost_estimate
            
            await self.db.commit()
            await self.db.refresh(job)
            
            logger.info(f"Updated job: {job.id}")
            return JobResponse.from_orm(job)
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating job {job_id}: {e}")
            raise
    
    async def cancel_job(self, job_id: int) -> Dict[str, Any]:
        """Cancel a running job"""
        try:
            job = await self.get_job(job_id)
            if not job:
                return {"success": False, "message": "Job not found"}
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return {"success": False, "message": f"Job is already {job.status.value}"}
            
            # Update job status to cancelled
            await self.update_job(job_id, JobUpdate(status=JobStatus.CANCELLED))
            
            # TODO: Cancel job on external platform if needed
            # This would involve calling the appropriate platform adapter
            
            logger.info(f"Cancelled job: {job_id}")
            return {"success": True, "message": "Job cancelled successfully"}
        
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return {"success": False, "message": f"Error cancelling job: {str(e)}"}
    
    async def get_job_logs(self, job_id: int) -> str:
        """Get job execution logs"""
        try:
            job = await self.get_job(job_id)
            if not job:
                return ""
            
            logs = job.logs or ""
            
            # Add step logs if available
            if job.job_steps:
                step_logs = []
                for step in job.job_steps:
                    if step.logs:
                        step_logs.append(f"=== Step: {step.step_name} ===\n{step.logs}")
                
                if step_logs:
                    logs += "\n\n" + "\n\n".join(step_logs)
            
            return logs
        
        except Exception as e:
            logger.error(f"Error getting logs for job {job_id}: {e}")
            return f"Error retrieving logs: {str(e)}"
    
    async def get_job_results(self, job_id: int) -> Dict[str, Any]:
        """Get job results and output files"""
        try:
            job = await self.get_job(job_id)
            if not job:
                return {"error": "Job not found"}
            
            results = {
                "job_id": job_id,
                "status": job.status,
                "results": job.results,
                "output_files": [],
                "step_results": []
            }
            
            # Add step results
            if job.job_steps:
                for step in job.job_steps:
                    step_result = {
                        "step_name": step.step_name,
                        "status": step.status,
                        "results": step.results,
                        "output_files": step.output_files
                    }
                    results["step_results"].append(step_result)
                    results["output_files"].extend(step.output_files)
            
            return results
        
        except Exception as e:
            logger.error(f"Error getting results for job {job_id}: {e}")
            return {"error": f"Error retrieving results: {str(e)}"}
    
    async def create_job_step(
        self,
        job_id: int,
        step_name: str,
        step_type: str,
        order_index: int,
        parameters: Dict[str, Any] = None
    ) -> JobStep:
        """Create a job step"""
        try:
            step = JobStep(
                job_id=job_id,
                step_name=step_name,
                step_type=step_type,
                order_index=order_index,
                parameters=parameters or {},
                status=JobStatus.PENDING
            )
            
            self.db.add(step)
            await self.db.commit()
            await self.db.refresh(step)
            
            logger.info(f"Created job step: {step.id} for job {job_id}")
            return step
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating job step: {e}")
            raise
