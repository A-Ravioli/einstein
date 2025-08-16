"""
Workflow execution service - orchestrates workflow execution across platforms
"""

from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.workflow_service import WorkflowService
from app.services.job_service import JobService
from app.services.platform_service import PlatformService
from app.models.job import JobStatus
from app.schemas.job import JobUpdate
from app.tasks.workflow_tasks import execute_workflow_task, monitor_job_task
from loguru import logger


class WorkflowExecutionService:
    """Service for orchestrating workflow execution"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.workflow_service = WorkflowService(db)
        self.job_service = JobService(db)
        self.platform_service = PlatformService()
    
    async def submit_workflow(
        self, 
        workflow_id: int, 
        platform: str, 
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> Dict[str, Any]:
        """Submit a workflow for execution"""
        try:
            # Validate workflow exists and is valid
            workflow = await self.workflow_service.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            validation = await self.workflow_service.validate_workflow(workflow_id)
            if not validation.is_valid:
                raise ValueError(f"Invalid workflow: {', '.join(validation.errors)}")
            
            # Check platform availability
            platform_status = await self.platform_service.check_platform_status(platform)
            if platform_status.get("status") != "available":
                raise ValueError(f"Platform {platform} is not available")
            
            # Submit execution task with priority
            task_options = {"priority": priority} if priority > 0 else {}
            task = execute_workflow_task.apply_async(
                args=[workflow_id, platform, parameters or {}],
                **task_options
            )
            
            logger.info(f"Submitted workflow {workflow_id} for execution (task: {task.id})")
            
            return {
                "task_id": task.id,
                "workflow_id": workflow_id,
                "platform": platform,
                "status": "submitted",
                "estimated_runtime": validation.estimated_runtime_minutes
            }
        
        except Exception as e:
            logger.error(f"Error submitting workflow {workflow_id}: {e}")
            raise
    
    async def get_execution_status(self, task_id: str) -> Dict[str, Any]:
        """Get execution status for a task"""
        try:
            from celery.result import AsyncResult
            
            task_result = AsyncResult(task_id)
            
            return {
                "task_id": task_id,
                "state": task_result.state,
                "info": task_result.info or {},
                "ready": task_result.ready(),
                "successful": task_result.successful() if task_result.ready() else None,
                "failed": task_result.failed() if task_result.ready() else None
            }
        
        except Exception as e:
            logger.error(f"Error getting execution status for task {task_id}: {e}")
            return {
                "task_id": task_id,
                "state": "UNKNOWN",
                "error": str(e)
            }
    
    async def cancel_execution(self, task_id: str) -> Dict[str, Any]:
        """Cancel a workflow execution"""
        try:
            from celery.result import AsyncResult
            
            task_result = AsyncResult(task_id)
            
            if task_result.state in ["PENDING", "STARTED", "PROGRESS"]:
                task_result.revoke(terminate=True)
                
                # Also try to cancel the job on the platform if it exists
                if task_result.info and "job_id" in task_result.info:
                    job_id = task_result.info["job_id"]
                    await self.job_service.cancel_job(job_id)
                
                logger.info(f"Cancelled workflow execution task: {task_id}")
                return {"success": True, "message": "Execution cancelled"}
            else:
                return {"success": False, "message": f"Cannot cancel task in state: {task_result.state}"}
        
        except Exception as e:
            logger.error(f"Error cancelling execution for task {task_id}: {e}")
            return {"success": False, "message": str(e)}
    
    async def retry_workflow(
        self, 
        workflow_id: int, 
        failed_job_id: int, 
        platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retry a failed workflow execution"""
        try:
            # Get the failed job to understand what went wrong
            failed_job = await self.job_service.get_job(failed_job_id)
            if not failed_job:
                raise ValueError(f"Job {failed_job_id} not found")
            
            if failed_job.status != JobStatus.FAILED:
                raise ValueError(f"Job {failed_job_id} is not in failed state")
            
            # Use the same platform unless specified otherwise
            retry_platform = platform or failed_job.platform
            
            # Re-submit with the same parameters
            result = await self.submit_workflow(
                workflow_id=workflow_id,
                platform=retry_platform,
                parameters=failed_job.parameters
            )
            
            # Update the original job to reference the retry
            await self.job_service.update_job(failed_job_id, JobUpdate(
                results={
                    **failed_job.results,
                    "retry_task_id": result["task_id"]
                }
            ))
            
            logger.info(f"Retrying workflow {workflow_id} (original job: {failed_job_id}, new task: {result['task_id']})")
            
            return {
                **result,
                "retry_of": failed_job_id,
                "original_error": failed_job.error_message
            }
        
        except Exception as e:
            logger.error(f"Error retrying workflow {workflow_id}: {e}")
            raise
    
    async def estimate_cost(
        self, 
        workflow_id: int, 
        platform: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Estimate the cost of running a workflow"""
        try:
            # Get workflow definition
            workflow = await self.workflow_service.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Basic cost estimation based on node types and platform
            # This is a simplified version - real implementation would use platform APIs
            
            nodes = workflow.definition.get("nodes", [])
            base_costs = {
                "galaxy": 0.10,  # $0.10 per hour base
                "nextflow": 0.15,  # $0.15 per hour base
                "seven_bridges": 0.20,  # $0.20 per hour base
                "aws_batch": 0.12   # $0.12 per hour base
            }
            
            node_multipliers = {
                "alphafold": 8.0,  # GPU-intensive
                "blast": 2.0,      # CPU-intensive
                "pymol": 1.0,      # Light processing
                "custom_script": 1.5,  # Variable
                "file_input": 0.1,     # Minimal
                "file_output": 0.1     # Minimal
            }
            
            base_cost = base_costs.get(platform, 0.15)
            total_multiplier = sum(node_multipliers.get(node.get("type"), 1.0) for node in nodes)
            
            # Estimate 2-hour runtime as baseline
            estimated_hours = 2.0
            estimated_cost = base_cost * total_multiplier * estimated_hours
            
            return {
                "platform": platform,
                "estimated_cost_usd": round(estimated_cost, 2),
                "estimated_runtime_hours": estimated_hours,
                "node_count": len(nodes),
                "cost_breakdown": {
                    "base_cost_per_hour": base_cost,
                    "complexity_multiplier": total_multiplier,
                    "estimated_hours": estimated_hours
                }
            }
        
        except Exception as e:
            logger.error(f"Error estimating cost for workflow {workflow_id}: {e}")
            raise
