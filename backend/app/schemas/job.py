"""
Job Pydantic schemas
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from app.models.job import JobStatus, PlatformType


class JobBase(BaseModel):
    """Base job schema"""
    workflow_id: int
    platform: PlatformType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    platform_config: Dict[str, Any] = Field(default_factory=dict)


class JobCreate(JobBase):
    """Schema for creating a job"""
    pass


class JobUpdate(BaseModel):
    """Schema for updating a job"""
    status: Optional[JobStatus] = None
    external_job_id: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    cpu_hours: Optional[float] = None
    memory_gb_hours: Optional[float] = None
    cost_estimate: Optional[float] = None


class JobStepResponse(BaseModel):
    """Schema for job step responses"""
    id: int
    step_name: str
    step_type: Optional[str] = None
    order_index: int
    status: JobStatus
    external_step_id: Optional[str] = None
    parameters: Dict[str, Any]
    input_files: List[str]
    output_files: List[str]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    results: Dict[str, Any]
    logs: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class JobResponse(JobBase):
    """Schema for job responses"""
    id: int
    status: JobStatus
    external_job_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    results: Dict[str, Any]
    logs: Optional[str] = None
    error_message: Optional[str] = None
    cpu_hours: Optional[float] = None
    memory_gb_hours: Optional[float] = None
    cost_estimate: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    job_steps: List[JobStepResponse] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Schema for job list responses"""
    jobs: List[JobResponse]
    total: int
    skip: int
    limit: int


class JobExecutionRequest(BaseModel):
    """Schema for job execution requests"""
    workflow_id: int
    platform: PlatformType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    platform_config: Dict[str, Any] = Field(default_factory=dict)
    priority: Optional[int] = Field(default=0, ge=0, le=10)


class JobStatusUpdate(BaseModel):
    """Schema for job status updates"""
    status: JobStatus
    message: Optional[str] = None
    progress_percentage: Optional[float] = Field(None, ge=0, le=100)
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None
