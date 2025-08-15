"""
Job execution database models
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class JobStatus(str, enum.Enum):
    """Job status enumeration"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlatformType(str, enum.Enum):
    """Platform type enumeration"""
    GALAXY = "galaxy"
    NEXTFLOW = "nextflow"
    SEVEN_BRIDGES = "seven_bridges"
    DNA_NEXUS = "dna_nexus"
    AWS_BATCH = "aws_batch"
    CUSTOM = "custom"


class Job(Base):
    """Job execution model"""
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Workflow relationship
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=False)
    workflow = relationship("Workflow", back_populates="jobs")
    
    # Execution details
    platform = Column(Enum(PlatformType), nullable=False)
    external_job_id = Column(String(255))  # ID from external platform
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    
    # Configuration
    parameters = Column(JSON, default=dict)
    platform_config = Column(JSON, default=dict)
    
    # Execution metadata
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    
    # Results and logs
    results = Column(JSON, default=dict)
    logs = Column(Text)
    error_message = Column(Text)
    
    # Resource usage
    cpu_hours = Column(Float)
    memory_gb_hours = Column(Float)
    cost_estimate = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    job_steps = relationship("JobStep", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Job(id={self.id}, workflow_id={self.workflow_id}, platform='{self.platform}', status='{self.status}')>"


class JobStep(Base):
    """Individual step within a job execution"""
    __tablename__ = "job_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Job relationship
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    job = relationship("Job", back_populates="job_steps")
    
    # Step details
    step_name = Column(String(255), nullable=False)
    step_type = Column(String(100))  # e.g., "alphafold", "blast", "pymol"
    order_index = Column(Integer, nullable=False)
    
    # Execution details
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    external_step_id = Column(String(255))
    
    # Step configuration
    parameters = Column(JSON, default=dict)
    input_files = Column(JSON, default=list)
    output_files = Column(JSON, default=list)
    
    # Execution metadata
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    
    # Results and logs
    results = Column(JSON, default=dict)
    logs = Column(Text)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<JobStep(id={self.id}, job_id={self.job_id}, step_name='{self.step_name}', status='{self.status}')>"
