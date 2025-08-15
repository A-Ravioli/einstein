"""
Workflow database models
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.core.database import Base


class WorkflowStatus(str, enum.Enum):
    """Workflow status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class Workflow(Base):
    """Workflow model"""
    __tablename__ = "workflows"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Workflow definition stored as JSON
    definition = Column(JSON, nullable=False)
    
    # Metadata
    status = Column(Enum(WorkflowStatus), default=WorkflowStatus.DRAFT)
    version = Column(String(50), default="1.0.0")
    tags = Column(JSON, default=list)  # List of tags for categorization
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    jobs = relationship("Job", back_populates="workflow", cascade="all, delete-orphan")
    files = relationship("WorkflowFile", back_populates="workflow", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Workflow(id={self.id}, name='{self.name}', status='{self.status}')>"


class WorkflowTemplate(Base):
    """Workflow template model for common scientific workflows"""
    __tablename__ = "workflow_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))  # e.g., "bioinformatics", "chemistry", "physics"
    
    # Template definition
    template_definition = Column(JSON, nullable=False)
    
    # Configuration schema for template parameters
    parameter_schema = Column(JSON)
    
    # Metadata
    is_public = Column(Boolean, default=True)
    download_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<WorkflowTemplate(id={self.id}, name='{self.name}', category='{self.category}')>"
