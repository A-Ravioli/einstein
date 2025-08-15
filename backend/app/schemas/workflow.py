"""
Workflow Pydantic schemas
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from app.models.workflow import WorkflowStatus


class WorkflowBase(BaseModel):
    """Base workflow schema"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    definition: Dict[str, Any] = Field(..., description="Workflow definition as JSON")
    tags: List[str] = Field(default_factory=list)


class WorkflowCreate(WorkflowBase):
    """Schema for creating a workflow"""
    version: str = Field(default="1.0.0", max_length=50)
    status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT)


class WorkflowUpdate(BaseModel):
    """Schema for updating a workflow"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    status: Optional[WorkflowStatus] = None
    version: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = None


class WorkflowResponse(WorkflowBase):
    """Schema for workflow responses"""
    id: int
    status: WorkflowStatus
    version: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class WorkflowListResponse(BaseModel):
    """Schema for workflow list responses"""
    workflows: List[WorkflowResponse]
    total: int
    skip: int
    limit: int


class WorkflowTemplateBase(BaseModel):
    """Base workflow template schema"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=100)
    template_definition: Dict[str, Any] = Field(..., description="Template definition as JSON")
    parameter_schema: Optional[Dict[str, Any]] = None
    is_public: bool = Field(default=True)


class WorkflowTemplateCreate(WorkflowTemplateBase):
    """Schema for creating a workflow template"""
    pass


class WorkflowTemplateResponse(WorkflowTemplateBase):
    """Schema for workflow template responses"""
    id: int
    download_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class WorkflowValidationResult(BaseModel):
    """Schema for workflow validation results"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    node_count: int = Field(default=0)
    edge_count: int = Field(default=0)
    estimated_runtime_minutes: Optional[float] = None
