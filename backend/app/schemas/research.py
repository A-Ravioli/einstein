from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ResearchStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ResearchProjectCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    research_goal: str = Field(..., min_length=10)
    field: Optional[str] = None
    subfield: Optional[str] = None
    keywords: Optional[List[str]] = None


class ResearchProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    research_goal: Optional[str] = None
    status: Optional[ResearchStatus] = None
    field: Optional[str] = None
    subfield: Optional[str] = None
    keywords: Optional[List[str]] = None


class ResearchProject(BaseModel):
    id: int
    title: str
    description: Optional[str]
    research_goal: str
    status: ResearchStatus
    field: Optional[str]
    subfield: Optional[str]
    keywords: Optional[List[str]]
    created_at: datetime
    updated_at: Optional[datetime]
    owner_id: int

    class Config:
        from_attributes = True


class WorkflowRequest(BaseModel):
    research_goal: str = Field(..., min_length=10)
    user_preferences: Optional[Dict[str, Any]] = None
    include_literature_review: bool = True
    include_hypothesis_generation: bool = True
    include_experiment_design: bool = True
    max_hypotheses: int = Field(default=5, ge=1, le=10)


class WorkflowResponse(BaseModel):
    workflow_id: str
    research_goal: str
    status: str
    steps: List[Dict[str, Any]]
    created_at: datetime
    
    
class LiteratureReviewRequest(BaseModel):
    query: str = Field(..., min_length=3)
    filters: Optional[Dict[str, Any]] = None
    max_papers: int = Field(default=50, ge=1, le=100)
    

class HypothesisGenerationRequest(BaseModel):
    research_goal: str = Field(..., min_length=10)
    literature_context: Optional[Dict[str, Any]] = None
    num_hypotheses: int = Field(default=5, ge=1, le=10)


class ExperimentDesignRequest(BaseModel):
    hypothesis_id: str
    hypothesis: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None


class ResearchUpdateRequest(BaseModel):
    interests: List[str] = Field(..., min_items=1)
    timeframe: str = Field(default="week", regex="^(day|week|month)$") 