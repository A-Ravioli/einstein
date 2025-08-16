"""
Workflow template endpoints
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.workflow import WorkflowTemplateCreate, WorkflowTemplateResponse, WorkflowCreate
from app.services.template_service import TemplateService
from app.services.workflow_service import WorkflowService

router = APIRouter()


@router.get("/", response_model=List[WorkflowTemplateResponse])
async def list_templates(
    category: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    public_only: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """List workflow templates"""
    service = TemplateService(db)
    templates = await service.list_templates(
        category=category,
        skip=skip,
        limit=limit,
        public_only=public_only
    )
    return templates


@router.get("/categories")
async def list_template_categories(db: AsyncSession = Depends(get_db)):
    """List available template categories"""
    # This could be enhanced to query the database for unique categories
    categories = [
        "structural_biology",
        "sequence_analysis", 
        "drug_discovery",
        "genomics",
        "proteomics",
        "machine_learning",
        "data_analysis"
    ]
    return {"categories": categories}


@router.get("/{template_id}", response_model=WorkflowTemplateResponse)
async def get_template(
    template_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific template"""
    service = TemplateService(db)
    template = await service.get_template(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )
    return template


@router.post("/", response_model=WorkflowTemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_template(
    template_data: WorkflowTemplateCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new workflow template"""
    service = TemplateService(db)
    template = await service.create_template(template_data)
    return template


@router.post("/{template_id}/use")
async def create_workflow_from_template(
    template_id: int,
    workflow_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    db: AsyncSession = Depends(get_db)
):
    """Create a new workflow from a template"""
    template_service = TemplateService(db)
    workflow_service = WorkflowService(db)
    
    try:
        # Create workflow definition from template
        workflow_data = await template_service.create_workflow_from_template(
            template_id=template_id,
            workflow_name=workflow_name,
            parameters=parameters
        )
        
        # Create the actual workflow
        workflow_create = WorkflowCreate(
            name=workflow_data["name"],
            description=workflow_data["description"],
            definition=workflow_data["definition"],
            tags=workflow_data["tags"]
        )
        
        workflow = await workflow_service.create_workflow(workflow_create)
        
        return {
            "workflow": workflow,
            "template_id": template_id,
            "message": f"Workflow created from template successfully"
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow from template: {str(e)}"
        )


@router.post("/{template_id}/download")
async def download_template(
    template_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Download a template (increments download count)"""
    service = TemplateService(db)
    
    template = await service.get_template(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )
    
    # Increment download count
    success = await service.increment_download_count(template_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record download"
        )
    
    return {
        "template": template,
        "download_count": template.download_count + 1,
        "message": "Template downloaded successfully"
    }
