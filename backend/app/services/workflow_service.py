"""
Workflow service layer
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from app.models.workflow import Workflow, WorkflowTemplate
from app.models.job import Job
from app.schemas.workflow import (
    WorkflowCreate, 
    WorkflowUpdate, 
    WorkflowResponse,
    WorkflowValidationResult
)
from app.services.job_service import JobService
from loguru import logger


class WorkflowService:
    """Service for workflow management"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_workflows(
        self, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[WorkflowResponse]:
        """List workflows with optional filtering"""
        try:
            query = select(Workflow).offset(skip).limit(limit)
            
            if status:
                query = query.where(Workflow.status == status)
            
            if tags:
                # Filter by tags (assuming tags is a JSON array)
                for tag in tags:
                    query = query.where(Workflow.tags.contains([tag]))
            
            result = await self.db.execute(query)
            workflows = result.scalars().all()
            
            return [WorkflowResponse.from_orm(workflow) for workflow in workflows]
        
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            raise
    
    async def get_workflow(self, workflow_id: int) -> Optional[WorkflowResponse]:
        """Get a specific workflow by ID"""
        try:
            query = select(Workflow).where(Workflow.id == workflow_id)
            result = await self.db.execute(query)
            workflow = result.scalar_one_or_none()
            
            if workflow:
                return WorkflowResponse.from_orm(workflow)
            return None
        
        except Exception as e:
            logger.error(f"Error getting workflow {workflow_id}: {e}")
            raise
    
    async def create_workflow(self, workflow_data: WorkflowCreate) -> WorkflowResponse:
        """Create a new workflow"""
        try:
            workflow = Workflow(
                name=workflow_data.name,
                description=workflow_data.description,
                definition=workflow_data.definition,
                status=workflow_data.status,
                version=workflow_data.version,
                tags=workflow_data.tags
            )
            
            self.db.add(workflow)
            await self.db.commit()
            await self.db.refresh(workflow)
            
            logger.info(f"Created workflow: {workflow.id}")
            return WorkflowResponse.from_orm(workflow)
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating workflow: {e}")
            raise
    
    async def update_workflow(
        self, 
        workflow_id: int, 
        workflow_data: WorkflowUpdate
    ) -> Optional[WorkflowResponse]:
        """Update an existing workflow"""
        try:
            query = select(Workflow).where(Workflow.id == workflow_id)
            result = await self.db.execute(query)
            workflow = result.scalar_one_or_none()
            
            if not workflow:
                return None
            
            # Update fields if provided
            if workflow_data.name is not None:
                workflow.name = workflow_data.name
            if workflow_data.description is not None:
                workflow.description = workflow_data.description
            if workflow_data.definition is not None:
                workflow.definition = workflow_data.definition
            if workflow_data.status is not None:
                workflow.status = workflow_data.status
            if workflow_data.version is not None:
                workflow.version = workflow_data.version
            if workflow_data.tags is not None:
                workflow.tags = workflow_data.tags
            
            await self.db.commit()
            await self.db.refresh(workflow)
            
            logger.info(f"Updated workflow: {workflow.id}")
            return WorkflowResponse.from_orm(workflow)
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating workflow {workflow_id}: {e}")
            raise
    
    async def delete_workflow(self, workflow_id: int) -> bool:
        """Delete a workflow"""
        try:
            query = select(Workflow).where(Workflow.id == workflow_id)
            result = await self.db.execute(query)
            workflow = result.scalar_one_or_none()
            
            if not workflow:
                return False
            
            await self.db.delete(workflow)
            await self.db.commit()
            
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting workflow {workflow_id}: {e}")
            raise
    
    async def validate_workflow(self, workflow_id: int) -> WorkflowValidationResult:
        """Validate a workflow definition"""
        try:
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                return WorkflowValidationResult(
                    is_valid=False,
                    errors=["Workflow not found"]
                )
            
            definition = workflow.definition
            errors = []
            warnings = []
            
            # Basic validation
            if not isinstance(definition, dict):
                errors.append("Workflow definition must be a JSON object")
                return WorkflowValidationResult(is_valid=False, errors=errors)
            
            # Check for required fields
            if "nodes" not in definition:
                errors.append("Workflow definition must contain 'nodes'")
            
            if "edges" not in definition:
                errors.append("Workflow definition must contain 'edges'")
            
            if errors:
                return WorkflowValidationResult(is_valid=False, errors=errors)
            
            nodes = definition.get("nodes", [])
            edges = definition.get("edges", [])
            
            # Validate nodes
            node_ids = set()
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    errors.append(f"Node {i} must be an object")
                    continue
                
                if "id" not in node:
                    errors.append(f"Node {i} missing required 'id' field")
                    continue
                
                node_id = node["id"]
                if node_id in node_ids:
                    errors.append(f"Duplicate node ID: {node_id}")
                node_ids.add(node_id)
                
                if "type" not in node:
                    warnings.append(f"Node {node_id} missing 'type' field")
            
            # Validate edges
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    errors.append(f"Edge {i} must be an object")
                    continue
                
                if "source" not in edge:
                    errors.append(f"Edge {i} missing 'source' field")
                if "target" not in edge:
                    errors.append(f"Edge {i} missing 'target' field")
                
                # Check that source and target nodes exist
                if edge.get("source") not in node_ids:
                    errors.append(f"Edge {i} references non-existent source node: {edge.get('source')}")
                if edge.get("target") not in node_ids:
                    errors.append(f"Edge {i} references non-existent target node: {edge.get('target')}")
            
            # TODO: Add more sophisticated validation (cycles, data types, etc.)
            
            return WorkflowValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                node_count=len(nodes),
                edge_count=len(edges)
            )
        
        except Exception as e:
            logger.error(f"Error validating workflow {workflow_id}: {e}")
            return WorkflowValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    async def execute_workflow(self, workflow_id: int, platform: str) -> Job:
        """Execute a workflow on a specified platform"""
        try:
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Validate workflow before execution
            validation = await self.validate_workflow(workflow_id)
            if not validation.is_valid:
                raise ValueError(f"Invalid workflow: {', '.join(validation.errors)}")
            
            # Create and submit job
            job_service = JobService(self.db)
            job = await job_service.create_job(
                workflow_id=workflow_id,
                platform=platform,
                parameters={},
                platform_config={}
            )
            
            logger.info(f"Submitted workflow {workflow_id} for execution on {platform}")
            return job
        
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            raise
