"""
Workflow template service for managing scientific workflow templates
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.workflow import WorkflowTemplate
from app.schemas.workflow import WorkflowTemplateCreate, WorkflowTemplateResponse
from loguru import logger


class TemplateService:
    """Service for workflow template management"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_templates(
        self,
        category: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        public_only: bool = True
    ) -> List[WorkflowTemplateResponse]:
        """List workflow templates"""
        try:
            query = select(WorkflowTemplate).offset(skip).limit(limit)
            
            filters = []
            if category:
                filters.append(WorkflowTemplate.category == category)
            if public_only:
                filters.append(WorkflowTemplate.is_public == True)
            
            if filters:
                query = query.where(and_(*filters))
            
            # Order by download count and creation date
            query = query.order_by(
                WorkflowTemplate.download_count.desc(),
                WorkflowTemplate.created_at.desc()
            )
            
            result = await self.db.execute(query)
            templates = result.scalars().all()
            
            return [WorkflowTemplateResponse.from_orm(template) for template in templates]
        
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            raise
    
    async def get_template(self, template_id: int) -> Optional[WorkflowTemplateResponse]:
        """Get a specific template by ID"""
        try:
            query = select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
            result = await self.db.execute(query)
            template = result.scalar_one_or_none()
            
            if template:
                return WorkflowTemplateResponse.from_orm(template)
            return None
        
        except Exception as e:
            logger.error(f"Error getting template {template_id}: {e}")
            raise
    
    async def create_template(
        self, 
        template_data: WorkflowTemplateCreate
    ) -> WorkflowTemplateResponse:
        """Create a new workflow template"""
        try:
            template = WorkflowTemplate(
                name=template_data.name,
                description=template_data.description,
                category=template_data.category,
                template_definition=template_data.template_definition,
                parameter_schema=template_data.parameter_schema,
                is_public=template_data.is_public
            )
            
            self.db.add(template)
            await self.db.commit()
            await self.db.refresh(template)
            
            logger.info(f"Created template: {template.id}")
            return WorkflowTemplateResponse.from_orm(template)
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating template: {e}")
            raise
    
    async def increment_download_count(self, template_id: int) -> bool:
        """Increment the download count for a template"""
        try:
            query = select(WorkflowTemplate).where(WorkflowTemplate.id == template_id)
            result = await self.db.execute(query)
            template = result.scalar_one_or_none()
            
            if template:
                template.download_count += 1
                await self.db.commit()
                logger.info(f"Incremented download count for template {template_id}")
                return True
            
            return False
        
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error incrementing download count for template {template_id}: {e}")
            raise
    
    async def create_workflow_from_template(
        self,
        template_id: int,
        workflow_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new workflow from a template"""
        try:
            # Get template
            template = await self.get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Apply parameters to template if provided
            workflow_definition = template.template_definition.copy()
            
            if parameters and template.parameter_schema:
                workflow_definition = self._apply_template_parameters(
                    workflow_definition,
                    parameters,
                    template.parameter_schema
                )
            
            # Increment download count
            await self.increment_download_count(template_id)
            
            # Return workflow definition ready for creation
            return {
                "name": workflow_name,
                "description": f"Created from template: {template.name}",
                "definition": workflow_definition,
                "tags": [template.category, "template", f"template:{template.name}"],
                "template_id": template_id
            }
        
        except Exception as e:
            logger.error(f"Error creating workflow from template {template_id}: {e}")
            raise
    
    def _apply_template_parameters(
        self,
        definition: Dict[str, Any],
        parameters: Dict[str, Any],
        parameter_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply user parameters to template definition"""
        try:
            # This is a simplified parameter substitution
            # In a real implementation, you'd want more sophisticated template processing
            
            modified_definition = definition.copy()
            nodes = modified_definition.get("nodes", [])
            
            for node in nodes:
                node_data = node.get("data", {})
                node_type = node.get("type", "")
                
                # Apply parameters based on node type and parameter schema
                if node_type in parameter_schema.get("node_parameters", {}):
                    node_params = parameter_schema["node_parameters"][node_type]
                    
                    for param_name, param_config in node_params.items():
                        if param_name in parameters:
                            # Apply parameter value to node data
                            if param_config.get("applies_to"):
                                target_field = param_config["applies_to"]
                                node_data[target_field] = parameters[param_name]
                            else:
                                node_data[param_name] = parameters[param_name]
            
            return modified_definition
        
        except Exception as e:
            logger.error(f"Error applying template parameters: {e}")
            return definition


# Predefined scientific workflow templates
SCIENTIFIC_TEMPLATES = [
    {
        "name": "AlphaFold Protein Structure Prediction",
        "description": "Predict protein structure using AlphaFold with visualization",
        "category": "structural_biology",
        "template_definition": {
            "nodes": [
                {
                    "id": "sequence_input",
                    "type": "file_input",
                    "data": {
                        "label": "Protein Sequence",
                        "file_type": "fasta",
                        "required": True
                    },
                    "position": {"x": 100, "y": 100}
                },
                {
                    "id": "alphafold_prediction",
                    "type": "alphafold",
                    "data": {
                        "model": "alphafold2",
                        "max_template_date": "2022-01-01",
                        "use_gpu": True
                    },
                    "position": {"x": 300, "y": 100}
                },
                {
                    "id": "structure_visualization",
                    "type": "pymol",
                    "data": {
                        "representation": "cartoon",
                        "color_scheme": "spectrum",
                        "output_format": "png"
                    },
                    "position": {"x": 500, "y": 100}
                },
                {
                    "id": "structure_output",
                    "type": "file_output",
                    "data": {
                        "label": "Predicted Structure"
                    },
                    "position": {"x": 700, "y": 100}
                }
            ],
            "edges": [
                {
                    "id": "edge1",
                    "source": "sequence_input",
                    "target": "alphafold_prediction",
                    "sourceHandle": "output",
                    "targetHandle": "sequence"
                },
                {
                    "id": "edge2",
                    "source": "alphafold_prediction",
                    "target": "structure_visualization",
                    "sourceHandle": "structure",
                    "targetHandle": "structure"
                },
                {
                    "id": "edge3",
                    "source": "structure_visualization",
                    "target": "structure_output",
                    "sourceHandle": "image",
                    "targetHandle": "input"
                }
            ]
        },
        "parameter_schema": {
            "parameters": {
                "alphafold_model": {
                    "type": "select",
                    "options": ["alphafold2", "alphafold3"],
                    "default": "alphafold2",
                    "description": "AlphaFold model version"
                },
                "visualization_style": {
                    "type": "select",
                    "options": ["cartoon", "surface", "sticks"],
                    "default": "cartoon",
                    "description": "Molecular visualization style"
                }
            },
            "node_parameters": {
                "alphafold": {
                    "alphafold_model": {
                        "applies_to": "model"
                    }
                },
                "pymol": {
                    "visualization_style": {
                        "applies_to": "representation"
                    }
                }
            }
        },
        "is_public": True
    },
    {
        "name": "BLAST Sequence Search and Analysis",
        "description": "Search protein/nucleotide databases with BLAST and analyze results",
        "category": "sequence_analysis",
        "template_definition": {
            "nodes": [
                {
                    "id": "query_sequence",
                    "type": "file_input",
                    "data": {
                        "label": "Query Sequence",
                        "file_type": "fasta",
                        "required": True
                    },
                    "position": {"x": 100, "y": 100}
                },
                {
                    "id": "blast_search",
                    "type": "blast",
                    "data": {
                        "program": "blastp",
                        "database": "nr",
                        "evalue": 0.001,
                        "max_target_seqs": 100
                    },
                    "position": {"x": 300, "y": 100}
                },
                {
                    "id": "results_analysis",
                    "type": "custom_script",
                    "data": {
                        "script_type": "python",
                        "script_content": "# Analyze BLAST results\nimport pandas as pd\n\n# Process BLAST output\n# Generate summary statistics\n# Create visualizations"
                    },
                    "position": {"x": 500, "y": 100}
                },
                {
                    "id": "results_output",
                    "type": "file_output",
                    "data": {
                        "label": "BLAST Results"
                    },
                    "position": {"x": 700, "y": 100}
                }
            ],
            "edges": [
                {
                    "id": "edge1",
                    "source": "query_sequence",
                    "target": "blast_search",
                    "sourceHandle": "output",
                    "targetHandle": "query"
                },
                {
                    "id": "edge2",
                    "source": "blast_search",
                    "target": "results_analysis",
                    "sourceHandle": "results",
                    "targetHandle": "input_files"
                },
                {
                    "id": "edge3",
                    "source": "results_analysis",
                    "target": "results_output",
                    "sourceHandle": "output_files",
                    "targetHandle": "input"
                }
            ]
        },
        "parameter_schema": {
            "parameters": {
                "blast_program": {
                    "type": "select",
                    "options": ["blastp", "blastn", "blastx", "tblastn"],
                    "default": "blastp",
                    "description": "BLAST program type"
                },
                "database": {
                    "type": "select",
                    "options": ["nr", "nt", "pdb", "swissprot"],
                    "default": "nr",
                    "description": "Database to search"
                },
                "evalue_threshold": {
                    "type": "number",
                    "default": 0.001,
                    "description": "E-value threshold"
                }
            },
            "node_parameters": {
                "blast": {
                    "blast_program": {"applies_to": "program"},
                    "database": {"applies_to": "database"},
                    "evalue_threshold": {"applies_to": "evalue"}
                }
            }
        },
        "is_public": True
    },
    {
        "name": "Drug Discovery Virtual Screening",
        "description": "Virtual screening of compound libraries against protein targets",
        "category": "drug_discovery",
        "template_definition": {
            "nodes": [
                {
                    "id": "compound_library",
                    "type": "file_input",
                    "data": {
                        "label": "Compound Library",
                        "file_type": "sdf",
                        "required": True
                    },
                    "position": {"x": 100, "y": 100}
                },
                {
                    "id": "target_structure",
                    "type": "file_input",
                    "data": {
                        "label": "Target Structure",
                        "file_type": "pdb",
                        "required": True
                    },
                    "position": {"x": 100, "y": 200}
                },
                {
                    "id": "molecular_docking",
                    "type": "custom_script",
                    "data": {
                        "script_type": "python",
                        "script_content": "# Molecular docking with AutoDock Vina\nfrom vina import Vina\n\n# Prepare receptor and ligands\n# Run docking simulation\n# Score and rank compounds"
                    },
                    "position": {"x": 300, "y": 150}
                },
                {
                    "id": "admet_prediction",
                    "type": "custom_script",
                    "data": {
                        "script_type": "python",
                        "script_content": "# ADMET prediction using RDKit\nfrom rdkit import Chem\nfrom rdkit.Chem import Descriptors\n\n# Calculate molecular properties\n# Predict ADMET properties\n# Filter compounds"
                    },
                    "position": {"x": 500, "y": 150}
                },
                {
                    "id": "results_visualization",
                    "type": "custom_script",
                    "data": {
                        "script_type": "python",
                        "script_content": "# Visualize screening results\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Create plots and charts\n# Generate summary report"
                    },
                    "position": {"x": 700, "y": 150}
                }
            ],
            "edges": [
                {
                    "id": "edge1",
                    "source": "compound_library",
                    "target": "molecular_docking",
                    "sourceHandle": "output",
                    "targetHandle": "input_files"
                },
                {
                    "id": "edge2",
                    "source": "target_structure",
                    "target": "molecular_docking",
                    "sourceHandle": "output",
                    "targetHandle": "input_files"
                },
                {
                    "id": "edge3",
                    "source": "molecular_docking",
                    "target": "admet_prediction",
                    "sourceHandle": "output_files",
                    "targetHandle": "input_files"
                },
                {
                    "id": "edge4",
                    "source": "admet_prediction",
                    "target": "results_visualization",
                    "sourceHandle": "output_files",
                    "targetHandle": "input_files"
                }
            ]
        },
        "parameter_schema": {
            "parameters": {
                "docking_exhaustiveness": {
                    "type": "number",
                    "default": 8,
                    "description": "Docking search exhaustiveness"
                },
                "top_compounds": {
                    "type": "number",
                    "default": 100,
                    "description": "Number of top compounds to analyze"
                }
            }
        },
        "is_public": True
    }
]


async def initialize_templates(db: AsyncSession):
    """Initialize default scientific workflow templates"""
    try:
        template_service = TemplateService(db)
        
        for template_data in SCIENTIFIC_TEMPLATES:
            # Check if template already exists
            existing_templates = await template_service.list_templates(
                category=template_data["category"],
                public_only=True
            )
            
            # Check by name
            if any(t.name == template_data["name"] for t in existing_templates):
                logger.info(f"Template '{template_data['name']}' already exists, skipping")
                continue
            
            # Create template
            await template_service.create_template(
                WorkflowTemplateCreate(**template_data)
            )
            
            logger.info(f"Created template: {template_data['name']}")
    
    except Exception as e:
        logger.error(f"Error initializing templates: {e}")
        raise
