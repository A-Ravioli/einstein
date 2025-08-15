# Platform Integration Guide

## Overview

This guide explains how to integrate Einstein Workflow Builder with external scientific computing platforms, enabling multi-cloud workflow execution without managing your own infrastructure.

## Supported Platforms

### 1. Galaxy Project
- **Type**: Open-source computational biology platform
- **Use Cases**: Genomics, proteomics, transcriptomics
- **API**: REST API with comprehensive workflow management
- **Authentication**: API keys

### 2. Nextflow Tower
- **Type**: Commercial workflow management platform
- **Use Cases**: Bioinformatics pipelines, multi-cloud execution
- **API**: REST API with cloud-native execution
- **Authentication**: Bearer tokens

### 3. Seven Bridges
- **Type**: Genomics analysis platform
- **Use Cases**: Clinical genomics, population genomics
- **API**: REST API with CWL workflow support
- **Authentication**: OAuth2/API tokens

### 4. DNAnexus
- **Type**: Secure cloud-based bioanalysis platform
- **Use Cases**: Clinical genomics, pharmaceutical research
- **API**: REST API with dx-toolkit integration
- **Authentication**: API tokens

## Architecture Overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Einstein UI       │    │   Einstein API      │    │   Platform          │
│   (Next.js)         │────│   (FastAPI)         │────│   Adapters          │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                      │                        │
                                      │                        │
                           ┌─────────────────────┐    ┌─────────────────────┐
                           │   Job Queue         │    │   External          │
                           │   (Celery/Redis)    │    │   Platforms         │
                           └─────────────────────┘    └─────────────────────┘
```

## Platform Adapter Implementation

### Base Adapter Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class PlatformAdapter(ABC):
    """Abstract base class for all platform adapters"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with the platform"""
        pass
    
    @abstractmethod
    async def submit_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Submit workflow and return external job ID"""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current job status and results"""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools/processes"""
        pass
```

### Galaxy Adapter Example

```python
class GalaxyAdapter(PlatformAdapter):
    def __init__(self):
        self.base_url = settings.GALAXY_BASE_URL
        self.api_key = settings.GALAXY_API_KEY
    
    async def submit_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Submit workflow to Galaxy"""
        
        # 1. Convert Einstein format to Galaxy format
        galaxy_workflow = self._convert_to_galaxy_format(workflow_definition)
        
        # 2. Upload input files to Galaxy
        input_files = await self._upload_input_files(workflow_definition)
        
        # 3. Submit workflow with inputs
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/workflows/{galaxy_workflow['id']}/invocations",
                headers={"X-API-KEY": self.api_key},
                json={
                    "inputs": input_files,
                    "parameters": workflow_definition.get("parameters", {})
                }
            )
            
        if response.status_code == 200:
            invocation = response.json()
            return invocation["id"]
        else:
            raise Exception(f"Workflow submission failed: {response.text}")
    
    def _convert_to_galaxy_format(self, workflow_def: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Einstein workflow format to Galaxy workflow format"""
        nodes = workflow_def.get("nodes", [])
        edges = workflow_def.get("edges", [])
        
        galaxy_steps = {}
        for i, node in enumerate(nodes):
            if node["type"] == "alphafold":
                galaxy_steps[str(i)] = {
                    "tool_id": "alphafold",
                    "tool_version": "2.3.0",
                    "inputs": self._map_node_inputs(node, edges),
                    "parameters": node.get("data", {})
                }
            elif node["type"] == "blast":
                galaxy_steps[str(i)] = {
                    "tool_id": "ncbi_blastp_wrapper",
                    "tool_version": "0.3.3",
                    "inputs": self._map_node_inputs(node, edges),
                    "parameters": node.get("data", {})
                }
        
        return {
            "name": workflow_def.get("name", "Einstein Workflow"),
            "steps": galaxy_steps,
            "inputs": self._extract_workflow_inputs(nodes),
            "outputs": self._extract_workflow_outputs(nodes)
        }
```

### Nextflow Adapter Example

```python
class NextflowAdapter(PlatformAdapter):
    def __init__(self):
        self.base_url = settings.NEXTFLOW_TOWER_URL
        self.token = settings.NEXTFLOW_TOWER_TOKEN
    
    async def submit_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Submit workflow to Nextflow Tower"""
        
        # 1. Convert to Nextflow script
        nextflow_script = self._convert_to_nextflow_script(workflow_definition)
        
        # 2. Submit to Tower
        submission_data = {
            "launch": {
                "pipeline": nextflow_script,
                "workDir": "s3://nextflow-workdir",
                "profiles": ["standard"],
                "params": workflow_definition.get("parameters", {})
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/workflow/launch",
                headers={"Authorization": f"Bearer {self.token}"},
                json=submission_data
            )
            
        if response.status_code == 200:
            result = response.json()
            return result["workflowId"]
        else:
            raise Exception(f"Workflow submission failed: {response.text}")
    
    def _convert_to_nextflow_script(self, workflow_def: Dict[str, Any]) -> str:
        """Convert Einstein workflow to Nextflow script"""
        nodes = workflow_def.get("nodes", [])
        edges = workflow_def.get("edges", [])
        
        script_parts = [
            "#!/usr/bin/env nextflow",
            "",
            "params.input = null",
            "params.output = './results'",
            "",
            "workflow {",
        ]
        
        # Generate processes from nodes
        for node in nodes:
            if node["type"] == "alphafold":
                script_parts.extend([
                    f"    alphafold_ch = alphafold(params.input)",
                ])
            elif node["type"] == "blast":
                script_parts.extend([
                    f"    blast_ch = blast(params.input)",
                ])
        
        script_parts.append("}")
        
        # Add process definitions
        for node in nodes:
            if node["type"] == "alphafold":
                script_parts.extend([
                    "",
                    "process alphafold {",
                    "    container 'alphafold:latest'",
                    "    publishDir params.output, mode: 'copy'",
                    "",
                    "    input:",
                    "    path sequence",
                    "",
                    "    output:",
                    "    path 'structure.pdb'",
                    "",
                    "    script:",
                    "    '''",
                    "    python /app/run_alphafold.py \\",
                    "        --fasta_paths=!{sequence} \\",
                    "        --model_preset=monomer \\",
                    "        --db_preset=full_dbs",
                    "    '''",
                    "}"
                ])
        
        return "\n".join(script_parts)
```

## Workflow Definition Format

### Einstein Internal Format

```json
{
  "name": "AlphaFold Protein Structure Prediction",
  "description": "Predict protein structure using AlphaFold",
  "nodes": [
    {
      "id": "input_1",
      "type": "file_input",
      "data": {
        "file_type": "fasta",
        "label": "Protein Sequence",
        "required": true
      },
      "position": {"x": 100, "y": 100}
    },
    {
      "id": "alphafold_1", 
      "type": "alphafold",
      "data": {
        "model": "alphafold2",
        "max_template_date": "2022-01-01",
        "use_gpu": true,
        "num_multimer_predictions": 5
      },
      "position": {"x": 300, "y": 100}
    },
    {
      "id": "pymol_1",
      "type": "pymol_visualization",
      "data": {
        "representation": "cartoon",
        "color_scheme": "spectrum",
        "output_format": "png"
      },
      "position": {"x": 500, "y": 100}
    },
    {
      "id": "output_1",
      "type": "file_output",
      "data": {
        "file_types": ["pdb", "png"],
        "label": "Results"
      },
      "position": {"x": 700, "y": 100}
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "input_1",
      "target": "alphafold_1",
      "sourceHandle": "output",
      "targetHandle": "sequence"
    },
    {
      "id": "edge_2", 
      "source": "alphafold_1",
      "target": "pymol_1",
      "sourceHandle": "structure",
      "targetHandle": "input"
    },
    {
      "id": "edge_3",
      "source": "pymol_1",
      "target": "output_1", 
      "sourceHandle": "visualization",
      "targetHandle": "input"
    }
  ],
  "parameters": {
    "global_timeout": 3600,
    "retry_attempts": 3
  }
}
```

### Platform-Specific Conversions

**Galaxy Workflow Format:**
```json
{
  "name": "AlphaFold Protein Structure Prediction",
  "steps": {
    "0": {
      "tool_id": "upload1",
      "tool_version": "1.1.6",
      "inputs": {},
      "parameters": {"file_type": "fasta"}
    },
    "1": {
      "tool_id": "alphafold",
      "tool_version": "2.3.0",
      "inputs": {
        "sequence": {"output_name": "output", "step": 0}
      },
      "parameters": {
        "model": "alphafold2",
        "max_template_date": "2022-01-01"
      }
    }
  }
}
```

**Nextflow Script:**
```nextflow
#!/usr/bin/env nextflow

params.input = null
params.output = './results'

workflow {
    sequence_ch = channel.fromPath(params.input)
    alphafold_ch = alphafold(sequence_ch)
    pymol_ch = pymol_visualization(alphafold_ch)
    
    pymol_ch.view()
}

process alphafold {
    container 'alphafold:latest'
    publishDir params.output, mode: 'copy'
    
    input:
    path sequence
    
    output:
    path 'structure.pdb'
    
    script:
    """
    python /app/run_alphafold.py \\
        --fasta_paths=${sequence} \\
        --model_preset=monomer \\
        --max_template_date=2022-01-01
    """
}
```

## Job Monitoring and Status Updates

### Status Polling Strategy

```python
class JobMonitoringService:
    def __init__(self):
        self.platform_service = PlatformService()
    
    async def monitor_job(self, job_id: int):
        """Monitor job status and update database"""
        job = await self.get_job(job_id)
        if not job:
            return
        
        adapter = self.platform_service.get_adapter(job.platform)
        if not adapter:
            return
        
        while job.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]:
            try:
                # Get status from external platform
                platform_status = await adapter.get_job_status(job.external_job_id)
                
                # Update local job status
                new_status = self._map_platform_status(platform_status)
                if new_status != job.status:
                    await self.update_job_status(job_id, new_status, platform_status)
                
                # Wait before next poll
                await asyncio.sleep(30)  # Poll every 30 seconds
                
                # Refresh job from database
                job = await self.get_job(job_id)
                
            except Exception as e:
                logger.error(f"Error monitoring job {job_id}: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _map_platform_status(self, platform_status: Dict[str, Any]) -> JobStatus:
        """Map platform-specific status to internal status"""
        # Galaxy status mapping
        if "state" in platform_status:
            galaxy_status = platform_status["state"]
            mapping = {
                "new": JobStatus.PENDING,
                "queued": JobStatus.QUEUED,
                "running": JobStatus.RUNNING,
                "ok": JobStatus.COMPLETED,
                "error": JobStatus.FAILED,
                "deleted": JobStatus.CANCELLED
            }
            return mapping.get(galaxy_status, JobStatus.PENDING)
        
        # Nextflow status mapping
        if "status" in platform_status:
            nextflow_status = platform_status["status"]
            mapping = {
                "submitted": JobStatus.PENDING,
                "running": JobStatus.RUNNING,
                "succeeded": JobStatus.COMPLETED,
                "failed": JobStatus.FAILED,
                "cancelled": JobStatus.CANCELLED
            }
            return mapping.get(nextflow_status, JobStatus.PENDING)
        
        return JobStatus.PENDING
```

## File Management and Data Transfer

### S3-Compatible Storage Integration

```python
class FileStorageService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.S3_BUCKET
    
    async def upload_file_to_platform(
        self, 
        file_path: str, 
        platform: str, 
        workflow_id: int
    ) -> str:
        """Upload file to platform-specific storage"""
        
        if platform == "galaxy":
            return await self._upload_to_galaxy(file_path, workflow_id)
        elif platform == "nextflow":
            return await self._upload_to_s3_for_nextflow(file_path, workflow_id)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    async def _upload_to_galaxy(self, file_path: str, workflow_id: int) -> str:
        """Upload file to Galaxy data library"""
        # Implementation for Galaxy file upload
        pass
    
    async def _upload_to_s3_for_nextflow(self, file_path: str, workflow_id: int) -> str:
        """Upload file to S3 for Nextflow access"""
        s3_key = f"workflows/{workflow_id}/inputs/{os.path.basename(file_path)}"
        
        # Upload to S3
        self.s3_client.upload_file(file_path, self.bucket, s3_key)
        
        # Return S3 URL
        return f"s3://{self.bucket}/{s3_key}"
```

## Error Handling and Retry Logic

### Robust Error Handling

```python
class WorkflowExecutionEngine:
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 60  # seconds
    
    async def execute_workflow_with_retry(
        self, 
        workflow_id: int, 
        platform: str
    ) -> Dict[str, Any]:
        """Execute workflow with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                result = await self._execute_workflow(workflow_id, platform)
                return {"success": True, "result": result}
                
            except PlatformTemporaryError as e:
                # Temporary error - retry
                if attempt < self.max_retries - 1:
                    logger.warning(f"Temporary error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Max retries exceeded for workflow {workflow_id}")
                    return {"success": False, "error": str(e)}
                    
            except PlatformPermanentError as e:
                # Permanent error - don't retry
                logger.error(f"Permanent error for workflow {workflow_id}: {e}")
                return {"success": False, "error": str(e)}
                
            except Exception as e:
                # Unknown error - don't retry
                logger.error(f"Unknown error for workflow {workflow_id}: {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}
```

## Testing Platform Integrations

### Mock Adapters for Testing

```python
class MockGalaxyAdapter(PlatformAdapter):
    """Mock Galaxy adapter for testing"""
    
    def __init__(self):
        self.submitted_workflows = {}
        self.job_counter = 0
    
    async def submit_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        self.job_counter += 1
        job_id = f"mock_galaxy_job_{self.job_counter}"
        self.submitted_workflows[job_id] = {
            "status": "running",
            "workflow": workflow_definition,
            "submitted_at": datetime.utcnow()
        }
        return job_id
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        job = self.submitted_workflows.get(job_id, {})
        return {"state": job.get("status", "error")}
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_galaxy_workflow_submission():
    """Test workflow submission to Galaxy"""
    adapter = MockGalaxyAdapter()
    
    workflow_def = {
        "nodes": [
            {"id": "input1", "type": "file_input"},
            {"id": "alphafold1", "type": "alphafold"}
        ],
        "edges": [
            {"source": "input1", "target": "alphafold1"}
        ]
    }
    
    job_id = await adapter.submit_workflow(workflow_def)
    assert job_id.startswith("mock_galaxy_job_")
    
    status = await adapter.get_job_status(job_id)
    assert status["state"] == "running"
```

## Deployment and Configuration

### Environment Variables

```bash
# Platform API Configuration
GALAXY_BASE_URL=https://usegalaxy.org
GALAXY_API_KEY=your_galaxy_api_key

NEXTFLOW_TOWER_URL=https://api.tower.nf
NEXTFLOW_TOWER_TOKEN=your_nextflow_token

SEVEN_BRIDGES_URL=https://api.sbgenomics.com/v2
SEVEN_BRIDGES_TOKEN=your_seven_bridges_token

DNA_NEXUS_URL=https://api.dnanexus.com
DNA_NEXUS_TOKEN=your_dnanexus_token

# File Storage
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET=einstein-workflows

# Job Monitoring
JOB_POLLING_INTERVAL=30
MAX_JOB_RETRIES=3
JOB_TIMEOUT_HOURS=24
```

### Kubernetes Deployment

```yaml
# kubernetes/platform-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: einstein-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: einstein-backend
  template:
    metadata:
      labels:
        app: einstein-backend
    spec:
      containers:
      - name: backend
        image: einstein/backend:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: einstein-secrets
              key: database-url
        - name: GALAXY_API_KEY
          valueFrom:
            secretKeyRef:
              name: platform-secrets
              key: galaxy-api-key
        ports:
        - containerPort: 8000
```

This comprehensive integration guide provides everything needed to implement the multi-cloud workflow execution strategy, allowing scientists to leverage existing platforms without managing infrastructure.
