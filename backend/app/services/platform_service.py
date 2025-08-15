"""
Platform integration service layer
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import httpx
from loguru import logger

from app.core.config import settings


class PlatformAdapter(ABC):
    """Abstract base class for platform adapters"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with the platform"""
        pass
    
    @abstractmethod
    async def check_status(self) -> Dict[str, Any]:
        """Check platform availability and status"""
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on the platform"""
        pass
    
    @abstractmethod
    async def submit_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Submit a workflow for execution and return job ID"""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status and results"""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        pass


class GalaxyAdapter(PlatformAdapter):
    """Galaxy platform adapter"""
    
    def __init__(self):
        self.base_url = settings.GALAXY_BASE_URL
        self.api_key = settings.GALAXY_API_KEY
        self.headers = {"X-API-KEY": self.api_key} if self.api_key else {}
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with Galaxy"""
        try:
            api_key = credentials.get("api_key")
            if not api_key:
                return False
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/users/current",
                    headers={"X-API-KEY": api_key}
                )
                return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Galaxy authentication error: {e}")
            return False
    
    async def check_status(self) -> Dict[str, Any]:
        """Check Galaxy platform status"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    return {
                        "status": "available",
                        "authenticated": bool(self.api_key),
                        "version": response.json().get("version_major", "unknown")
                    }
                else:
                    return {"status": "unavailable", "authenticated": False}
        
        except Exception as e:
            logger.error(f"Galaxy status check error: {e}")
            return {"status": "error", "message": str(e), "authenticated": False}
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available Galaxy tools"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/tools",
                    headers=self.headers
                )
                if response.status_code == 200:
                    tools = response.json()
                    return [
                        {
                            "id": tool.get("id"),
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "version": tool.get("version", ""),
                        }
                        for tool in tools
                    ]
                return []
        
        except Exception as e:
            logger.error(f"Galaxy tools list error: {e}")
            return []
    
    async def submit_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Submit workflow to Galaxy"""
        try:
            # TODO: Convert internal workflow format to Galaxy format
            galaxy_workflow = self._convert_to_galaxy_format(workflow_definition)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/workflows",
                    headers=self.headers,
                    json=galaxy_workflow
                )
                if response.status_code == 200:
                    workflow_data = response.json()
                    # TODO: Actually invoke the workflow
                    return workflow_data.get("id", "")
                else:
                    raise Exception(f"Workflow submission failed: {response.text}")
        
        except Exception as e:
            logger.error(f"Galaxy workflow submission error: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get Galaxy job status"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/jobs/{job_id}",
                    headers=self.headers
                )
                if response.status_code == 200:
                    return response.json()
                return {"state": "error", "message": "Job not found"}
        
        except Exception as e:
            logger.error(f"Galaxy job status error: {e}")
            return {"state": "error", "message": str(e)}
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel Galaxy job"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/api/jobs/{job_id}",
                    headers=self.headers
                )
                return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Galaxy job cancellation error: {e}")
            return False
    
    def _convert_to_galaxy_format(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal workflow format to Galaxy format"""
        # TODO: Implement actual conversion logic
        return {
            "name": workflow_definition.get("name", "Untitled Workflow"),
            "steps": {},
            "inputs": {},
            "outputs": {}
        }


class NextflowAdapter(PlatformAdapter):
    """Nextflow Tower platform adapter"""
    
    def __init__(self):
        self.base_url = settings.NEXTFLOW_TOWER_URL
        self.token = settings.NEXTFLOW_TOWER_TOKEN
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with Nextflow Tower"""
        try:
            token = credentials.get("token")
            if not token:
                return False
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/user-info",
                    headers={"Authorization": f"Bearer {token}"}
                )
                return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Nextflow authentication error: {e}")
            return False
    
    async def check_status(self) -> Dict[str, Any]:
        """Check Nextflow Tower status"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/service-info")
                if response.status_code == 200:
                    return {
                        "status": "available",
                        "authenticated": bool(self.token),
                        "service": "nextflow-tower"
                    }
                else:
                    return {"status": "unavailable", "authenticated": False}
        
        except Exception as e:
            logger.error(f"Nextflow status check error: {e}")
            return {"status": "error", "message": str(e), "authenticated": False}
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available Nextflow processes/modules"""
        # Nextflow doesn't have a centralized tool registry like Galaxy
        # Return common bioinformatics processes
        return [
            {"id": "blast", "name": "BLAST", "description": "Basic Local Alignment Search Tool"},
            {"id": "bwa", "name": "BWA", "description": "Burrows-Wheeler Aligner"},
            {"id": "samtools", "name": "SAMtools", "description": "Tools for manipulating SAM/BAM files"},
            {"id": "fastqc", "name": "FastQC", "description": "Quality control tool for sequence data"},
        ]
    
    async def submit_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Submit workflow to Nextflow Tower"""
        try:
            # TODO: Convert internal workflow format to Nextflow script
            nextflow_script = self._convert_to_nextflow_format(workflow_definition)
            
            submission_data = {
                "launch": {
                    "pipeline": nextflow_script,
                    "workDir": "s3://nextflow-workdir",
                    "revision": "main"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/workflow/launch",
                    headers=self.headers,
                    json=submission_data
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("workflowId", "")
                else:
                    raise Exception(f"Workflow submission failed: {response.text}")
        
        except Exception as e:
            logger.error(f"Nextflow workflow submission error: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get Nextflow job status"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/workflow/{job_id}",
                    headers=self.headers
                )
                if response.status_code == 200:
                    return response.json()
                return {"status": "error", "message": "Workflow not found"}
        
        except Exception as e:
            logger.error(f"Nextflow job status error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel Nextflow job"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/workflow/{job_id}/cancel",
                    headers=self.headers
                )
                return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Nextflow job cancellation error: {e}")
            return False
    
    def _convert_to_nextflow_format(self, workflow_definition: Dict[str, Any]) -> str:
        """Convert internal workflow format to Nextflow script"""
        # TODO: Implement actual conversion logic
        return """
        #!/usr/bin/env nextflow
        
        workflow {
            // Generated workflow
        }
        """


class PlatformService:
    """Service for managing platform integrations"""
    
    def __init__(self):
        self.adapters = {
            "galaxy": GalaxyAdapter(),
            "nextflow": NextflowAdapter(),
            # TODO: Add more platform adapters
        }
    
    async def list_available_platforms(self) -> List[Dict[str, Any]]:
        """List all available platforms"""
        platforms = []
        for name, adapter in self.adapters.items():
            status = await adapter.check_status()
            platforms.append({
                "name": name,
                "status": status.get("status", "unknown"),
                "authenticated": status.get("authenticated", False),
                "description": self._get_platform_description(name)
            })
        return platforms
    
    async def check_platform_status(self, platform_name: str) -> Dict[str, Any]:
        """Check specific platform status"""
        adapter = self.adapters.get(platform_name)
        if not adapter:
            return {"status": "not_supported", "message": f"Platform {platform_name} not supported"}
        
        return await adapter.check_status()
    
    async def authenticate_platform(
        self, 
        platform_name: str, 
        credentials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Authenticate with a platform"""
        adapter = self.adapters.get(platform_name)
        if not adapter:
            return {"success": False, "message": f"Platform {platform_name} not supported"}
        
        success = await adapter.authenticate(credentials)
        return {
            "success": success,
            "message": "Authentication successful" if success else "Authentication failed"
        }
    
    async def list_platform_tools(self, platform_name: str) -> List[Dict[str, Any]]:
        """List tools available on a platform"""
        adapter = self.adapters.get(platform_name)
        if not adapter:
            return []
        
        return await adapter.list_tools()
    
    def get_adapter(self, platform_name: str) -> Optional[PlatformAdapter]:
        """Get platform adapter by name"""
        return self.adapters.get(platform_name)
    
    def _get_platform_description(self, platform_name: str) -> str:
        """Get platform description"""
        descriptions = {
            "galaxy": "Open-source computational biology platform",
            "nextflow": "Workflow management system for bioinformatics",
            "seven_bridges": "Genomics analysis platform",
            "dna_nexus": "Secure cloud-based bioanalysis platform"
        }
        return descriptions.get(platform_name, "Scientific computing platform")
