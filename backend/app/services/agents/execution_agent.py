from typing import Dict, List, Optional, Any
import asyncio
import subprocess
from .base_agent import BaseAgent


class ExecutionAgent(BaseAgent):
    """Agent for executing computational experiments and analyses"""
    
    def __init__(self, openai_client, settings):
        super().__init__(openai_client, settings, "ExecutionAgent")
    
    async def _setup(self):
        """Setup execution environment"""
        self.supported_languages = ["python", "r", "julia", "bash"]
    
    async def execute_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a computational experiment"""
        
        self.logger.info("Executing experiment", exp_id=experiment.get('id'))
        
        if experiment.get('type') != 'computational':
            return {
                'status': 'skipped',
                'message': 'Only computational experiments can be auto-executed',
                'experiment_id': experiment.get('id')
            }
        
        # Generate execution script
        script = await self._generate_execution_script(experiment)
        
        # Execute (in a safe, sandboxed way in production)
        result = await self._execute_script(script, experiment)
        
        return {
            'experiment_id': experiment.get('id'),
            'status': result.get('status', 'completed'),
            'script': script,
            'output': result.get('output', ''),
            'errors': result.get('errors', []),
            'execution_time': result.get('execution_time', 0)
        }
    
    async def _generate_execution_script(self, experiment: Dict[str, Any]) -> str:
        """Generate execution script for an experiment"""
        
        script_prompt = [
            {
                "role": "system",
                "content": "You are a computational research specialist. Generate safe, executable code."
            },
            {
                "role": "user",
                "content": f"""Generate a Python script to execute this experiment:

Title: {experiment.get('title', '')}
Methodology: {experiment.get('methodology', '')}
Measurements: {experiment.get('measurements', [])}

Requirements:
- Use standard scientific libraries (numpy, pandas, matplotlib, scipy)
- Include proper error handling
- Generate meaningful output and visualizations
- Keep execution time under 60 seconds
- No file system access outside working directory

Provide complete, executable Python code."""
            }
        ]
        
        try:
            script = await self.generate_completion(script_prompt)
            return script
        except Exception as e:
            self.logger.error("Script generation failed", error=str(e))
            return f"# Error generating script: {str(e)}\nprint('Script generation failed')"
    
    async def _execute_script(self, script: str, experiment: Dict) -> Dict[str, Any]:
        """Execute script safely (mock implementation for demo)"""
        
        # In production, this would run in a proper sandbox
        # For demo purposes, we'll simulate execution
        
        try:
            # Simulate script execution
            await asyncio.sleep(1)  # Simulate processing time
            
            return {
                'status': 'completed',
                'output': f'Mock execution of experiment {experiment.get("id")}\nResults: Analysis completed successfully.',
                'errors': [],
                'execution_time': 1.0
            }
            
        except Exception as e:
            self.logger.error("Script execution failed", error=str(e))
            return {
                'status': 'failed',
                'output': '',
                'errors': [str(e)],
                'execution_time': 0
            } 