from typing import Dict, List, Optional, Any
from .base_agent import BaseAgent


class ExperimentAgent(BaseAgent):
    """Agent for designing experiments to test research hypotheses"""
    
    def __init__(self, openai_client, settings):
        super().__init__(openai_client, settings, "ExperimentAgent")
    
    async def _setup(self):
        """Setup experiment design parameters"""
        self.experiment_types = [
            "computational", "wet_lab", "field_study", 
            "simulation", "survey", "meta_analysis"
        ]
    
    async def design_experiments(
        self, 
        hypothesis: Dict[str, Any], 
        constraints: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Design experiments to test a hypothesis"""
        
        self.logger.info("Designing experiments", hypothesis_id=hypothesis.get('id'))
        
        design_prompt = [
            {
                "role": "system",
                "content": self.create_system_prompt(
                    "an experimental design expert",
                    [
                        "Design rigorous experiments to test hypotheses",
                        "Consider controls, variables, and statistical power",
                        "Suggest multiple experimental approaches",
                        "Account for practical constraints and resources"
                    ]
                )
            },
            {
                "role": "user",
                "content": f"""Design experiments to test this hypothesis:

Hypothesis: {hypothesis.get('statement', '')}
Rationale: {hypothesis.get('rationale', '')}

Constraints: {constraints or 'None specified'}

Please provide 2-3 experimental designs with:
1. Experiment type (computational/wet_lab/field_study/etc.)
2. Methodology overview
3. Required resources
4. Expected timeline
5. Control conditions
6. Measurement approaches

Format as JSON array."""
            }
        ]
        
        try:
            response = await self.generate_structured_completion(design_prompt, schema={})
            experiments = response.get('experiments', [])
            
            # Format experiments
            formatted_experiments = []
            for i, exp in enumerate(experiments):
                formatted_exp = {
                    'id': f"exp_{hypothesis.get('id', 'unknown')}_{i+1}",
                    'hypothesis_id': hypothesis.get('id'),
                    'title': exp.get('title', f'Experiment {i+1}'),
                    'type': exp.get('type', 'computational'),
                    'methodology': exp.get('methodology', ''),
                    'resources': exp.get('resources', []),
                    'timeline': exp.get('timeline', 'Not specified'),
                    'controls': exp.get('controls', []),
                    'measurements': exp.get('measurements', [])
                }
                formatted_experiments.append(formatted_exp)
            
            return formatted_experiments
            
        except Exception as e:
            self.logger.error("Experiment design failed", error=str(e))
            return [{
                'id': f"exp_{hypothesis.get('id', 'unknown')}_fallback",
                'hypothesis_id': hypothesis.get('id'),
                'title': 'Basic Experimental Design',
                'type': 'computational',
                'methodology': f'Design experiment to test: {hypothesis.get("statement", "")}',
                'resources': ['Standard lab equipment'],
                'timeline': 'To be determined',
                'controls': ['Control group'],
                'measurements': ['Primary outcome measures']
            }] 