from typing import Dict, List, Optional, Any
from .base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """Agent for analyzing experimental results and synthesizing research"""
    
    def __init__(self, openai_client, settings):
        super().__init__(openai_client, settings, "AnalysisAgent")
    
    async def _setup(self):
        """Setup analysis parameters"""
        self.analysis_types = [
            "statistical", "visual", "qualitative", "meta_analysis"
        ]
    
    async def analyze_results(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experimental results"""
        
        analysis_prompt = [
            {
                "role": "system",
                "content": self.create_system_prompt(
                    "a data analysis expert specializing in scientific research",
                    [
                        "Analyze experimental data rigorously",
                        "Apply appropriate statistical methods",
                        "Identify patterns and significant findings",
                        "Provide clear interpretations and conclusions"
                    ]
                )
            },
            {
                "role": "user",
                "content": f"""Analyze these experimental results:

Experiment: {experiment_data.get('title', '')}
Data/Output: {experiment_data.get('output', '')}
Status: {experiment_data.get('status', '')}

Provide:
1. Key findings
2. Statistical significance (if applicable)
3. Limitations and caveats
4. Interpretation of results
5. Recommendations for next steps

Format as JSON."""
            }
        ]
        
        try:
            analysis = await self.generate_structured_completion(analysis_prompt, schema={})
            return {
                'experiment_id': experiment_data.get('experiment_id'),
                'analysis': analysis,
                'key_findings': analysis.get('key_findings', []),
                'significance': analysis.get('statistical_significance', 'Not determined'),
                'limitations': analysis.get('limitations', []),
                'interpretation': analysis.get('interpretation', ''),
                'next_steps': analysis.get('recommendations', [])
            }
        except Exception as e:
            self.logger.error("Results analysis failed", error=str(e))
            return {
                'experiment_id': experiment_data.get('experiment_id'),
                'error': f'Analysis failed: {str(e)}'
            }
    
    async def synthesize_research_plan(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize literature, hypotheses, and experiments into research plan"""
        
        synthesis_prompt = [
            {
                "role": "system",
                "content": "You are a research strategist who creates comprehensive research plans."
            },
            {
                "role": "user",
                "content": f"""Synthesize this research information into a strategic plan:

Literature Review Summary: {research_data.get('literature_review', {}).get('summary', {})}
Generated Hypotheses: {len(research_data.get('hypotheses', []))} hypotheses
Designed Experiments: {len(research_data.get('experiments', []))} experiments

Create a research plan with:
1. Research priorities
2. Timeline recommendations
3. Resource requirements
4. Risk assessment
5. Success metrics

Format as JSON."""
            }
        ]
        
        try:
            plan = await self.generate_structured_completion(synthesis_prompt, schema={})
            return plan
        except Exception as e:
            self.logger.error("Research synthesis failed", error=str(e))
            return {'error': f'Synthesis failed: {str(e)}'} 