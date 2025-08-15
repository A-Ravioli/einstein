from typing import Dict, List, Optional, Any
import asyncio
from .base_agent import BaseAgent


class HypothesisAgent(BaseAgent):
    """
    Specialized agent for generating, evaluating, and refining research hypotheses.
    Uses literature context and research goals to create testable hypotheses.
    """
    
    def __init__(self, openai_client, settings):
        super().__init__(openai_client, settings, "HypothesisAgent")
    
    async def _setup(self):
        """Setup hypothesis generation parameters"""
        self.evaluation_criteria = [
            "testability",
            "novelty", 
            "feasibility",
            "potential_impact",
            "clarity"
        ]
    
    async def generate_hypotheses(
        self, 
        research_goal: str, 
        literature_context: Optional[Dict] = None,
        num_hypotheses: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate multiple research hypotheses based on research goal and literature"""
        self.logger.info("Generating hypotheses", goal=research_goal, num=num_hypotheses)
        
        # Generate initial hypotheses
        raw_hypotheses = await self._generate_raw_hypotheses(
            research_goal, literature_context, num_hypotheses
        )
        
        # Evaluate and score each hypothesis
        evaluated_hypotheses = await self._evaluate_hypotheses(raw_hypotheses, research_goal)
        
        # Rank hypotheses by overall score
        ranked_hypotheses = sorted(
            evaluated_hypotheses, 
            key=lambda x: x.get('overall_score', 0), 
            reverse=True
        )
        
        return ranked_hypotheses
    
    async def _generate_raw_hypotheses(
        self, 
        research_goal: str, 
        literature_context: Optional[Dict], 
        num_hypotheses: int
    ) -> List[Dict[str, Any]]:
        """Generate initial set of hypotheses"""
        
        # Prepare context from literature review
        literature_summary = ""
        if literature_context:
            key_findings = literature_context.get('key_findings', [])
            research_gaps = literature_context.get('research_gaps', [])
            literature_summary = f"""
Key Findings from Literature:
{chr(10).join(f"- {finding}" for finding in key_findings[:5])}

Research Gaps Identified:
{chr(10).join(f"- {gap}" for gap in research_gaps[:3])}
"""
        
        generation_prompt = [
            {
                "role": "system",
                "content": self.create_system_prompt(
                    "a research hypothesis generation expert with deep scientific knowledge",
                    [
                        "Generate novel, testable research hypotheses",
                        "Base hypotheses on current literature and identified gaps",
                        "Ensure hypotheses are specific, measurable, and falsifiable",
                        "Consider practical constraints and feasibility"
                    ]
                )
            },
            {
                "role": "user", 
                "content": f"""Generate {num_hypotheses} distinct research hypotheses for:

Research Goal: {research_goal}

{literature_summary}

For each hypothesis, provide:
1. Title (concise, descriptive)
2. Statement (clear, testable hypothesis)
3. Rationale (why this hypothesis is worth investigating)

Format as JSON with array of hypotheses."""
            }
        ]
        
        try:
            response = await self.generate_structured_completion(generation_prompt, schema={})
            hypotheses = response.get('hypotheses', [])
            
            # Ensure we have the expected structure
            formatted_hypotheses = []
            for i, hyp in enumerate(hypotheses):
                formatted_hyp = {
                    'id': f"hyp_{i+1}",
                    'title': hyp.get('title', f'Hypothesis {i+1}'),
                    'statement': hyp.get('statement', ''),
                    'rationale': hyp.get('rationale', ''),
                    'generated_from_goal': research_goal
                }
                formatted_hypotheses.append(formatted_hyp)
            
            return formatted_hypotheses
            
        except Exception as e:
            self.logger.error("Failed to generate hypotheses", error=str(e))
            # Return a fallback hypothesis
            return [{
                'id': 'hyp_fallback',
                'title': 'Fallback Hypothesis',
                'statement': f'Investigation of {research_goal} will reveal significant patterns.',
                'rationale': 'Generated due to processing error',
                'generated_from_goal': research_goal
            }]
    
    async def _evaluate_hypotheses(
        self, 
        hypotheses: List[Dict], 
        research_goal: str
    ) -> List[Dict[str, Any]]:
        """Evaluate hypotheses on multiple criteria"""
        
        evaluated = []
        
        # Process hypotheses in parallel
        tasks = [
            self._evaluate_single_hypothesis(hyp, research_goal) 
            for hyp in hypotheses
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                evaluated.append(result)
            else:
                self.logger.warning("Hypothesis evaluation failed", error=str(result))
        
        return evaluated
    
    async def _evaluate_single_hypothesis(
        self, 
        hypothesis: Dict, 
        research_goal: str
    ) -> Dict[str, Any]:
        """Evaluate a single hypothesis on multiple criteria"""
        
        evaluation_prompt = [
            {
                "role": "system",
                "content": "You are a research methodology expert. Evaluate hypotheses scientifically."
            },
            {
                "role": "user",
                "content": f"""Evaluate this hypothesis (score 0-100 for each):

Hypothesis: {hypothesis.get('statement', '')}
Rationale: {hypothesis.get('rationale', '')}

Score on:
1. Testability - Can this be tested?
2. Novelty - How original is this?
3. Feasibility - How practical to test?
4. Impact - Potential scientific impact?
5. Clarity - How clear is the hypothesis?

Format as JSON with numerical scores."""
            }
        ]
        
        try:
            evaluation = await self.generate_structured_completion(evaluation_prompt, schema={})
            
            # Calculate overall score
            scores = {
                'testability_score': evaluation.get('testability', 50),
                'novelty_score': evaluation.get('novelty', 50),
                'feasibility_score': evaluation.get('feasibility', 50),
                'impact_score': evaluation.get('impact', 50),
                'clarity_score': evaluation.get('clarity', 50)
            }
            
            overall_score = sum(scores.values()) / len(scores)
            
            # Combine hypothesis with evaluation
            evaluated_hypothesis = hypothesis.copy()
            evaluated_hypothesis.update(scores)
            evaluated_hypothesis['overall_score'] = overall_score
            
            return evaluated_hypothesis
            
        except Exception as e:
            self.logger.error("Single hypothesis evaluation failed", error=str(e))
            # Return hypothesis with default scores
            hypothesis.update({
                'testability_score': 50,
                'novelty_score': 50, 
                'feasibility_score': 50,
                'impact_score': 50,
                'clarity_score': 50,
                'overall_score': 50
            })
            return hypothesis
    
    async def refine_hypothesis(
        self, 
        hypothesis: Dict[str, Any], 
        feedback: str
    ) -> Dict[str, Any]:
        """Refine a hypothesis based on user feedback"""
        
        refinement_prompt = [
            {
                "role": "system",
                "content": self.create_system_prompt(
                    "a research mentor who helps refine and improve research hypotheses",
                    [
                        "Incorporate user feedback to improve hypotheses",
                        "Maintain scientific rigor while addressing concerns",
                        "Suggest specific improvements and modifications",
                        "Preserve the core research intent while enhancing clarity"
                    ]
                )
            },
            {
                "role": "user",
                "content": f"""Refine this research hypothesis based on the provided feedback:

Original Hypothesis:
Title: {hypothesis.get('title', '')}
Statement: {hypothesis.get('statement', '')}
Rationale: {hypothesis.get('rationale', '')}

User Feedback: {feedback}

Evaluation Scores:
- Testability: {hypothesis.get('testability_score', 'N/A')}
- Novelty: {hypothesis.get('novelty_score', 'N/A')}
- Feasibility: {hypothesis.get('feasibility_score', 'N/A')}

Please provide a refined version that addresses the feedback while improving the overall quality.

Format as JSON with the same structure as the original hypothesis."""
            }
        ]
        
        try:
            refined = await self.generate_structured_completion(refinement_prompt, schema={})
            
            # Preserve original metadata
            refined_hypothesis = hypothesis.copy()
            refined_hypothesis.update({
                'title': refined.get('title', hypothesis.get('title')),
                'statement': refined.get('statement', hypothesis.get('statement')),
                'rationale': refined.get('rationale', hypothesis.get('rationale')),
                'refinement_history': hypothesis.get('refinement_history', []) + [{
                    'feedback': feedback,
                    'refined_at': asyncio.current_task().get_name(),
                    'changes_made': refined.get('changes_made', [])
                }]
            })
            
            return refined_hypothesis
            
        except Exception as e:
            self.logger.error("Hypothesis refinement failed", error=str(e))
            return hypothesis
    
    async def generate_research_questions(
        self, 
        hypothesis: Dict[str, Any]
    ) -> List[str]:
        """Generate specific research questions from a hypothesis"""
        
        questions_prompt = [
            {
                "role": "system",
                "content": "You are a research methodology expert who breaks down hypotheses into specific, actionable research questions."
            },
            {
                "role": "user",
                "content": f"""Generate 3-5 specific research questions that would need to be answered to test this hypothesis:

Hypothesis: {hypothesis.get('statement', '')}
Rationale: {hypothesis.get('rationale', '')}

The questions should be:
1. Specific and focused
2. Empirically answerable
3. Logically connected to the hypothesis
4. Feasible to investigate

Format as a JSON array of question strings."""
            }
        ]
        
        try:
            response = await self.generate_structured_completion(questions_prompt, schema={})
            return response.get('questions', [])
        except Exception as e:
            self.logger.error("Research questions generation failed", error=str(e))
            return [f"How can we test the hypothesis: {hypothesis.get('statement', '')}?"]
    
    async def compare_hypotheses(
        self, 
        hypotheses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple hypotheses and provide recommendations"""
        
        if len(hypotheses) < 2:
            return {"error": "Need at least 2 hypotheses to compare"}
        
        comparison_prompt = [
            {
                "role": "system",
                "content": self.create_system_prompt(
                    "a research strategy advisor who compares and ranks research hypotheses",
                    [
                        "Compare hypotheses objectively across multiple dimensions",
                        "Identify unique strengths and weaknesses of each hypothesis",
                        "Provide strategic recommendations for research prioritization",
                        "Consider resource constraints and potential impact"
                    ]
                )
            },
            {
                "role": "user",
                "content": f"""Compare these research hypotheses and provide a strategic analysis:

{chr(10).join([f"Hypothesis {i+1}: {h.get('title', '')} - {h.get('statement', '')} (Score: {h.get('overall_score', 'N/A')})" for i, h in enumerate(hypotheses)])}

Provide:
1. Strengths and weaknesses of each hypothesis
2. Recommended prioritization and rationale
3. Potential for combining or sequencing hypotheses
4. Resource allocation recommendations
5. Risk assessment for each hypothesis

Format as JSON."""
            }
        ]
        
        try:
            comparison = await self.generate_structured_completion(comparison_prompt, schema={})
            return comparison
        except Exception as e:
            self.logger.error("Hypothesis comparison failed", error=str(e))
            return {"error": f"Comparison failed: {str(e)}"} 