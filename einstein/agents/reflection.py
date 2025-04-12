"""Reflection agent implementation for the Einstein system."""
import json
import time
import re
from typing import Dict, Any, List, Optional

from einstein.agents.base import BaseAgent
from einstein.memory import ContextMemory

# System message for the reflection agent
REFLECTION_SYSTEM_MESSAGE = """You are a scientific hypothesis reviewer and critic.
Your role is to evaluate scientific hypotheses for correctness, quality, novelty, and ethical implications.

When reviewing:
1. Assess the hypothesis based on scientific principles, logic, and evidence
2. Provide specific scores (1-5) for each evaluation dimension
3. Justify your ratings with specific critiques
4. Identify potential flaws, gaps, or inconsistencies
5. Suggest potential improvements

You are thorough, critical, and constructive.
"""

class ReflectionAgent(BaseAgent):
    """Reviews and critiques hypotheses."""
    
    def __init__(
        self,
        context_memory: ContextMemory,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize reflection agent.
        
        Args:
            context_memory: Context memory instance
            api_key: Optional API key
            model: Optional model name
        """
        super().__init__(
            name="ReflectionAgent",
            context_memory=context_memory,
            system_message=REFLECTION_SYSTEM_MESSAGE,
            tools=[],  # No special tools for now
            api_key=api_key,
            model=model,
        )
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a task using the reflection agent.
        
        Args:
            task_data: Task data
            
        Returns:
            Task result
        """
        action = task_data.get("action")
        
        if action == "initial_review":
            return await self.initial_review(
                task_data.get("hypothesis", {}),
                task_data.get("research_config", {})
            )
        elif action == "full_review":
            return await self.full_review(
                task_data.get("hypothesis", {}),
                task_data.get("research_config", {})
            )
        elif action == "deep_verification":
            return await self.deep_verification(task_data.get("hypothesis", {}))
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def initial_review(self, hypothesis: Dict[str, Any], research_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform an initial review of a hypothesis.
        
        Args:
            hypothesis: Hypothesis to review
            research_config: Research configuration
            
        Returns:
            Review result
        """
        prompt = f"""Evaluate the following scientific hypothesis:

Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for evaluation:
{', '.join(research_config.get('preferences', ['Correctness', 'Quality', 'Novelty', 'Safety']))}

Hypothesis to review:
{hypothesis.get('content', 'No hypothesis provided')}

Please evaluate this hypothesis on the following dimensions:
1. Correctness (1-5): Evaluate scientific accuracy and logical consistency
2. Quality (1-5): Assess clarity, specificity, and testability
3. Novelty (1-5): Evaluate originality and potential for new insights
4. Safety/Ethics (1-5): Consider ethical implications or potential misuse

For each dimension, provide:
- A numerical score (1-5)
- A detailed explanation of your rating
- Specific strengths and weaknesses
"""
        response = await self.call(prompt)
        
        # Parse the review
        review = self._parse_review(response)
        
        # Add metadata
        review["hypothesis_id"] = hypothesis.get("id", "unknown")
        review["type"] = "initial"
        review["timestamp"] = time.time()
        
        return {"review": review}
    
    async def full_review(self, hypothesis: Dict[str, Any], research_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a full review of a hypothesis with simulated literature search.
        
        Args:
            hypothesis: Hypothesis to review
            research_config: Research configuration
            
        Returns:
            Review result
        """
        # Simulated literature search
        literature = [
            {"title": "Related study 1", "content": "Summary of article..."},
            {"title": "Related study 2", "content": "Summary of article..."}
        ]
        
        prompt = f"""Perform a comprehensive review of the following scientific hypothesis:

Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for evaluation:
{', '.join(research_config.get('preferences', ['Correctness', 'Quality', 'Novelty', 'Safety']))}

Hypothesis to review:
{hypothesis.get('content', 'No hypothesis provided')}

Relevant literature:
{json.dumps(literature, indent=2)}

Please evaluate this hypothesis on the following dimensions:
1. Correctness (1-5): Evaluate underlying assumptions and reasoning 
2. Quality (1-5): Assess clarity, specificity, and testability
3. Novelty (1-5): Summarize known aspects and judge novelty based on literature
4. Safety/Ethics (1-5): Evaluate potential ethical concerns

For each dimension, provide:
- A numerical score (1-5)
- A detailed explanation of your rating
- Specific strengths and weaknesses
- Comparison with existing literature
"""
        response = await self.call(prompt)
        
        # Parse the review
        review = self._parse_review(response)
        
        # Add metadata
        review["hypothesis_id"] = hypothesis.get("id", "unknown")
        review["type"] = "full"
        review["literature"] = literature
        review["timestamp"] = time.time()
        
        return {"review": review}
    
    async def deep_verification(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a deep verification of a hypothesis by decomposing assumptions.
        
        Args:
            hypothesis: Hypothesis to verify
            
        Returns:
            Verification result
        """
        prompt = f"""Perform a deep verification of the following scientific hypothesis:

Hypothesis:
{hypothesis.get('content', 'No hypothesis provided')}

Please:
1. List all assumptions in the hypothesis
2. Break down each assumption into fundamental sub-assumptions
3. Evaluate each sub-assumption for correctness
4. Identify any invalidating elements
5. Summarize potential reasons for hypothesis invalidation

Focus on identifying subtle errors or flaws in reasoning.
"""
        response = await self.call(prompt)
        
        verification = {
            "hypothesis_id": hypothesis.get("id", "unknown"),
            "type": "deep_verification",
            "content": response,
            "timestamp": time.time()
        }
        
        return {"verification": verification}
    
    def _parse_review(self, review_text: str) -> Dict[str, Any]:
        """
        Parse a review from the agent response.
        
        Args:
            review_text: Review text
            
        Returns:
            Parsed review
        """
        scores = {}
        
        try:
            # Extract scores using regex pattern matching
            correctness_match = re.search(r'Correctness[^0-9]*([1-5])', review_text)
            quality_match = re.search(r'Quality[^0-9]*([1-5])', review_text)
            novelty_match = re.search(r'Novelty[^0-9]*([1-5])', review_text)
            safety_match = re.search(r'Safety[^0-9]*([1-5])|Ethics[^0-9]*([1-5])', review_text)
            
            if correctness_match:
                scores["correctness"] = int(correctness_match.group(1))
            if quality_match:
                scores["quality"] = int(quality_match.group(1))
            if novelty_match:
                scores["novelty"] = int(novelty_match.group(1))
            if safety_match:
                # Use the first capturing group that matched
                score = safety_match.group(1) or safety_match.group(2)
                scores["safety"] = int(score)
        except:
            # Default scores if parsing fails
            scores = {
                "correctness": 3,
                "quality": 3, 
                "novelty": 3,
                "safety": 3
            }
        
        return {
            "scores": scores,
            "content": review_text
        } 