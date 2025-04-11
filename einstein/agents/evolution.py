"""Evolution agent implementation for the Einstein system."""
import json
import time
from typing import Dict, Any, List, Optional

from einstein_pkg.agents.base import BaseAgent
from einstein_pkg.memory import ContextMemory

# System message for the evolution agent
EVOLUTION_SYSTEM_MESSAGE = """You are a scientific hypothesis improvement specialist.
Your role is to refine and evolve scientific hypotheses based on feedback and evaluation.

When improving hypotheses:
1. Address specific critiques and limitations identified in reviews
2. Maintain the core scientific insights of the original hypothesis
3. Enhance clarity, testability, and scientific rigor
4. Consider novel perspectives or mechanisms that strengthen the hypothesis
5. Provide a complete, revised hypothesis that builds upon the original

You are innovative but grounded in scientific principles.
"""

class EvolutionAgent(BaseAgent):
    """Improves existing hypotheses."""
    
    def __init__(
        self,
        context_memory: ContextMemory,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize evolution agent.
        
        Args:
            context_memory: Context memory instance
            api_key: Optional API key
            model: Optional model name
        """
        super().__init__(
            name="EvolutionAgent",
            context_memory=context_memory,
            system_message=EVOLUTION_SYSTEM_MESSAGE,
            tools=[],  # No special tools for now
            api_key=api_key,
            model=model,
        )
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a task using the evolution agent.
        
        Args:
            task_data: Task data
            
        Returns:
            Task result
        """
        action = task_data.get("action")
        
        if action == "improve_hypothesis":
            improved = await self.improve_hypothesis(
                task_data.get("hypothesis", {}),
                task_data.get("research_config", {}),
                task_data.get("reviews", {})
            )
            return {"improved_hypothesis": improved}
        elif action == "combine_hypotheses":
            combined = await self.combine_hypotheses(
                task_data.get("hypotheses", []),
                task_data.get("research_config", {})
            )
            return {"combined_hypothesis": combined}
        elif action == "out_of_box_thinking":
            novel = await self.out_of_box_thinking(
                task_data.get("research_config", {}),
                task_data.get("existing_hypotheses", [])
            )
            return {"novel_hypothesis": novel}
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def improve_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        research_config: Dict[str, Any],
        reviews: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Improve a hypothesis based on reviews and research configuration.
        
        Args:
            hypothesis: Hypothesis to improve
            research_config: Research configuration
            reviews: Hypothesis reviews
            
        Returns:
            Improved hypothesis
        """
        # Get reviews for the hypothesis if available
        hypothesis_review = reviews.get(hypothesis.get("id", ""), {})
        
        prompt = f"""Improve the following scientific hypothesis based on reviews and critiques:

Research Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for a strong hypothesis:
{', '.join(research_config.get('preferences', ['Clarity', 'Testability', 'Novelty']))}

Original Hypothesis:
{hypothesis.get('content', 'No content provided')}

Reviews and Critiques:
{json.dumps(hypothesis_review, indent=2) if hypothesis_review else "No reviews available"}

Please improve this hypothesis by:
1. Addressing specific critiques mentioned in the reviews
2. Enhancing clarity and specificity
3. Improving testability and scientific rigor
4. Strengthening the conceptual framework
5. Maintaining or enhancing the originality of the core idea

Provide a complete, revised hypothesis that builds upon the strengths of the original while addressing its limitations.
"""
        response = await self.call(prompt)
        
        # Create the improved hypothesis object
        improved_hypothesis = {
            "id": f"{hypothesis.get('id', 'unknown')}_improved",
            "content": response,
            "parent_id": hypothesis.get("id"),
            "source": "evolution",
            "timestamp": time.time()
        }
        
        return improved_hypothesis
    
    async def combine_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        research_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine multiple hypotheses to create a new, improved hypothesis.
        
        Args:
            hypotheses: List of hypotheses to combine
            research_config: Research configuration
            
        Returns:
            Combined hypothesis
        """
        # Extract content from hypotheses
        hypothesis_contents = [h.get('content', 'No content provided') for h in hypotheses]
        
        prompt = f"""Create a new unified hypothesis by combining insights from multiple hypotheses:

Research Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for a strong hypothesis:
{', '.join(research_config.get('preferences', ['Clarity', 'Testability', 'Novelty']))}

Existing Hypotheses to Combine:

{json.dumps(hypothesis_contents, indent=2)}

Please create a single unified hypothesis that:
1. Integrates the strengths and key insights from each input hypothesis
2. Resolves any contradictions between them
3. Creates a coherent conceptual framework
4. Maintains scientific rigor and testability
5. Results in a hypothesis that is stronger than any individual input

Provide a complete, integrated hypothesis that represents the best combination of the input hypotheses.
"""
        response = await self.call(prompt)
        
        # Get IDs of parent hypotheses
        parent_ids = [h.get("id", f"unknown_{i}") for i, h in enumerate(hypotheses)]
        
        # Create the combined hypothesis object
        combined_hypothesis = {
            "id": self.context_memory.create_id("combined"),
            "content": response,
            "parent_ids": parent_ids,
            "source": "combination",
            "timestamp": time.time()
        }
        
        return combined_hypothesis
    
    async def out_of_box_thinking(
        self,
        research_config: Dict[str, Any],
        existing_hypotheses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a novel hypothesis through out-of-box thinking.
        
        Args:
            research_config: Research configuration
            existing_hypotheses: Existing hypotheses
            
        Returns:
            Novel hypothesis
        """
        prompt = f"""Generate a completely novel hypothesis through out-of-box thinking:

Research Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for a robust hypothesis:
{', '.join(research_config.get('preferences', ['Be novel', 'Be testable', 'Be clear']))}

Existing hypotheses (for reference only):
{json.dumps([h.get('content', 'No content') for h in existing_hypotheses], indent=2)}

Generate a completely novel hypothesis that:
1. Takes a fundamentally different approach than existing hypotheses
2. Challenges conventional thinking in the field
3. Draws inspiration from analogous domains or principles
4. Remains scientifically plausible and testable

Think outside-the-box while ensuring scientific rigor.
"""
        response = await self.call(prompt)
        
        # Create the novel hypothesis object
        novel_hypothesis = {
            "id": self.context_memory.create_id("novel"),
            "content": response,
            "source": "out_of_box",
            "timestamp": time.time()
        }
        
        return novel_hypothesis 