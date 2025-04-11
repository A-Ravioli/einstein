"""Generation agent implementation for the Einstein system."""
import json
import time
from typing import Dict, Any, List, Optional

from einstein_pkg.agents.base import BaseAgent
from einstein_pkg.memory import ContextMemory
from einstein_pkg.tools import search_literature_tool

# System message for the generation agent
GENERATION_SYSTEM_MESSAGE = """You are a scientific hypothesis generation expert.
Your role is to generate novel, testable scientific hypotheses based on research goals and literature.

When generating hypotheses:
1. Ensure they are specific, testable, and address the research goal
2. Base them on sound scientific principles and relevant literature
3. Format each hypothesis clearly, starting with "Hypothesis X:"
4. Provide sufficient detail and rationale for each hypothesis
5. Ensure hypotheses are distinct from each other

You are creative but scientifically rigorous.
"""

class GenerationAgent(BaseAgent):
    """Generates initial research hypotheses."""
    
    def __init__(
        self,
        context_memory: ContextMemory,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize generation agent.
        
        Args:
            context_memory: Context memory instance
            api_key: Optional API key
            model: Optional model name
        """
        super().__init__(
            name="GenerationAgent",
            context_memory=context_memory,
            system_message=GENERATION_SYSTEM_MESSAGE,
            tools=[search_literature_tool],
            api_key=api_key,
            model=model,
        )
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a task using the generation agent.
        
        Args:
            task_data: Task data
            
        Returns:
            Task result
        """
        action = task_data.get("action")
        
        if action == "generate_initial_hypotheses":
            return await self.generate_initial_hypotheses(task_data.get("params", {}))
        elif action == "simulated_debate":
            return await self.simulated_debate(
                task_data.get("research_config", {}),
                task_data.get("existing_hypotheses", [])
            )
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def generate_initial_hypotheses(self, research_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate initial research hypotheses.
        
        Args:
            research_config: Research configuration
            
        Returns:
            Generated hypotheses
        """
        prompt = f"""Generate 3 detailed scientific hypotheses to address the following research goal:

Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for a strong hypothesis:
{', '.join(research_config.get('preferences', ['Be specific', 'Be testable', 'Be novel']))}

Format each hypothesis clearly starting with "Hypothesis 1:", "Hypothesis 2:", etc.

For each hypothesis, provide:
1. The core claim
2. The mechanism or process involved
3. How it could be tested
4. Expected outcomes if true
"""
        response = await self.call(prompt)
        
        # Parse hypotheses from the response
        hypotheses = self._parse_hypotheses(response)
        
        return {"hypotheses": hypotheses}
    
    async def simulated_debate(self, research_config: Dict[str, Any], existing_hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a hypothesis through simulated scientific debate.
        
        Args:
            research_config: Research configuration
            existing_hypotheses: Existing hypotheses
            
        Returns:
            Generated hypothesis
        """
        prompt = f"""You are an expert participating in a collaborative discourse concerning the generation of a hypothesis. 
You will engage in a simulated discussion with other experts.

Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for a high-quality hypothesis:
{', '.join(research_config.get('preferences', ['Be specific', 'Be testable', 'Be novel']))}

Existing hypotheses:
{json.dumps([h.get('content', 'No content') for h in existing_hypotheses], indent=2)}

Simulate a scientific debate among experts to refine these hypotheses and generate a novel hypothesis.
Conclude your debate with "HYPOTHESIS:" followed by the final hypothesis.
"""
        response = await self.call(prompt)
        
        # Extract the final hypothesis from the debate
        final_hypothesis = self._extract_final_hypothesis(response)
        
        hypothesis = {
            "id": self.context_memory.create_id("h"),
            "content": final_hypothesis,
            "source": "debate",
            "timestamp": time.time()
        }
        
        return {"hypothesis": hypothesis}
    
    def _parse_hypotheses(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse hypotheses from the agent response.
        
        Args:
            response_text: Response text
            
        Returns:
            List of hypotheses
        """
        hypotheses = []
        
        # Simple splitting by "Hypothesis" marker
        parts = response_text.split("Hypothesis")
        for i, part in enumerate(parts[1:], 1):  # Skip the first part (before "Hypothesis 1")
            # Extract until the next hypothesis or end of text
            content = part.strip()
            
            hypotheses.append({
                "id": self.context_memory.create_id("h"),
                "content": content,
                "source": "generation",
                "timestamp": time.time()
            })
        
        return hypotheses
    
    def _extract_final_hypothesis(self, debate_response: str) -> str:
        """
        Extract the final hypothesis from a debate response.
        
        Args:
            debate_response: Debate response text
            
        Returns:
            Final hypothesis
        """
        # In a real implementation, this would be more sophisticated
        # Looking for a "HYPOTHESIS:" marker
        if "HYPOTHESIS:" in debate_response:
            parts = debate_response.split("HYPOTHESIS:")
            return parts[1].strip()
        else:
            # Fallback: return the last paragraph
            paragraphs = debate_response.split("\n\n")
            return paragraphs[-1].strip() 