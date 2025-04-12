"""Ranking agent implementation for the Einstein system."""
import json
import time
from typing import Dict, Any, List, Optional

from einstein.agents.base import BaseAgent
from einstein.memory import ContextMemory

# System message for the ranking agent
RANKING_SYSTEM_MESSAGE = """You are a scientific hypothesis evaluator.
Your role is to compare and rank competing scientific hypotheses based on merit.

When comparing hypotheses:
1. Assess each hypothesis against the research goal and evaluation criteria
2. Conduct a rigorous comparison considering strengths and weaknesses of each
3. Make judgments based on scientific merit, not personal preference
4. Provide detailed reasoning for your comparative evaluation
5. Clearly indicate which hypothesis is superior (or if they are equal)

You are objective, analytical, and focused on scientific quality.
"""

class RankingAgent(BaseAgent):
    """Evaluates and compares hypotheses using an Elo-based tournament."""
    
    def __init__(
        self,
        context_memory: ContextMemory,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        initial_elo: int = 1200,
    ):
        """
        Initialize ranking agent.
        
        Args:
            context_memory: Context memory instance
            api_key: Optional API key
            model: Optional model name
            initial_elo: Initial Elo rating for new hypotheses
        """
        super().__init__(
            name="RankingAgent",
            context_memory=context_memory,
            system_message=RANKING_SYSTEM_MESSAGE,
            tools=[],  # No special tools for now
            api_key=api_key,
            model=model,
        )
        self.initial_elo = initial_elo
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a task using the ranking agent.
        
        Args:
            task_data: Task data
            
        Returns:
            Task result
        """
        action = task_data.get("action")
        
        if action == "scientific_debate":
            result = await self.scientific_debate(
                task_data.get("hypothesis1", {}),
                task_data.get("hypothesis2", {}),
                task_data.get("research_config", {}),
                task_data.get("reviews", {})
            )
            return {"debate_result": result}
        elif action == "update_elo_ratings":
            updated_ratings = await self.update_elo_ratings(
                task_data.get("match_result", {}),
                task_data.get("elo_ratings", {})
            )
            return {"updated_elo_ratings": updated_ratings}
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def scientific_debate(
        self,
        hypothesis1: Dict[str, Any],
        hypothesis2: Dict[str, Any],
        research_config: Dict[str, Any],
        reviews: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Conduct a simulated scientific debate between two hypotheses.
        
        Args:
            hypothesis1: First hypothesis
            hypothesis2: Second hypothesis
            research_config: Research configuration
            reviews: Hypothesis reviews
            
        Returns:
            Debate result
        """
        # Get reviews for both hypotheses if available
        review1 = reviews.get(hypothesis1.get("id", ""), {})
        review2 = reviews.get(hypothesis2.get("id", ""), {})
        
        prompt = f"""Conduct a simulated scientific debate to compare these two hypotheses:

Research Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for evaluation:
{', '.join(research_config.get('preferences', ['Scientific merit', 'Testability', 'Novelty']))}

Hypothesis 1:
{hypothesis1.get('content', 'No content provided')}

Hypothesis 2:
{hypothesis2.get('content', 'No content provided')}

Previous review of Hypothesis 1:
{json.dumps(review1, indent=2) if review1 else "No prior review available"}

Previous review of Hypothesis 2:
{json.dumps(review2, indent=2) if review2 else "No prior review available"}

Simulate a detailed scientific debate between experts evaluating these hypotheses. Consider:
1. Logical consistency and scientific accuracy of each hypothesis
2. Evidence or theoretical support for each
3. Testability and practical implementation
4. Novelty and potential scientific impact
5. Alignment with the research goal

After thorough discussion, conclude with "Better Hypothesis: 1" or "Better Hypothesis: 2".
If they are equal in merit, conclude with "Equal Merit: Neither is clearly superior".
"""
        response = await self.call(prompt)
        
        # Determine the winner
        if "Better Hypothesis: 1" in response:
            winner = 1
        elif "Better Hypothesis: 2" in response:
            winner = 2
        else:
            # If no clear winner is determined, consider it a tie
            winner = 0
        
        return {
            "hypothesis1_id": hypothesis1.get("id", "unknown"),
            "hypothesis2_id": hypothesis2.get("id", "unknown"),
            "winner": winner,
            "debate": response,
            "timestamp": time.time()
        }
    
    async def update_elo_ratings(self, match_result: Dict[str, Any], elo_ratings: Dict[str, float]) -> Dict[str, float]:
        """
        Update Elo ratings based on match results.
        
        Args:
            match_result: Match result (including winner)
            elo_ratings: Current Elo ratings
            
        Returns:
            Updated Elo ratings
        """
        h1_id = match_result.get("hypothesis1_id", "unknown")
        h2_id = match_result.get("hypothesis2_id", "unknown")
        
        # Get current ratings (or use initial rating if not available)
        rating1 = elo_ratings.get(h1_id, self.initial_elo)
        rating2 = elo_ratings.get(h2_id, self.initial_elo)
        
        # Calculate expected scores
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
        
        # Calculate actual scores
        if match_result.get("winner") == 1:
            actual1 = 1.0
            actual2 = 0.0
        elif match_result.get("winner") == 2:
            actual1 = 0.0
            actual2 = 1.0
        else:  # Tie
            actual1 = 0.5
            actual2 = 0.5
        
        # Update ratings (K factor of 32 is standard)
        k_factor = 32
        new_rating1 = rating1 + k_factor * (actual1 - expected1)
        new_rating2 = rating2 + k_factor * (actual2 - expected2)
        
        # Update the ratings dictionary
        updated_ratings = elo_ratings.copy()
        updated_ratings[h1_id] = new_rating1
        updated_ratings[h2_id] = new_rating2
        
        return updated_ratings 