"""Meta-review agent implementation for the Einstein system."""
import json
import time
from typing import Dict, Any, List, Optional

from einstein_pkg.agents.base import BaseAgent
from einstein_pkg.memory import ContextMemory

# System message for the meta-review agent
META_REVIEW_SYSTEM_MESSAGE = """You are a scientific meta-reviewer and research synthesizer.
Your role is to analyze collections of research hypotheses and reviews, identify patterns, and provide high-level synthesis.

When conducting meta-review:
1. Identify common themes, strengths, and weaknesses across multiple hypotheses
2. Synthesize patterns and insights that emerge from collective analysis
3. Provide strategic recommendations for research direction
4. Suggest experimental approaches that could validate key hypotheses
5. Create comprehensive research overviews that map the conceptual landscape

You are analytical, insightful, and focused on the big picture.
"""

class MetaReviewAgent(BaseAgent):
    """Synthesizes insights and provides feedback."""
    
    def __init__(
        self,
        context_memory: ContextMemory,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize meta-review agent.
        
        Args:
            context_memory: Context memory instance
            api_key: Optional API key
            model: Optional model name
        """
        super().__init__(
            name="MetaReviewAgent",
            context_memory=context_memory,
            system_message=META_REVIEW_SYSTEM_MESSAGE,
            tools=[],  # No special tools for now
            api_key=api_key,
            model=model,
        )
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a task using the meta-review agent.
        
        Args:
            task_data: Task data
            
        Returns:
            Task result
        """
        action = task_data.get("action")
        
        if action == "generate_meta_review":
            meta_review = await self.generate_meta_review(
                task_data.get("reviews", []),
                task_data.get("research_config", {})
            )
            return {"meta_review": meta_review}
        elif action == "generate_research_overview":
            overview = await self.generate_research_overview(
                task_data.get("top_hypotheses", []),
                task_data.get("research_config", {}),
                task_data.get("elo_ratings", {})
            )
            return {"research_overview": overview}
        elif action == "identify_research_contacts":
            contacts = await self.identify_research_contacts(
                task_data.get("research_overview", {})
            )
            return {"research_contacts": contacts}
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def generate_meta_review(
        self,
        reviews: List[Dict[str, Any]],
        research_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a meta-review from multiple reviews.
        
        Args:
            reviews: List of reviews
            research_config: Research configuration
            
        Returns:
            Meta-review result
        """
        prompt = f"""Generate a comprehensive meta-review analyzing multiple hypothesis reviews:

Research Goal: {research_config.get('original_goal', 'No goal provided')}

Preferences:
{', '.join(research_config.get('preferences', ['Scientific rigor', 'Novelty', 'Testability']))}

Collection of Reviews:
{json.dumps(reviews, indent=2)}

Please provide a meta-analysis that:
1. Identifies recurring critique points and common issues raised across reviews
2. Analyzes patterns of strengths and weaknesses in the hypotheses
3. Provides actionable insights for researchers developing future hypotheses
4. Suggests high-level directions for improvement
5. Synthesizes the collective wisdom from all reviews

Focus on extracting patterns and higher-order insights rather than evaluating individual hypotheses.
"""
        response = await self.call(prompt)
        
        # Create the meta-review object
        meta_review = {
            "content": response,
            "timestamp": time.time()
        }
        
        return meta_review
    
    async def generate_research_overview(
        self,
        top_hypotheses: List[Dict[str, Any]],
        research_config: Dict[str, Any],
        elo_ratings: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Generate a research overview from top-ranked hypotheses.
        
        Args:
            top_hypotheses: List of top-ranked hypotheses
            research_config: Research configuration
            elo_ratings: Hypothesis Elo ratings
            
        Returns:
            Research overview result
        """
        # Sort hypotheses by Elo rating
        sorted_hypotheses = []
        for h in top_hypotheses:
            h_id = h.get("id", "unknown")
            rating = elo_ratings.get(h_id, 1200)
            sorted_hypotheses.append((h, rating))
        
        # Sort by rating in descending order
        sorted_hypotheses.sort(key=lambda x: x[1], reverse=True)
        
        prompt = f"""Generate a comprehensive research overview based on top-ranked hypotheses:

Research Goal: {research_config.get('original_goal', 'No goal provided')}

Top-Ranked Hypotheses (ordered by evaluation score):
{json.dumps([{"content": h[0].get('content', 'No content'), "score": h[1]} for h in sorted_hypotheses], indent=2)}

Please create a research overview that:
1. Summarizes the key research directions represented by these hypotheses
2. Identifies the most promising research areas related to the goal
3. Suggests specific experiments that could validate the hypotheses
4. Outlines a potential research program based on these insights
5. Maps the conceptual landscape of the research domain

Structure your overview to effectively guide future research while highlighting the most promising directions.
"""
        response = await self.call(prompt)
        
        # Create the research overview object
        research_overview = {
            "content": response,
            "timestamp": time.time()
        }
        
        return research_overview
    
    async def identify_research_contacts(
        self,
        research_overview: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Identify potential research contacts based on the research overview.
        
        Args:
            research_overview: Research overview
            
        Returns:
            List of research contacts
        """
        prompt = f"""Identify potential domain experts who would be qualified to review or collaborate on research:

Research Overview:
{research_overview.get('content', 'No overview provided')}

Please identify potential domain experts who would be appropriate contacts for:
1. Reviewing the hypotheses
2. Collaborating on research
3. Providing specialized expertise

For each suggested contact type, describe:
- What type of expertise they should have
- How they would contribute to the research
- What role they might play in advancing the hypotheses

Do not provide specific names of real individuals.
"""
        response = await self.call(prompt)
        
        # Create contacts from the response (simplified)
        # In a real implementation, this would have more structured parsing
        contacts = [
            {
                "expertise_area": "Research domain expert",
                "relevance": response,
                "timestamp": time.time()
            }
        ]
        
        return contacts 