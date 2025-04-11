"""Research goal parsing tool for the Einstein system."""
import json

from einstein.models.research_config import ResearchGoalConfig

async def parse_research_goal_tool(research_goal: str) -> str:
    """
    Parse a research goal into a configuration.
    
    Args:
        research_goal: Research goal text
        
    Returns:
        JSON string with research configuration
    """
    config = ResearchGoalConfig(research_goal)
    return json.dumps(config.to_dict()) 