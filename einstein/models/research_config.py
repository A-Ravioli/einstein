"""Research goal configuration models."""
from typing import List, Dict, Any

class ResearchGoalConfig:
    """Configuration parsed from a research goal."""
    
    def __init__(self, research_goal: str):
        """
        Initialize with a research goal.
        
        Args:
            research_goal: Research goal text
        """
        self.original_goal = research_goal
        self.preferences = []
        self.attributes = []
        self.constraints = []
        self.parse_goal(research_goal)
    
    def parse_goal(self, goal: str):
        """
        Parse the research goal to extract preferences, attributes, and constraints.
        
        Args:
            goal: Research goal text
        """
        # Example parsing logic (simplified)
        if "novel" in goal.lower():
            self.attributes.append("Novelty")
        
        if "testable" in goal.lower() or "experiment" in goal.lower():
            self.attributes.append("Testability")
            
        if "mechanism" in goal.lower():
            self.preferences.append("Focus on mechanistic explanations")
            
        # Default constraints
        self.constraints.append("should be correct")
        self.constraints.append("should be novel")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "original_goal": self.original_goal,
            "preferences": self.preferences,
            "attributes": self.attributes,
            "constraints": self.constraints
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ResearchGoalConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            ResearchGoalConfig instance
        """
        instance = cls(config_dict.get("original_goal", ""))
        instance.preferences = config_dict.get("preferences", [])
        instance.attributes = config_dict.get("attributes", [])
        instance.constraints = config_dict.get("constraints", [])
        return instance 