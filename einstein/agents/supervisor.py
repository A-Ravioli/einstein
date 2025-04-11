"""Supervisor agent implementation for the Einstein system."""
import json
from typing import Dict, Any, Optional

from einstein_pkg.agents.base import BaseAgent
from einstein_pkg.memory import ContextMemory
from einstein_pkg.tools import parse_research_goal_tool

# System message for the supervisor agent
SUPERVISOR_SYSTEM_MESSAGE = """You are a scientific research supervisor in charge of coordinating a team of specialized AI agents.
Your role is to:
1. Understand and parse research goals
2. Create and prioritize research tasks
3. Assign tasks to appropriate specialized agents
4. Analyze and synthesize results from the team
5. Make strategic decisions about research direction

You are methodical, organized, and focused on producing high-quality scientific hypotheses.
"""

class SupervisorAgent(BaseAgent):
    """Orchestrates other agents and manages the workflow."""
    
    def __init__(
        self,
        context_memory: ContextMemory,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize supervisor agent.
        
        Args:
            context_memory: Context memory instance
            api_key: Optional API key
            model: Optional model name
        """
        super().__init__(
            name="SupervisorAgent",
            context_memory=context_memory,
            system_message=SUPERVISOR_SYSTEM_MESSAGE,
            tools=[parse_research_goal_tool],
            api_key=api_key,
            model=model,
        )
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a task using the supervisor agent.
        
        Args:
            task_data: Task data
            
        Returns:
            Task result
        """
        action = task_data.get("action")
        
        if action == "parse_research_goal":
            return await self.parse_research_goal(task_data.get("research_goal", ""))
        elif action == "create_task_queue":
            return await self.create_task_queue(task_data.get("research_config", {}))
        elif action == "assign_agents":
            return await self.assign_agents(task_data.get("tasks", []))
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def parse_research_goal(self, research_goal: str) -> Dict[str, Any]:
        """
        Parse a research goal.
        
        Args:
            research_goal: Research goal text
            
        Returns:
            Research configuration
        """
        prompt = f"Parse this research goal into a structured configuration: {research_goal}"
        response = await self.call(prompt)
        
        # Try to extract JSON
        result_dict = self._extract_json(response)
        if result_dict:
            return {"research_config": result_dict}
        
        # Fallback: call the tool directly
        tool_result = await parse_research_goal_tool(research_goal)
        return {"research_config": json.loads(tool_result)}
    
    async def create_task_queue(self, research_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a queue of tasks.
        
        Args:
            research_config: Research configuration
            
        Returns:
            Task queue
        """
        prompt = f"""Based on this research configuration, create a prioritized queue of tasks:
        
Research Goal: {research_config.get('original_goal', 'No goal provided')}

Preferences: {', '.join(research_config.get('preferences', []))}
Attributes: {', '.join(research_config.get('attributes', []))}
Constraints: {', '.join(research_config.get('constraints', []))}

Return a JSON array of tasks, where each task has "agent", "action", and "params" fields.
"""
        response = await self.call(prompt)
        
        # Try to extract JSON
        result_dict = self._extract_json(response)
        if result_dict and isinstance(result_dict, list):
            return {"tasks": result_dict}
        
        # Fallback: create default task queue
        tasks = [
            {"agent": "GenerationAgent", "action": "generate_initial_hypotheses", "params": research_config}
        ]
        return {"tasks": tasks}
    
    async def assign_agents(self, tasks: list) -> Dict[str, Any]:
        """
        Assign tasks to agents.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Agent assignments
        """
        prompt = f"""Assign these tasks to specialized agents and allocate resources:
        
Tasks: {json.dumps(tasks, indent=2)}

Return a JSON object with "task_queue" (the list of tasks) and "resource_allocation" (a dictionary mapping agent names to resource percentages).
"""
        response = await self.call(prompt)
        
        # Try to extract JSON
        result_dict = self._extract_json(response)
        if result_dict:
            return result_dict
        
        # Fallback: create default assignment
        assignment = {
            "task_queue": tasks,
            "resource_allocation": {
                "GenerationAgent": 0.3,
                "ReflectionAgent": 0.2,
                "RankingAgent": 0.15,
                "EvolutionAgent": 0.2,
                "ProximityAgent": 0.05,
                "MetaReviewAgent": 0.1
            }
        }
        return assignment 