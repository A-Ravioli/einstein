"""Base agent implementation for the Einstein system."""
import re
import os
import json
from typing import Dict, Any, List, Optional, Callable, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_core import CancellationToken

from einstein_pkg.memory import ContextMemory
from einstein_pkg.config import get_config

class BaseAgent:
    """Base class for all specialized agents."""
    
    def __init__(
        self,
        name: str,
        context_memory: ContextMemory,
        system_message: str,
        tools: Optional[List[Callable]] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            context_memory: Context memory instance
            system_message: System message for the agent
            tools: Optional list of tools
            api_key: Optional API key (if None, use from config)
            model: Optional model name (if None, use from config)
        """
        self.name = name
        self.context_memory = context_memory
        
        # Set up the API key and model
        config = get_config()
        if api_key is None:
            api_key = config.openai_api_key
        
        if model is None:
            model = config.default_model
        
        # Create model client
        self.model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key
        )
        
        # Create the agent
        self.agent = AssistantAgent(
            name=name,
            system_message=system_message,
            model_client=self.model_client,
            tools=tools or []
        )
    
    async def call(self, prompt: str, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Call the agent with a prompt.
        
        Args:
            prompt: Prompt text
            additional_context: Optional additional context
            
        Returns:
            Agent response text
        """
        # Prepare the message
        if additional_context:
            prompt = f"{prompt}\n\nAdditional context: {json.dumps(additional_context)}"
        
        # Run the agent with the prompt
        message = TextMessage(content=prompt, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Return the response text
        return response.chat_message.content
    
    async def run(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a task using the agent.
        
        Args:
            task_data: Task data
            
        Returns:
            Task result
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement run()")
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON as dictionary or None if not found
        """
        try:
            # Look for JSON-like content in the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group(0))
                return result_dict
            return None
        except:
            return None 