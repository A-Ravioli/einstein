from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import structlog
from openai import AsyncOpenAI

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Base class for all AI agents in the co-scientist platform.
    Provides common functionality and interface.
    """
    
    def __init__(self, openai_client: AsyncOpenAI, settings: Any, agent_name: str):
        self.openai_client = openai_client
        self.settings = settings
        self.agent_name = agent_name
        self.logger = logger.bind(agent=agent_name)
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent"""
        self.logger.info(f"Initializing {self.agent_name} agent")
        await self._setup()
        self.initialized = True
        self.logger.info(f"{self.agent_name} agent initialized successfully")
    
    @abstractmethod
    async def _setup(self):
        """Agent-specific setup logic"""
        pass
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate completion using OpenAI"""
        try:
            response = await self.openai_client.chat.completions.create(
                model=model or self.settings.OPENAI_MODEL,
                messages=messages,
                temperature=temperature or self.settings.OPENAI_TEMPERATURE,
                max_tokens=max_tokens or self.settings.OPENAI_MAX_TOKENS,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error("Failed to generate completion", error=str(e))
            raise
    
    async def generate_structured_completion(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured completion with JSON schema"""
        try:
            response = await self.openai_client.chat.completions.create(
                model=model or self.settings.OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error("Failed to generate structured completion", error=str(e))
            raise
    
    def create_system_prompt(self, role_description: str, guidelines: List[str]) -> str:
        """Create a standardized system prompt"""
        prompt = f"""You are {role_description}.

Core Guidelines:
{chr(10).join(f"- {guideline}" for guideline in guidelines)}

Always respond with accurate, evidence-based information. When uncertain, clearly state your limitations.
Focus on scientific rigor and practical applicability in your responses."""
        
        return prompt
    
    async def cleanup(self):
        """Cleanup agent resources"""
        self.logger.info(f"Cleaning up {self.agent_name} agent")
        self.initialized = False 