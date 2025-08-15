from typing import Dict, List, Optional, Any
from .base_agent import BaseAgent


class WritingAgent(BaseAgent):
    """Agent for assisting with scientific writing and documentation"""
    
    def __init__(self, openai_client, settings):
        super().__init__(openai_client, settings, "WritingAgent")
    
    async def _setup(self):
        """Setup writing parameters"""
        self.content_types = [
            "abstract", "introduction", "methods", "results", 
            "discussion", "conclusion", "grant_proposal", "research_summary"
        ]
    
    async def assist_writing(self, content_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assist with scientific writing"""
        
        writing_prompt = [
            {
                "role": "system",
                "content": self.create_system_prompt(
                    "a scientific writing expert",
                    [
                        "Write clear, accurate scientific content",
                        "Follow academic writing conventions",
                        "Maintain scientific rigor and objectivity",
                        "Adapt style to the target audience"
                    ]
                )
            },
            {
                "role": "user",
                "content": f"""Help write a {content_type} based on this research context:

{str(context)}

Requirements:
- Scientific accuracy and clarity
- Appropriate academic tone
- Proper structure for {content_type}
- Cite relevant information from context

Provide the written content and any writing tips."""
            }
        ]
        
        try:
            response = await self.generate_structured_completion(writing_prompt, schema={})
            return {
                'content_type': content_type,
                'written_content': response.get('content', ''),
                'writing_tips': response.get('tips', []),
                'word_count': len(response.get('content', '').split()),
                'suggestions': response.get('suggestions', [])
            }
        except Exception as e:
            self.logger.error("Writing assistance failed", error=str(e))
            return {
                'content_type': content_type,
                'error': f'Writing assistance failed: {str(e)}'
            } 