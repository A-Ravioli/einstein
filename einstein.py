import os
import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

# Import AutoGen components
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_core import CancellationToken

# For context memory persistence
class ContextMemory:
    """Persistent storage for maintaining state across operations."""
    
    def __init__(self, storage_path: str = "context_memory.json"):
        self.storage_path = storage_path
        self.memory = self._load_memory() if os.path.exists(storage_path) else {}
    
    def _load_memory(self) -> Dict:
        """Load memory from disk."""
        with open(self.storage_path, 'r') as f:
            return json.load(f)
    
    def _save_memory(self):
        """Save memory to disk."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f)
    
    def set(self, key: str, value: Any):
        """Store value in memory."""
        self.memory[key] = value
        self._save_memory()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve value from memory."""
        return self.memory.get(key, default)
    
    def update_state(self, state_updates: Dict):
        """Update multiple state values at once."""
        self.memory.update(state_updates)
        self._save_memory()

class ResearchGoalConfig:
    """Configuration parsed from a research goal."""
    
    def __init__(self, research_goal: str):
        self.original_goal = research_goal
        self.preferences = []
        self.attributes = []
        self.constraints = []
        self.parse_goal(research_goal)
    
    def parse_goal(self, goal: str):
        """Parse the research goal to extract preferences, attributes, and constraints."""
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
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "original_goal": self.original_goal,
            "preferences": self.preferences,
            "attributes": self.attributes,
            "constraints": self.constraints
        } 

# Define custom tools for the agents
async def search_literature_tool(query: str) -> str:
    """Search for relevant literature using web search API."""
    # Simulated response for demonstration
    literature = [
        {"title": "Recent advances in the field", "content": "Summary of article..."},
        {"title": "Key findings from 2024", "content": "Summary of article..."}
    ]
    return json.dumps(literature)

async def parse_research_goal_tool(research_goal: str) -> str:
    """Parse a research goal into a configuration."""
    config = ResearchGoalConfig(research_goal)
    return json.dumps(config.to_dict())

# Agent implementations
class SupervisorAgent:
    """Orchestrates other agents and manages the workflow."""
    
    def __init__(self, context_memory: ContextMemory, api_key: str = None):
        self.context_memory = context_memory
        
        # Set up the API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # Create model client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4-turbo",
            api_key=api_key
        )
        
        # Define the system message for the supervisor
        system_message = """You are a scientific research supervisor in charge of coordinating a team of specialized AI agents.
Your role is to:
1. Understand and parse research goals
2. Create and prioritize research tasks
3. Assign tasks to appropriate specialized agents
4. Analyze and synthesize results from the team
5. Make strategic decisions about research direction

You are methodical, organized, and focused on producing high-quality scientific hypotheses.
"""
        
        # Create the assistant agent
        self.agent = AssistantAgent(
            name="SupervisorAgent",
            system_message=system_message,
            model_client=self.model_client,
            tools=[parse_research_goal_tool]
        )
    
    async def run(self, task: str, additional_context: Dict = None) -> Dict:
        """Run a task using the supervisor agent."""
        # Prepare the message
        if additional_context:
            task = f"{task}\n\nAdditional context: {json.dumps(additional_context)}"
        
        # Run the agent with the task
        message = TextMessage(content=task, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Process the response
        result_content = response.chat_message.content
        
        # Try to extract JSON if present
        try:
            # Look for JSON-like content in the response
            import re
            json_match = re.search(r'\{.*\}', result_content, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group(0))
                return result_dict
            else:
                return {"response": result_content}
        except:
            # If JSON extraction fails, return the raw text
            return {"response": result_content} 

class GenerationAgent:
    """Generates initial research hypotheses."""
    
    def __init__(self, context_memory: ContextMemory, api_key: str = None):
        self.context_memory = context_memory
        
        # Set up the API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # Create model client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4-turbo",
            api_key=api_key
        )
        
        # Define the system message for the generation agent
        system_message = """You are a scientific hypothesis generation expert.
Your role is to generate novel, testable scientific hypotheses based on research goals and literature.

When generating hypotheses:
1. Ensure they are specific, testable, and address the research goal
2. Base them on sound scientific principles and relevant literature
3. Format each hypothesis clearly, starting with "Hypothesis X:"
4. Provide sufficient detail and rationale for each hypothesis
5. Ensure hypotheses are distinct from each other

You are creative but scientifically rigorous.
"""
        
        # Create the assistant agent
        self.agent = AssistantAgent(
            name="GenerationAgent",
            system_message=system_message,
            model_client=self.model_client,
            tools=[search_literature_tool]
        )
    
    async def generate_initial_hypotheses(self, research_config: Dict) -> List[Dict]:
        """Generate initial research hypotheses."""
        # Create prompt
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
        
        # Run the agent with the prompt
        message = TextMessage(content=prompt, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Extract hypotheses from the response
        response_text = response.chat_message.content
        hypotheses = self._parse_hypotheses(response_text)
        
        return hypotheses
    
    def _parse_hypotheses(self, response_text: str) -> List[Dict]:
        """Parse hypotheses from the agent response."""
        hypotheses = []
        
        # Simple splitting by "Hypothesis" marker
        parts = response_text.split("Hypothesis")
        for i, part in enumerate(parts[1:], 1):  # Skip the first part (before "Hypothesis 1")
            # Extract until the next hypothesis or end of text
            content = part.strip()
            
            hypotheses.append({
                "id": f"h{i}",
                "content": content,
                "source": "generation",
                "timestamp": time.time()
            })
        
        return hypotheses

    async def run(self, task_data: Dict) -> Dict:
        """Run the appropriate task based on the task data."""
        action = task_data.get("action")
        
        if action == "generate_initial_hypotheses":
            hypotheses = await self.generate_initial_hypotheses(task_data.get("params", {}))
            return {"hypotheses": hypotheses}
        else:
            return {"error": f"Unknown action: {action}"}


class ReflectionAgent:
    """Reviews and critiques hypotheses."""
    
    def __init__(self, context_memory: ContextMemory, api_key: str = None):
        self.context_memory = context_memory
        
        # Set up the API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # Create model client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4-turbo",
            api_key=api_key
        )
        
        # Define the system message for the reflection agent
        system_message = """You are a scientific hypothesis reviewer and critic.
Your role is to evaluate scientific hypotheses for correctness, quality, novelty, and ethical implications.

When reviewing:
1. Assess the hypothesis based on scientific principles, logic, and evidence
2. Provide specific scores (1-5) for each evaluation dimension
3. Justify your ratings with specific critiques
4. Identify potential flaws, gaps, or inconsistencies
5. Suggest potential improvements

You are thorough, critical, and constructive.
"""
        
        # Create the assistant agent
        self.agent = AssistantAgent(
            name="ReflectionAgent",
            system_message=system_message,
            model_client=self.model_client,
            tools=[]  # No special tools for now
        )
    
    async def initial_review(self, hypothesis: Dict, research_config: Dict) -> Dict:
        """Perform an initial review of a hypothesis."""
        # Create prompt
        prompt = f"""Evaluate the following scientific hypothesis:

Goal: {research_config.get('original_goal', 'No goal provided')}

Criteria for evaluation:
{', '.join(research_config.get('preferences', ['Correctness', 'Quality', 'Novelty', 'Safety']))}

Hypothesis to review:
{hypothesis.get('content', 'No hypothesis provided')}

Please evaluate this hypothesis on the following dimensions:
1. Correctness (1-5): Evaluate scientific accuracy and logical consistency
2. Quality (1-5): Assess clarity, specificity, and testability
3. Novelty (1-5): Evaluate originality and potential for new insights
4. Safety/Ethics (1-5): Consider ethical implications or potential misuse

For each dimension, provide:
- A numerical score (1-5)
- A detailed explanation of your rating
- Specific strengths and weaknesses
"""
        
        # Run the agent with the prompt
        message = TextMessage(content=prompt, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Extract review from the response
        response_text = response.chat_message.content
        review = self._parse_review(response_text)
        
        # Add metadata
        review["hypothesis_id"] = hypothesis["id"]
        review["type"] = "initial"
        review["timestamp"] = time.time()
        
        return review
    
    def _parse_review(self, review_text: str) -> Dict:
        """Parse a review from the agent response."""
        # For simplicity, we're using basic pattern matching
        # In a production system, you would want more robust parsing
        scores = {}
        
        try:
            # Extract scores using basic pattern matching
            import re
            
            # Look for patterns like "Correctness (1-5): 4" or "Correctness: 4/5"
            correctness_match = re.search(r'Correctness[^0-9]*([1-5])', review_text)
            quality_match = re.search(r'Quality[^0-9]*([1-5])', review_text)
            novelty_match = re.search(r'Novelty[^0-9]*([1-5])', review_text)
            safety_match = re.search(r'Safety[^0-9]*([1-5])|Ethics[^0-9]*([1-5])', review_text)
            
            if correctness_match:
                scores["correctness"] = int(correctness_match.group(1))
            if quality_match:
                scores["quality"] = int(quality_match.group(1))
            if novelty_match:
                scores["novelty"] = int(novelty_match.group(1))
            if safety_match:
                # Use the first capturing group that matched
                score = safety_match.group(1) or safety_match.group(2)
                scores["safety"] = int(score)
        except:
            # Default scores if parsing fails
            scores = {
                "correctness": 3,
                "quality": 3, 
                "novelty": 3,
                "safety": 3
            }
        
        return {
            "scores": scores,
            "content": review_text
        }
    
    async def run(self, task_data: Dict) -> Dict:
        """Run the appropriate task based on the task data."""
        action = task_data.get("action")
        
        if action == "initial_review":
            review = await self.initial_review(
                task_data.get("hypothesis", {}),
                task_data.get("research_config", {})
            )
            return {"review": review}
        else:
            return {"error": f"Unknown action: {action}"} 

class RankingAgent:
    """Evaluates and compares hypotheses using an Elo-based tournament."""
    
    def __init__(self, context_memory: ContextMemory, api_key: str = None):
        self.context_memory = context_memory
        self.initial_elo = 1200  # Initial Elo rating for new hypotheses
        
        # Set up the API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # Create model client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4-turbo",
            api_key=api_key
        )
        
        # Define the system message for the ranking agent
        system_message = """You are a scientific hypothesis evaluator.
Your role is to compare and rank competing scientific hypotheses based on merit.

When comparing hypotheses:
1. Assess each hypothesis against the research goal and evaluation criteria
2. Conduct a rigorous comparison considering strengths and weaknesses of each
3. Make judgments based on scientific merit, not personal preference
4. Provide detailed reasoning for your comparative evaluation
5. Clearly indicate which hypothesis is superior (or if they are equal)

You are objective, analytical, and focused on scientific quality.
"""
        
        # Create the assistant agent
        self.agent = AssistantAgent(
            name="RankingAgent",
            system_message=system_message,
            model_client=self.model_client,
            tools=[]  # No special tools for now
        )
    
    async def scientific_debate(self, hypothesis1: Dict, hypothesis2: Dict, 
                               research_config: Dict, reviews: Dict) -> Dict:
        """Conduct a simulated scientific debate between two hypotheses."""
        # Get reviews for both hypotheses if available
        review1 = reviews.get(hypothesis1.get("id", ""), {})
        review2 = reviews.get(hypothesis2.get("id", ""), {})
        
        # Create prompt
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
        
        # Run the agent with the prompt
        message = TextMessage(content=prompt, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Extract result from the response
        response_text = response.chat_message.content
        
        # Determine the winner
        if "Better Hypothesis: 1" in response_text:
            winner = 1
        elif "Better Hypothesis: 2" in response_text:
            winner = 2
        else:
            # If no clear winner is determined, consider it a tie
            winner = 0
        
        return {
            "hypothesis1_id": hypothesis1.get("id"),
            "hypothesis2_id": hypothesis2.get("id"),
            "winner": winner,
            "debate": response_text,
            "timestamp": time.time()
        }
    
    async def update_elo_ratings(self, match_result: Dict, elo_ratings: Dict) -> Dict:
        """Update Elo ratings based on match results."""
        h1_id = match_result.get("hypothesis1_id")
        h2_id = match_result.get("hypothesis2_id")
        
        # Get current ratings (or use initial rating if not available)
        rating1 = elo_ratings.get(h1_id, self.initial_elo)
        rating2 = elo_ratings.get(h2_id, self.initial_elo)
        
        # Calculate expected scores
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
        
        # Calculate actual scores
        if match_result.get("winner") == 1:
            actual1 = 1
            actual2 = 0
        elif match_result.get("winner") == 2:
            actual1 = 0
            actual2 = 1
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
    
    async def run(self, task_data: Dict) -> Dict:
        """Run the appropriate task based on the task data."""
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
            return {"error": f"Unknown action: {action}"}


class EvolutionAgent:
    """Improves existing hypotheses."""
    
    def __init__(self, context_memory: ContextMemory, api_key: str = None):
        self.context_memory = context_memory
        
        # Set up the API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # Create model client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4-turbo",
            api_key=api_key
        )
        
        # Define the system message for the evolution agent
        system_message = """You are a scientific hypothesis improvement specialist.
Your role is to refine and evolve scientific hypotheses based on feedback and evaluation.

When improving hypotheses:
1. Address specific critiques and limitations identified in reviews
2. Maintain the core scientific insights of the original hypothesis
3. Enhance clarity, testability, and scientific rigor
4. Consider novel perspectives or mechanisms that strengthen the hypothesis
5. Provide a complete, revised hypothesis that builds upon the original

You are innovative but grounded in scientific principles.
"""
        
        # Create the assistant agent
        self.agent = AssistantAgent(
            name="EvolutionAgent",
            system_message=system_message,
            model_client=self.model_client,
            tools=[]  # No special tools for now
        )
    
    async def improve_hypothesis(self, hypothesis: Dict, research_config: Dict, reviews: Dict) -> Dict:
        """Improve a hypothesis based on reviews and research configuration."""
        # Get reviews for the hypothesis if available
        hypothesis_review = reviews.get(hypothesis.get("id", ""), {})
        
        # Create prompt
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
        
        # Run the agent with the prompt
        message = TextMessage(content=prompt, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Extract improved hypothesis from the response
        improved_content = response.chat_message.content
        
        # Create the improved hypothesis object
        improved_hypothesis = {
            "id": f"{hypothesis.get('id', 'unknown')}_improved",
            "content": improved_content,
            "parent_id": hypothesis.get("id"),
            "source": "evolution",
            "timestamp": time.time()
        }
        
        return improved_hypothesis
    
    async def combine_hypotheses(self, hypotheses: List[Dict], research_config: Dict) -> Dict:
        """Combine multiple hypotheses to create a new, improved hypothesis."""
        # Extract content from hypotheses
        hypothesis_contents = [h.get('content', 'No content provided') for h in hypotheses]
        
        # Create prompt
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
        
        # Run the agent with the prompt
        message = TextMessage(content=prompt, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Extract combined hypothesis from the response
        combined_content = response.chat_message.content
        
        # Get IDs of parent hypotheses
        parent_ids = [h.get("id", f"unknown_{i}") for i, h in enumerate(hypotheses)]
        
        # Create the combined hypothesis object
        combined_hypothesis = {
            "id": f"combined_{'_'.join(parent_ids)}",
            "content": combined_content,
            "parent_ids": parent_ids,
            "source": "combination",
            "timestamp": time.time()
        }
        
        return combined_hypothesis
    
    async def run(self, task_data: Dict) -> Dict:
        """Run the appropriate task based on the task data."""
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
        else:
            return {"error": f"Unknown action: {action}"} 

class MetaReviewAgent:
    """Synthesizes insights and provides feedback."""
    
    def __init__(self, context_memory: ContextMemory, api_key: str = None):
        self.context_memory = context_memory
        
        # Set up the API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # Create model client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4-turbo",
            api_key=api_key
        )
        
        # Define the system message for the meta-review agent
        system_message = """You are a scientific meta-reviewer and research synthesizer.
Your role is to analyze collections of research hypotheses and reviews, identify patterns, and provide high-level synthesis.

When conducting meta-review:
1. Identify common themes, strengths, and weaknesses across multiple hypotheses
2. Synthesize patterns and insights that emerge from collective analysis
3. Provide strategic recommendations for research direction
4. Suggest experimental approaches that could validate key hypotheses
5. Create comprehensive research overviews that map the conceptual landscape

You are analytical, insightful, and focused on the big picture.
"""
        
        # Create the assistant agent
        self.agent = AssistantAgent(
            name="MetaReviewAgent",
            system_message=system_message,
            model_client=self.model_client,
            tools=[]  # No special tools for now
        )
    
    async def generate_meta_review(self, reviews: List[Dict], research_config: Dict) -> Dict:
        """Generate a meta-review from multiple reviews."""
        # Create prompt
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
        
        # Run the agent with the prompt
        message = TextMessage(content=prompt, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Extract meta-review from the response
        meta_review_content = response.chat_message.content
        
        # Create the meta-review object
        meta_review = {
            "content": meta_review_content,
            "timestamp": time.time()
        }
        
        return meta_review
    
    async def generate_research_overview(self, top_hypotheses: List[Dict], 
                                       research_config: Dict, elo_ratings: Dict) -> Dict:
        """Generate a research overview from top-ranked hypotheses."""
        # Sort hypotheses by Elo rating
        sorted_hypotheses = []
        for h in top_hypotheses:
            h_id = h.get("id", "unknown")
            rating = elo_ratings.get(h_id, 1200)
            sorted_hypotheses.append((h, rating))
        
        # Sort by rating in descending order
        sorted_hypotheses.sort(key=lambda x: x[1], reverse=True)
        
        # Create prompt
        prompt = f"""Generate a comprehensive research overview based on top-ranked hypotheses:

Research Goal: {research_config.get('original_goal', 'No goal provided')}

Top-Ranked Hypotheses (ordered by evaluation score):
{json.dumps([{"content": h[0].get('content'), "score": h[1]} for h in sorted_hypotheses], indent=2)}

Please create a research overview that:
1. Summarizes the key research directions represented by these hypotheses
2. Identifies the most promising research areas related to the goal
3. Suggests specific experiments that could validate the hypotheses
4. Outlines a potential research program based on these insights
5. Maps the conceptual landscape of the research domain

Structure your overview to effectively guide future research while highlighting the most promising directions.
"""
        
        # Run the agent with the prompt
        message = TextMessage(content=prompt, source="user")
        response = await self.agent.on_messages([message], CancellationToken())
        
        # Extract research overview from the response
        overview_content = response.chat_message.content
        
        # Create the research overview object
        research_overview = {
            "content": overview_content,
            "timestamp": time.time()
        }
        
        return research_overview
    
    async def run(self, task_data: Dict) -> Dict:
        """Run the appropriate task based on the task data."""
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
        else:
            return {"error": f"Unknown action: {action}"}


class AICoScientist:
    """Main class for the AI Co-scientist system using AutoGen."""
    
    def __init__(self, api_key: str = None):
        """Initialize the AI Co-scientist system."""
        # Initialize context memory
        self.context_memory = ContextMemory()
        
        # Initialize agents
        self.supervisor = SupervisorAgent(self.context_memory, api_key)
        self.generation_agent = GenerationAgent(self.context_memory, api_key)
        self.reflection_agent = ReflectionAgent(self.context_memory, api_key)
        self.ranking_agent = RankingAgent(self.context_memory, api_key)
        self.evolution_agent = EvolutionAgent(self.context_memory, api_key)
        self.meta_review_agent = MetaReviewAgent(self.context_memory, api_key)
        
        # Store results
        self.results = {}
    
    async def process_research_goal(self, research_goal: str):
        """Process a research goal and generate hypotheses."""
        print("Starting research process...")
        
        # Step 1: Parse the research goal
        print("Parsing research goal...")
        research_config = await self._parse_research_goal(research_goal)
        
        # Step 2: Generate initial hypotheses
        print("Generating initial hypotheses...")
        hypotheses = await self._generate_initial_hypotheses(research_config)
        
        # Step 3: Review hypotheses
        print("Reviewing hypotheses...")
        reviews = await self._review_hypotheses(hypotheses, research_config)
        
        # Step 4: Conduct tournament
        print("Conducting hypothesis tournament...")
        elo_ratings = await self._conduct_tournament(hypotheses, research_config, reviews)
        
        # Step 5: Select and improve top hypotheses
        print("Improving top hypotheses...")
        improved_hypotheses = await self._improve_top_hypotheses(hypotheses, elo_ratings, research_config, reviews)
        
        # Step 6: Combine improved hypotheses
        print("Combining hypotheses...")
        combined_hypothesis = await self._combine_hypotheses(improved_hypotheses, research_config)
        
        # Step 7: Generate research overview
        print("Generating research overview...")
        all_hypotheses = hypotheses + improved_hypotheses + [combined_hypothesis]
        overview = await self._generate_research_overview(all_hypotheses, research_config, elo_ratings)
        
        # Store results
        self.results = {
            "research_config": research_config,
            "hypotheses": hypotheses,
            "reviews": reviews,
            "elo_ratings": elo_ratings,
            "improved_hypotheses": improved_hypotheses,
            "combined_hypothesis": combined_hypothesis,
            "research_overview": overview
        }
        
        print("Research process complete!")
        return self.results
    
    async def _parse_research_goal(self, research_goal: str) -> Dict:
        """Parse the research goal into a configuration."""
        task_data = {
            "action": "parse_research_goal",
            "research_goal": research_goal
        }
        
        try:
            # Call the supervisor agent to parse the goal
            result = await self.supervisor.run(f"Parse this research goal: {research_goal}")
            
            # Check if we have a structured response
            if isinstance(result, dict) and "response" in result:
                # Try to extract a research config
                config = ResearchGoalConfig(research_goal)
                research_config = config.to_dict()
            else:
                # Use the result directly
                research_config = result
        except Exception as e:
            print(f"Error parsing research goal: {e}")
            # Fallback to simple parsing
            config = ResearchGoalConfig(research_goal)
            research_config = config.to_dict()
        
        # Store in context memory
        self.context_memory.set("research_config", research_config)
        return research_config
    
    async def _generate_initial_hypotheses(self, research_config: Dict) -> List[Dict]:
        """Generate initial hypotheses for the research goal."""
        task_data = {
            "action": "generate_initial_hypotheses",
            "params": research_config
        }
        
        result = await self.generation_agent.run(task_data)
        hypotheses = result.get("hypotheses", [])
        
        # Store in context memory
        self.context_memory.set("hypotheses", hypotheses)
        return hypotheses
    
    async def _review_hypotheses(self, hypotheses: List[Dict], research_config: Dict) -> Dict:
        """Review all hypotheses."""
        reviews = {}
        
        for hypothesis in hypotheses:
            task_data = {
                "action": "initial_review",
                "hypothesis": hypothesis,
                "research_config": research_config
            }
            
            result = await self.reflection_agent.run(task_data)
            review = result.get("review", {})
            reviews[hypothesis["id"]] = review
        
        # Store in context memory
        self.context_memory.set("reviews", reviews)
        return reviews
    
    async def _conduct_tournament(self, hypotheses: List[Dict], research_config: Dict, reviews: Dict) -> Dict:
        """Conduct a tournament to rank hypotheses."""
        elo_ratings = {h["id"]: 1200 for h in hypotheses}  # Initial ratings
        
        # Conduct pairwise comparisons
        for i in range(len(hypotheses)):
            for j in range(i+1, len(hypotheses)):
                task_data = {
                    "action": "scientific_debate",
                    "hypothesis1": hypotheses[i],
                    "hypothesis2": hypotheses[j],
                    "research_config": research_config,
                    "reviews": reviews
                }
                
                result = await self.ranking_agent.run(task_data)
                debate_result = result.get("debate_result", {})
                
                # Update Elo ratings
                task_data = {
                    "action": "update_elo_ratings",
                    "match_result": debate_result,
                    "elo_ratings": elo_ratings
                }
                
                result = await self.ranking_agent.run(task_data)
                elo_ratings = result.get("updated_elo_ratings", elo_ratings)
        
        # Store in context memory
        self.context_memory.set("elo_ratings", elo_ratings)
        return elo_ratings
    
    async def _improve_top_hypotheses(self, hypotheses: List[Dict], elo_ratings: Dict, 
                                    research_config: Dict, reviews: Dict) -> List[Dict]:
        """Improve the top-ranked hypotheses."""
        # Sort hypotheses by Elo rating
        sorted_hypotheses = sorted(
            [(h, elo_ratings.get(h["id"], 1200)) for h in hypotheses],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top hypotheses
        top_hypotheses = [h[0] for h in sorted_hypotheses[:3]]
        
        # Improve each hypothesis
        improved_hypotheses = []
        for hypothesis in top_hypotheses:
            task_data = {
                "action": "improve_hypothesis",
                "hypothesis": hypothesis,
                "research_config": research_config,
                "reviews": reviews
            }
            
            result = await self.evolution_agent.run(task_data)
            improved = result.get("improved_hypothesis", {})
            improved_hypotheses.append(improved)
        
        # Store in context memory
        self.context_memory.set("improved_hypotheses", improved_hypotheses)
        return improved_hypotheses
    
    async def _combine_hypotheses(self, hypotheses: List[Dict], research_config: Dict) -> Dict:
        """Combine multiple hypotheses into a unified hypothesis."""
        task_data = {
            "action": "combine_hypotheses",
            "hypotheses": hypotheses,
            "research_config": research_config
        }
        
        result = await self.evolution_agent.run(task_data)
        combined = result.get("combined_hypothesis", {})
        
        # Store in context memory
        self.context_memory.set("combined_hypothesis", combined)
        return combined
    
    async def _generate_research_overview(self, hypotheses: List[Dict], 
                                        research_config: Dict, elo_ratings: Dict) -> Dict:
        """Generate a research overview from the hypotheses."""
        task_data = {
            "action": "generate_research_overview",
            "top_hypotheses": hypotheses,
            "research_config": research_config,
            "elo_ratings": elo_ratings
        }
        
        result = await self.meta_review_agent.run(task_data)
        overview = result.get("research_overview", {})
        
        # Store in context memory
        self.context_memory.set("research_overview", overview)
        return overview
    
    async def add_scientist_feedback(self, hypothesis_content: str):
        """Add a hypothesis from a scientist."""
        # Get existing hypotheses
        hypotheses = self.context_memory.get("hypotheses", [])
        
        # Create a new hypothesis
        scientist_hypothesis = {
            "id": f"scientist_{int(time.time())}",
            "content": hypothesis_content,
            "source": "scientist",
            "timestamp": time.time()
        }
        
        # Add to hypotheses
        hypotheses.append(scientist_hypothesis)
        
        # Update context memory
        self.context_memory.set("hypotheses", hypotheses)
        
        # Review the hypothesis
        research_config = self.context_memory.get("research_config", {})
        
        task_data = {
            "action": "initial_review",
            "hypothesis": scientist_hypothesis,
            "research_config": research_config
        }
        
        result = await self.reflection_agent.run(task_data)
        review = result.get("review", {})
        
        # Update reviews
        reviews = self.context_memory.get("reviews", {})
        reviews[scientist_hypothesis["id"]] = review
        self.context_memory.set("reviews", reviews)
        
        return {
            "hypothesis": scientist_hypothesis,
            "review": review
        }


# Example usage
async def main():
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize the AI Co-scientist
    co_scientist = AICoScientist(api_key)
    
    # Process a research goal
    research_goal = """
    Develop a novel hypothesis for the key factor or process which contributes to ALS progression,
    focusing on protein homeostasis. We can use iPSC cells for the in-vitro experiments.
    """
    
    results = await co_scientist.process_research_goal(research_goal)
    
    # Print the research overview
    print("\n=== Research Overview ===\n")
    print(results["research_overview"]["content"])
    
    # Add scientist feedback
    feedback = """
    I have a hypothesis that TDP-43 aggregation in motor neurons might be driven by impaired
    autophagy specifically in response to ER stress, causing accumulation of misfolded proteins.
    """
    
    scientist_contribution = await co_scientist.add_scientist_feedback(feedback)
    
    print("\n=== Scientist Contribution Review ===\n")
    print(scientist_contribution["review"]["content"])


if __name__ == "__main__":
    asyncio.run(main()) 