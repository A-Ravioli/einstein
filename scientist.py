import os
import time
import asyncio
import json
from typing import List, Dict, Any, Optional, Union, Tuple

# You would need to replace this with your actual LLM API
from llm_api import LLMClient

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
        """Parse the research goal to extract preferences, attributes, and constraints.
        
        In a real implementation, this would use the LLM to extract structured information.
        """
        # This is a simplified implementation
        # In the real system, we would use the LLM to parse the goal
        
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


class BaseAgent:
    """Base class for all specialized agents."""
    
    def __init__(self, llm_client: LLMClient, context_memory: ContextMemory):
        self.llm_client = llm_client
        self.context_memory = context_memory
        
    async def execute(self, task: Dict) -> Dict:
        """Execute the agent's task."""
        raise NotImplementedError("Subclasses must implement execute()")


class SupervisorAgent(BaseAgent):
    """Orchestrates other agents and manages the workflow."""
    
    async def parse_research_goal(self, research_goal: str) -> Dict:
        """Parse a research goal into a configuration."""
        config = ResearchGoalConfig(research_goal)
        return config.to_dict()
    
    async def create_task_queue(self, research_config: Dict) -> List[Dict]:
        """Create a queue of tasks based on the research configuration."""
        tasks = [
            {"agent": "GenerationAgent", "action": "generate_initial_hypotheses", "params": research_config},
            # Additional tasks would be added dynamically based on the research configuration
        ]
        return tasks
    
    async def assign_agents(self, tasks: List[Dict]) -> Dict:
        """Assign tasks to specific agents."""
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
    
    async def execute(self, task: Dict) -> Dict:
        """Execute supervisor tasks."""
        if task["action"] == "parse_research_goal":
            return await self.parse_research_goal(task["research_goal"])
        elif task["action"] == "create_task_queue":
            return await self.create_task_queue(task["research_config"])
        elif task["action"] == "assign_agents":
            return await self.assign_agents(task["tasks"])
        else:
            raise ValueError(f"Unknown task action: {task['action']}")


class GenerationAgent(BaseAgent):
    """Generates initial research hypotheses."""
    
    async def search_literature(self, query: str) -> List[Dict]:
        """Search for relevant literature using web search API."""
        # In a real implementation, this would use a web search API
        # Simulated response for demonstration
        return [
            {"title": "Recent advances in the field", "content": "Summary of article..."},
            {"title": "Key findings from 2024", "content": "Summary of article..."}
        ]
    
    async def generate_initial_hypotheses(self, research_config: Dict) -> List[Dict]:
        """Generate initial research hypotheses."""
        # Search for relevant literature
        literature = await self.search_literature(research_config["original_goal"])
        
        # Generate hypotheses using the LLM
        prompt = self._create_generation_prompt(research_config, literature)
        response = await self.llm_client.generate(prompt)
        
        # Parse hypotheses from the response
        # This would be more sophisticated in a real implementation
        hypotheses = self._parse_hypotheses(response)
        
        return hypotheses
    
    def _create_generation_prompt(self, research_config: Dict, literature: List[Dict]) -> str:
        """Create a prompt for the LLM to generate hypotheses."""
        # In a real implementation, this would be a more sophisticated prompt
        prompt = f"""You are an expert tasked with formulating a novel and robust hypothesis to address the following objective.

Goal: {research_config['original_goal']}

Criteria for a strong hypothesis:
{', '.join(research_config['preferences'])}

Literature review and analytical rationale:
{json.dumps(literature, indent=2)}

Propose 3 detailed hypotheses that address this research goal:
"""
        return prompt
    
    def _parse_hypotheses(self, llm_response: str) -> List[Dict]:
        """Parse hypotheses from the LLM response."""
        # In a real implementation, this would use structured parsing
        # Simplified for demonstration
        hypotheses = []
        
        # Simple splitting by "Hypothesis" marker
        parts = llm_response.split("Hypothesis")
        for i, part in enumerate(parts[1:], 1):  # Skip the first part (before "Hypothesis 1")
            hypotheses.append({
                "id": f"h{i}",
                "content": part.strip(),
                "source": "generation",
                "timestamp": time.time()
            })
        
        return hypotheses
    
    async def simulated_debate(self, research_config: Dict, existing_hypotheses: List[Dict]) -> Dict:
        """Generate hypotheses through simulated scientific debate."""
        prompt = f"""You are an expert participating in a collaborative discourse concerning the generation of a hypothesis. 
        You will engage in a simulated discussion with other experts.

        Goal: {research_config['original_goal']}
        
        Criteria for a high-quality hypothesis:
        {', '.join(research_config['preferences'])}
        
        Existing hypotheses:
        {json.dumps([h['content'] for h in existing_hypotheses], indent=2)}
        
        Simulate a scientific debate among experts to refine these hypotheses and generate a novel hypothesis:
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Extract the final hypothesis from the debate
        final_hypothesis = self._extract_final_hypothesis(response)
        
        return {
            "id": f"h{len(existing_hypotheses) + 1}",
            "content": final_hypothesis,
            "source": "debate",
            "timestamp": time.time()
        }
    
    def _extract_final_hypothesis(self, debate_response: str) -> str:
        """Extract the final hypothesis from a debate response."""
        # In a real implementation, this would be more sophisticated
        # Looking for a "HYPOTHESIS" marker as mentioned in the paper
        if "HYPOTHESIS" in debate_response:
            parts = debate_response.split("HYPOTHESIS")
            return parts[1].strip()
        else:
            # Fallback: return the last paragraph
            paragraphs = debate_response.split("\n\n")
            return paragraphs[-1].strip()
    
    async def execute(self, task: Dict) -> Dict:
        """Execute generation tasks."""
        if task["action"] == "generate_initial_hypotheses":
            hypotheses = await self.generate_initial_hypotheses(task["params"])
            return {"hypotheses": hypotheses}
        elif task["action"] == "simulated_debate":
            hypothesis = await self.simulated_debate(task["research_config"], task["existing_hypotheses"])
            return {"hypothesis": hypothesis}
        else:
            raise ValueError(f"Unknown task action: {task['action']}")


class ReflectionAgent(BaseAgent):
    """Reviews and critiques hypotheses."""
    
    async def initial_review(self, hypothesis: Dict, research_config: Dict) -> Dict:
        """Perform an initial review of a hypothesis."""
        prompt = f"""You are an expert in scientific hypothesis evaluation.

        Goal: {research_config['original_goal']}
        
        Criteria for evaluation:
        {', '.join(research_config['preferences'])}
        
        Hypothesis:
        {hypothesis['content']}
        
        Perform an initial review assessing:
        1. Correctness
        2. Quality
        3. Novelty
        4. Safety (ethics)
        
        Provide a score (1-5) for each aspect and a detailed explanation.
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Parse the review
        review = self._parse_review(response)
        review["hypothesis_id"] = hypothesis["id"]
        review["type"] = "initial"
        review["timestamp"] = time.time()
        
        return review
    
    async def full_review(self, hypothesis: Dict, research_config: Dict) -> Dict:
        """Perform a full review of a hypothesis with web search."""
        # First, search for relevant literature
        literature = await self._search_relevant_literature(hypothesis["content"])
        
        prompt = f"""You are an expert in scientific hypothesis evaluation.

        Goal: {research_config['original_goal']}
        
        Criteria for evaluation:
        {', '.join(research_config['preferences'])}
        
        Hypothesis:
        {hypothesis['content']}
        
        Relevant literature:
        {json.dumps(literature, indent=2)}
        
        Perform a full review addressing:
        1. Correctness: Evaluate underlying assumptions and reasoning
        2. Quality: Assess clarity, specificity, and testability
        3. Novelty: Summarize known aspects and judge novelty based on literature
        4. Safety (ethics): Evaluate potential ethical concerns
        
        Provide a score (1-5) for each aspect and a detailed explanation.
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Parse the review
        review = self._parse_review(response)
        review["hypothesis_id"] = hypothesis["id"]
        review["type"] = "full"
        review["literature"] = literature
        review["timestamp"] = time.time()
        
        return review
    
    async def deep_verification(self, hypothesis: Dict) -> Dict:
        """Perform a deep verification of a hypothesis by decomposing assumptions."""
        prompt = f"""You are an expert in scientific verification.
        
        Hypothesis:
        {hypothesis['content']}
        
        Perform a deep verification by:
        1. Listing all assumptions in the hypothesis
        2. Breaking down each assumption into fundamental sub-assumptions
        3. Evaluating each sub-assumption for correctness
        4. Identifying any invalidating elements
        5. Summarizing potential reasons for hypothesis invalidation
        
        Focus on identifying subtle errors or flaws in reasoning.
        """
        
        response = await self.llm_client.generate(prompt)
        
        verification = {
            "hypothesis_id": hypothesis["id"],
            "type": "deep_verification",
            "content": response,
            "timestamp": time.time()
        }
        
        return verification
    
    async def _search_relevant_literature(self, hypothesis_content: str) -> List[Dict]:
        """Search for literature relevant to the hypothesis."""
        # In a real implementation, this would use a web search API
        # Simplified for demonstration
        return [
            {"title": "Related study 1", "content": "Summary of article..."},
            {"title": "Related study 2", "content": "Summary of article..."}
        ]
    
    def _parse_review(self, review_text: str) -> Dict:
        """Parse a review from the LLM response."""
        # In a real implementation, this would be more sophisticated
        # Simplified for demonstration
        
        # Extract scores using simple pattern matching
        scores = {
            "correctness": self._extract_score("Correctness", review_text),
            "quality": self._extract_score("Quality", review_text),
            "novelty": self._extract_score("Novelty", review_text),
            "safety": self._extract_score("Safety", review_text)
        }
        
        return {
            "scores": scores,
            "content": review_text,
        }
    
    def _extract_score(self, aspect: str, text: str) -> float:
        """Extract a score for a specific aspect from the review text."""
        # This is a very simplified implementation
        # In reality, would use more sophisticated pattern matching or structured output
        try:
            if f"{aspect}: " in text:
                # Extract the first digit after the aspect label
                score_text = text.split(f"{aspect}: ")[1][0]
                return float(score_text)
            else:
                return 3.0  # Default middle score
        except:
            return 3.0  # Default middle score
    
    async def execute(self, task: Dict) -> Dict:
        """Execute reflection tasks."""
        if task["action"] == "initial_review":
            review = await self.initial_review(task["hypothesis"], task["research_config"])
            return {"review": review}
        elif task["action"] == "full_review":
            review = await self.full_review(task["hypothesis"], task["research_config"])
            return {"review": review}
        elif task["action"] == "deep_verification":
            verification = await self.deep_verification(task["hypothesis"])
            return {"verification": verification}
        else:
            raise ValueError(f"Unknown task action: {task['action']}")


class RankingAgent(BaseAgent):
    """Evaluates and compares hypotheses using an Elo-based tournament."""
    
    def __init__(self, llm_client: LLMClient, context_memory: ContextMemory):
        super().__init__(llm_client, context_memory)
        self.initial_elo = 1200
    
    async def compare_hypotheses(self, hypothesis1: Dict, hypothesis2: Dict, 
                                research_config: Dict, reviews: Dict) -> Dict:
        """Compare two hypotheses and determine which is superior."""
        # Get reviews for both hypotheses
        review1 = reviews.get(hypothesis1["id"], {})
        review2 = reviews.get(hypothesis2["id"], {})
        
        prompt = f"""You are an expert evaluator tasked with comparing two hypotheses.
        
        Goal: {research_config['original_goal']}
        
        Evaluation criteria:
        {', '.join(research_config['preferences'])}
        
        Hypothesis 1:
        {hypothesis1['content']}
        
        Hypothesis 2:
        {hypothesis2['content']}
        
        Review of hypothesis 1:
        {json.dumps(review1, indent=2)}
        
        Review of hypothesis 2:
        {json.dumps(review2, indent=2)}
        
        Compare these hypotheses based on the evaluation criteria.
        End your response with "better hypothesis: 1" or "better hypothesis: 2".
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Determine the winner
        if "better hypothesis: 1" in response.lower():
            winner = 1
        elif "better hypothesis: 2" in response.lower():
            winner = 2
        else:
            # If no clear winner is determined, consider it a tie
            winner = 0
        
        return {
            "hypothesis1_id": hypothesis1["id"],
            "hypothesis2_id": hypothesis2["id"],
            "winner": winner,
            "rationale": response,
            "timestamp": time.time()
        }
    
    async def scientific_debate(self, hypothesis1: Dict, hypothesis2: Dict, 
                               research_config: Dict, reviews: Dict) -> Dict:
        """Conduct a simulated scientific debate between two hypotheses."""
        # Get reviews for both hypotheses
        review1 = reviews.get(hypothesis1["id"], {})
        review2 = reviews.get(hypothesis2["id"], {})
        
        prompt = f"""You are an expert in comparative analysis, simulating a panel of domain experts
        engaged in a structured discussion to evaluate two competing hypotheses.
        
        Goal: {research_config['original_goal']}
        
        Criteria for hypothesis superiority:
        {', '.join(research_config['preferences'])}
        
        Hypothesis 1:
        {hypothesis1['content']}
        
        Hypothesis 2:
        {hypothesis2['content']}
        
        Initial review of hypothesis 1:
        {json.dumps(review1, indent=2)}
        
        Initial review of hypothesis 2:
        {json.dumps(review2, indent=2)}
        
        Simulate a scientific debate evaluating these hypotheses.
        After thorough discussion, conclude with "better idea: 1" or "better idea: 2".
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Determine the winner
        if "better idea: 1" in response.lower():
            winner = 1
        elif "better idea: 2" in response.lower():
            winner = 2
        else:
            # If no clear winner is determined, consider it a tie
            winner = 0
        
        return {
            "hypothesis1_id": hypothesis1["id"],
            "hypothesis2_id": hypothesis2["id"],
            "winner": winner,
            "debate": response,
            "timestamp": time.time()
        }
    
    async def update_elo_ratings(self, match_result: Dict, elo_ratings: Dict) -> Dict:
        """Update Elo ratings based on match results."""
        h1_id = match_result["hypothesis1_id"]
        h2_id = match_result["hypothesis2_id"]
        
        # Get current ratings (or use initial rating if not available)
        rating1 = elo_ratings.get(h1_id, self.initial_elo)
        rating2 = elo_ratings.get(h2_id, self.initial_elo)
        
        # Calculate expected scores
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
        
        # Calculate actual scores
        if match_result["winner"] == 1:
            actual1 = 1
            actual2 = 0
        elif match_result["winner"] == 2:
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
    
    async def execute(self, task: Dict) -> Dict:
        """Execute ranking tasks."""
        if task["action"] == "compare_hypotheses":
            result = await self.compare_hypotheses(
                task["hypothesis1"], 
                task["hypothesis2"],
                task["research_config"],
                task["reviews"]
            )
            return {"comparison_result": result}
        elif task["action"] == "scientific_debate":
            result = await self.scientific_debate(
                task["hypothesis1"],
                task["hypothesis2"],
                task["research_config"],
                task["reviews"]
            )
            return {"debate_result": result}
        elif task["action"] == "update_elo_ratings":
            updated_ratings = await self.update_elo_ratings(
                task["match_result"],
                task["elo_ratings"]
            )
            return {"updated_elo_ratings": updated_ratings}
        else:
            raise ValueError(f"Unknown task action: {task['action']}")


class ProximityAgent(BaseAgent):
    """Calculates similarity between hypotheses."""
    
    async def calculate_similarity(self, hypothesis1: Dict, hypothesis2: Dict) -> float:
        """Calculate similarity between two hypotheses."""
        prompt = f"""You are an expert in analyzing scientific ideas.
        
        Compare the following two hypotheses and determine their similarity on a scale from 0 to 1,
        where 0 means completely different and 1 means identical.
        
        Hypothesis 1:
        {hypothesis1['content']}
        
        Hypothesis 2:
        {hypothesis2['content']}
        
        Consider similarities in:
        - Core concepts and mechanisms
        - Proposed entities and relationships
        - Experimental approaches
        
        Return only a single number between 0 and 1 representing the similarity score.
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Extract the similarity score
        try:
            # Find a decimal number in the response
            import re
            similarity = float(re.search(r"0\.\d+", response).group(0))
        except:
            # If extraction fails, default to a moderate similarity
            similarity = 0.5
        
        return similarity
    
    async def build_proximity_graph(self, hypotheses: List[Dict]) -> Dict:
        """Build a proximity graph for all hypotheses."""
        proximity_graph = {}
        
        for i, h1 in enumerate(hypotheses):
            h1_id = h1["id"]
            proximity_graph[h1_id] = {}
            
            for j, h2 in enumerate(hypotheses):
                h2_id = h2["id"]
                
                if h1_id == h2_id:
                    # Same hypothesis, perfect similarity
                    proximity_graph[h1_id][h2_id] = 1.0
                elif j > i:
                    # Calculate similarity
                    similarity = await self.calculate_similarity(h1, h2)
                    proximity_graph[h1_id][h2_id] = similarity
                    
                    # Mirror the similarity (symmetric relationship)
                    if h2_id not in proximity_graph:
                        proximity_graph[h2_id] = {}
                    proximity_graph[h2_id][h1_id] = similarity
        
        return proximity_graph
    
    async def execute(self, task: Dict) -> Dict:
        """Execute proximity tasks."""
        if task["action"] == "calculate_similarity":
            similarity = await self.calculate_similarity(task["hypothesis1"], task["hypothesis2"])
            return {"similarity": similarity}
        elif task["action"] == "build_proximity_graph":
            proximity_graph = await self.build_proximity_graph(task["hypotheses"])
            return {"proximity_graph": proximity_graph}
        else:
            raise ValueError(f"Unknown task action: {task['action']}")


class EvolutionAgent(BaseAgent):
    """Improves existing hypotheses."""
    
    async def improve_hypothesis(self, hypothesis: Dict, research_config: Dict, reviews: Dict) -> Dict:
        """Improve a hypothesis based on reviews and research configuration."""
        # Get reviews for the hypothesis
        hypothesis_review = reviews.get(hypothesis["id"], {})
        
        prompt = f"""You are an expert in scientific research and hypothesis refinement.
        
        Original hypothesis:
        {hypothesis['content']}
        
        Research goal:
        {research_config['original_goal']}
        
        Criteria for a robust hypothesis:
        {', '.join(research_config['preferences'])}
        
        Reviews of the hypothesis:
        {json.dumps(hypothesis_review, indent=2)}
        
        Improve this hypothesis by addressing the critiques in the reviews.
        Focus on:
        1. Enhancing clarity and specificity
        2. Addressing identified weaknesses
        3. Improving testability and feasibility
        4. Maintaining or enhancing novelty
        
        Provide a complete, revised hypothesis that maintains the core insight while addressing limitations.
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Create the improved hypothesis
        improved_hypothesis = {
            "id": f"{hypothesis['id']}_improved",
            "content": response,
            "parent_id": hypothesis["id"],
            "source": "evolution",
            "timestamp": time.time()
        }
        
        return improved_hypothesis
    
    async def combine_hypotheses(self, hypotheses: List[Dict], research_config: Dict) -> Dict:
        """Combine multiple hypotheses to create a new, improved hypothesis."""
        prompt = f"""You are an expert in scientific research and hypothesis development.
        
        Research goal:
        {research_config['original_goal']}
        
        Criteria for a robust hypothesis:
        {', '.join(research_config['preferences'])}
        
        Consider the following hypotheses:
        
        {json.dumps([h['content'] for h in hypotheses], indent=2)}
        
        Create a new hypothesis that combines the strengths of these hypotheses.
        Focus on:
        1. Identifying complementary aspects from each hypothesis
        2. Resolving any contradictions between them
        3. Creating a coherent, integrated hypothesis that leverages insights from all input hypotheses
        4. Ensuring the combined hypothesis is novel, testable, and aligned with the research goal
        
        Provide a complete, integrated hypothesis.
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Create the combined hypothesis
        parent_ids = [h["id"] for h in hypotheses]
        combined_hypothesis = {
            "id": f"combined_{'_'.join(parent_ids)}",
            "content": response,
            "parent_ids": parent_ids,
            "source": "combination",
            "timestamp": time.time()
        }
        
        return combined_hypothesis
    
    async def out_of_box_thinking(self, research_config: Dict, existing_hypotheses: List[Dict]) -> Dict:
        """Generate a novel hypothesis through out-of-box thinking."""
        prompt = f"""You are an expert researcher tasked with generating a novel hypothesis through out-of-box thinking.
        
        Research goal:
        {research_config['original_goal']}
        
        Criteria for a robust hypothesis:
        {', '.join(research_config['preferences'])}
        
        Existing hypotheses (for inspiration, not direct replication):
        {json.dumps([h['content'] for h in existing_hypotheses], indent=2)}
        
        Generate a completely novel hypothesis that:
        1. Takes a fundamentally different approach than existing hypotheses
        2. Challenges conventional thinking in the field
        3. Draws inspiration from analogous domains or principles
        4. Remains scientifically plausible and testable
        
        Think outside-the-box while ensuring scientific rigor.
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Create the novel hypothesis
        novel_hypothesis = {
            "id": f"novel_{int(time.time())}",
            "content": response,
            "source": "out_of_box",
            "timestamp": time.time()
        }
        
        return novel_hypothesis
    
    async def execute(self, task: Dict) -> Dict:
        """Execute evolution tasks."""
        if task["action"] == "improve_hypothesis":
            improved = await self.improve_hypothesis(
                task["hypothesis"],
                task["research_config"],
                task["reviews"]
            )
            return {"improved_hypothesis": improved}
        elif task["action"] == "combine_hypotheses":
            combined = await self.combine_hypotheses(
                task["hypotheses"],
                task["research_config"]
            )
            return {"combined_hypothesis": combined}
        elif task["action"] == "out_of_box_thinking":
            novel = await self.out_of_box_thinking(
                task["research_config"],
                task["existing_hypotheses"]
            )
            return {"novel_hypothesis": novel}
        else:
            raise ValueError(f"Unknown task action: {task['action']}")


class MetaReviewAgent(BaseAgent):
    """Synthesizes insights and provides feedback."""
    
    async def generate_meta_review(self, reviews: List[Dict], research_config: Dict) -> Dict:
        """Generate a meta-review from multiple reviews."""
        prompt = f"""You are an expert in scientific research and meta-analysis.
        
        Research goal:
        {research_config['original_goal']}
        
        Preferences:
        {', '.join(research_config['preferences'])}
        
        Reviews for meta-analysis:
        {json.dumps(reviews, indent=2)}
        
        Generate a comprehensive meta-review that:
        1. Identifies recurring critique points and common issues raised by reviewers
        2. Provides actionable insights for researchers developing future proposals
        3. Synthesizes patterns in strengths and weaknesses across reviewed hypotheses
        4. Suggests high-level directions for improvement
        
        Focus on providing a synthesized meta-analysis rather than evaluating individual proposals or reviews.
        """
        
        response = await self.llm_client.generate(prompt)
        
        meta_review = {
            "content": response,
            "timestamp": time.time()
        }
        
        return meta_review
    
    async def generate_research_overview(self, top_hypotheses: List[Dict], 
                                       research_config: Dict, elo_ratings: Dict) -> Dict:
        """Generate a research overview from top-ranked hypotheses."""
        # Sort hypotheses by Elo rating
        sorted_hypotheses = sorted(
            [(h, elo_ratings.get(h["id"], 1200)) for h in top_hypotheses],
            key=lambda x: x[1],
            reverse=True
        )
        
        prompt = f"""You are an expert in scientific research synthesis.
        
        Research goal:
        {research_config['original_goal']}
        
        Top-ranked research hypotheses:
        {json.dumps([{"content": h[0]["content"], "elo_rating": h[1]} for h in sorted_hypotheses], indent=2)}
        
        Generate a comprehensive research overview that:
        1. Summarizes the key research directions represented by these hypotheses
        2. Identifies potential research areas and directions relevant to the research goal
        3. Suggests specific experiments within each research area
        4. Provides illustrative example topics for each area
        
        Structure your overview to effectively map the boundary of current knowledge and highlight future areas of exploration.
        """
        
        response = await self.llm_client.generate(prompt)
        
        research_overview = {
            "content": response,
            "timestamp": time.time()
        }
        
        return research_overview
    
    async def identify_research_contacts(self, research_overview: Dict) -> List[Dict]:
        """Identify potential research contacts based on the research overview."""
        prompt = f"""You are an expert in the scientific community and research networking.
        
        Research overview:
        {research_overview['content']}
        
        Identify potential domain experts who would be qualified to review or collaborate on 
        research related to this overview. For each suggested contact:
        
        1. Provide their area of expertise
        2. Explain why they would be relevant to this research
        3. Describe how their expertise complements the proposed research directions
        
        Focus on identifying experts across different relevant subdisciplines represented in the overview.
        """
        
        response = await self.llm_client.generate(prompt)
        
        # In a real implementation, this would parse structured contact information
        # Simplified for demonstration
        contacts = [
            {
                "expertise_area": "Key area from overview",
                "relevance": "Description of relevance",
                "timestamp": time.time()
            }
        ]
        
        return contacts
    
    async def execute(self, task: Dict) -> Dict:
        """Execute meta-review tasks."""
        if task["action"] == "generate_meta_review":
            meta_review = await self.generate_meta_review(task["reviews"], task["research_config"])
            return {"meta_review": meta_review}
        elif task["action"] == "generate_research_overview":
            overview = await self.generate_research_overview(
                task["top_hypotheses"],
                task["research_config"],
                task["elo_ratings"]
            )
            return {"research_overview": overview}
        elif task["action"] == "identify_research_contacts":
            contacts = await self.identify_research_contacts(task["research_overview"])
            return {"research_contacts": contacts}
        else:
            raise ValueError(f"Unknown task action: {task['action']}")


class AICoScientist:
    """Main class for the AI Co-scientist system."""
    
    def __init__(self):
        # Initialize LLM client
        self.llm_client = LLMClient()
        
        # Initialize context memory
        self.context_memory = ContextMemory()
        
        # Initialize agents
        self.supervisor = SupervisorAgent(self.llm_client, self.context_memory)
        self.agents = {
            "GenerationAgent": GenerationAgent(self.llm_client, self.context_memory),
            "ReflectionAgent": ReflectionAgent(self.llm_client, self.context_memory),
            "RankingAgent": RankingAgent(self.llm_client, self.context_memory),
            "ProximityAgent": ProximityAgent(self.llm_client, self.context_memory),
            "EvolutionAgent": EvolutionAgent(self.llm_client, self.context_memory),
            "MetaReviewAgent": MetaReviewAgent(self.llm_client, self.context_memory),
            "SupervisorAgent": self.supervisor
        }
        
        # Worker queue for managing asynchronous tasks
        self.task_queue = asyncio.Queue()
        self.results = {}
    
    async def _worker(self, worker_id: int):
        """Worker process for executing tasks."""
        while True:
            task = await self.task_queue.get()
            
            try:
                # Get the appropriate agent
                agent = self.agents[task["agent"]]
                
                # Execute the task
                result = await agent.execute(task)
                
                # Store the result
                self.results[task["id"]] = result
                
                # Update context memory with task completion
                self.context_memory.set(f"task_{task['id']}_completed", True)
                self.context_memory.set(f"task_{task['id']}_result", result)
                
            except Exception as e:
                print(f"Worker {worker_id} encountered an error: {str(e)}")
                # Store the error
                self.results[task["id"]] = {"error": str(e)}
                
            finally:
                # Mark the task as done
                self.task_queue.task_done()
    
    async def start_workers(self, num_workers: int = 4):
        """Start worker processes."""
        for i in range(num_workers):
            asyncio.create_task(self._worker(i))
    
    async def process_research_goal(self, research_goal: str):
        """Process a research goal and generate hypotheses."""
        # Start workers
        await self.start_workers()
        
        # Parse the research goal
        task_id = f"parse_goal_{int(time.time())}"
        await self.task_queue.put({
            "id": task_id,
            "agent": "SupervisorAgent",
            "action": "parse_research_goal",
            "research_goal": research_goal
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the result
        research_config = self.results[task_id]["research_config"]
        
        # Store the research configuration
        self.context_memory.set("research_config", research_config)
        
        # Generate initial hypotheses
        task_id = f"generate_initial_{int(time.time())}"
        await self.task_queue.put({
            "id": task_id,
            "agent": "GenerationAgent",
            "action": "generate_initial_hypotheses",
            "params": research_config
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the result
        hypotheses = self.results[task_id]["hypotheses"]
        
        # Store the hypotheses
        self.context_memory.set("hypotheses", hypotheses)
        
        # Review the hypotheses
        reviews = {}
        for hypothesis in hypotheses:
            task_id = f"review_{hypothesis['id']}_{int(time.time())}"
            await self.task_queue.put({
                "id": task_id,
                "agent": "ReflectionAgent",
                "action": "initial_review",
                "hypothesis": hypothesis,
                "research_config": research_config
            })
            
            # Wait for the task to complete
            await self.task_queue.join()
            
            # Get the result
            review = self.results[task_id]["review"]
            reviews[hypothesis["id"]] = review
        
        # Store the reviews
        self.context_memory.set("reviews", reviews)
        
        # Build a proximity graph
        task_id = f"proximity_{int(time.time())}"
        await self.task_queue.put({
            "id": task_id,
            "agent": "ProximityAgent",
            "action": "build_proximity_graph",
            "hypotheses": hypotheses
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the result
        proximity_graph = self.results[task_id]["proximity_graph"]
        
        # Store the proximity graph
        self.context_memory.set("proximity_graph", proximity_graph)
        
        # Conduct tournament (simplified)
        elo_ratings = {h["id"]: 1200 for h in hypotheses}  # Initial ratings
        
        for i in range(len(hypotheses)):
            for j in range(i+1, len(hypotheses)):
                task_id = f"debate_{hypotheses[i]['id']}_{hypotheses[j]['id']}_{int(time.time())}"
                await self.task_queue.put({
                    "id": task_id,
                    "agent": "RankingAgent",
                    "action": "scientific_debate",
                    "hypothesis1": hypotheses[i],
                    "hypothesis2": hypotheses[j],
                    "research_config": research_config,
                    "reviews": reviews
                })
                
                # Wait for the task to complete
                await self.task_queue.join()
                
                # Get the result
                debate_result = self.results[task_id]["debate_result"]
                
                # Update Elo ratings
                task_id = f"update_elo_{int(time.time())}"
                await self.task_queue.put({
                    "id": task_id,
                    "agent": "RankingAgent",
                    "action": "update_elo_ratings",
                    "match_result": debate_result,
                    "elo_ratings": elo_ratings
                })
                
                # Wait for the task to complete
                await self.task_queue.join()
                
                # Get the result
                elo_ratings = self.results[task_id]["updated_elo_ratings"]
        
        # Store the Elo ratings
        self.context_memory.set("elo_ratings", elo_ratings)
        
        # Sort hypotheses by Elo rating
        sorted_hypotheses = sorted(
            [(h, elo_ratings.get(h["id"], 1200)) for h in hypotheses],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top hypotheses for improvement
        top_hypotheses = [h[0] for h in sorted_hypotheses[:3]]
        
        # Improve top hypotheses
        improved_hypotheses = []
        for hypothesis in top_hypotheses:
            task_id = f"improve_{hypothesis['id']}_{int(time.time())}"
            await self.task_queue.put({
                "id": task_id,
                "agent": "EvolutionAgent",
                "action": "improve_hypothesis",
                "hypothesis": hypothesis,
                "research_config": research_config,
                "reviews": reviews
            })
            
            # Wait for the task to complete
            await self.task_queue.join()
            
            # Get the result
            improved = self.results[task_id]["improved_hypothesis"]
            improved_hypotheses.append(improved)
        
        # Combine the improved hypotheses
        task_id = f"combine_{int(time.time())}"
        await self.task_queue.put({
            "id": task_id,
            "agent": "EvolutionAgent",
            "action": "combine_hypotheses",
            "hypotheses": improved_hypotheses,
            "research_config": research_config
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the result
        combined = self.results[task_id]["combined_hypothesis"]
        
        # Generate a research overview
        task_id = f"overview_{int(time.time())}"
        await self.task_queue.put({
            "id": task_id,
            "agent": "MetaReviewAgent",
            "action": "generate_research_overview",
            "top_hypotheses": top_hypotheses + improved_hypotheses + [combined],
            "research_config": research_config,
            "elo_ratings": elo_ratings
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the result
        overview = self.results[task_id]["research_overview"]
        
        # Return the final results
        return {
            "research_config": research_config,
            "hypotheses": hypotheses,
            "reviews": reviews,
            "elo_ratings": elo_ratings,
            "improved_hypotheses": improved_hypotheses,
            "combined_hypothesis": combined,
            "research_overview": overview
        }
    
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
        reviews = self.context_memory.get("reviews", {})
        
        task_id = f"review_{scientist_hypothesis['id']}_{int(time.time())}"
        await self.task_queue.put({
            "id": task_id,
            "agent": "ReflectionAgent",
            "action": "initial_review",
            "hypothesis": scientist_hypothesis,
            "research_config": research_config
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the result
        review = self.results[task_id]["review"]
        reviews[scientist_hypothesis["id"]] = review
        
        # Update context memory
        self.context_memory.set("reviews", reviews)
        
        return {
            "hypothesis": scientist_hypothesis,
            "review": review
        }


# Example usage
async def main():
    # Initialize the AI Co-scientist
    co_scientist = AICoScientist()
    
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