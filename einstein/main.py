"""Main class for the Einstein AI Co-scientist system."""
import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union

from einstein_pkg.memory import ContextMemory
from einstein_pkg.config import get_config, configure
from einstein_pkg.agents import (
    SupervisorAgent,
    GenerationAgent,
    ReflectionAgent,
    RankingAgent,
    EvolutionAgent,
    MetaReviewAgent
)

class AICoScientist:
    """Main class for the AI Co-scientist system."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        context_memory_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AI Co-scientist system.
        
        Args:
            api_key: Optional API key
            model: Optional model name
            context_memory_path: Optional path to context memory file
            config: Optional configuration dictionary
        """
        # Apply configuration if provided
        if config:
            configure(config)
        
        # Set up configuration
        if api_key:
            configure({"openai_api_key": api_key})
        if model:
            configure({"default_model": model})
        if context_memory_path:
            configure({"context_memory_path": context_memory_path})
        
        # Initialize context memory
        self.context_memory = ContextMemory()
        
        # Initialize agents
        self.supervisor = SupervisorAgent(self.context_memory)
        self.agents = {
            "SupervisorAgent": self.supervisor,
            "GenerationAgent": GenerationAgent(self.context_memory),
            "ReflectionAgent": ReflectionAgent(self.context_memory),
            "RankingAgent": RankingAgent(self.context_memory),
            "EvolutionAgent": EvolutionAgent(self.context_memory),
            "MetaReviewAgent": MetaReviewAgent(self.context_memory)
        }
        
        # Task queue for managing asynchronous tasks
        self.task_queue = asyncio.Queue()
        self.results = {}
        
        # Track if workers have been started
        self.workers_started = False
    
    async def _worker(self, worker_id: int):
        """
        Worker process for executing tasks.
        
        Args:
            worker_id: Worker ID
        """
        while True:
            try:
                task = await self.task_queue.get()
                
                try:
                    # Get the appropriate agent
                    agent = self.agents[task["agent"]]
                    
                    # Execute the task
                    result = await agent.run(task)
                    
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
            except asyncio.CancelledError:
                # Worker is being cancelled
                break
            except Exception as e:
                print(f"Worker {worker_id} encountered an unexpected error: {str(e)}")
                # Don't exit the worker loop
    
    async def start_workers(self, num_workers: int = 4):
        """
        Start worker processes.
        
        Args:
            num_workers: Number of worker processes
        """
        if not self.workers_started:
            self.worker_tasks = []
            for i in range(num_workers):
                worker_task = asyncio.create_task(self._worker(i))
                self.worker_tasks.append(worker_task)
            self.workers_started = True
    
    async def stop_workers(self):
        """Stop worker processes."""
        if self.workers_started:
            for worker_task in self.worker_tasks:
                worker_task.cancel()
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            self.workers_started = False
    
    async def process_research_goal(self, research_goal: str) -> Dict[str, Any]:
        """
        Process a research goal and generate hypotheses.
        
        Args:
            research_goal: Research goal text
            
        Returns:
            Dictionary with research results
        """
        # Start workers if not already started
        await self.start_workers()
        
        # Parse the research goal
        task_id = self.context_memory.create_id("parse_goal")
        await self.task_queue.put({
            "id": task_id,
            "agent": "SupervisorAgent",
            "action": "parse_research_goal",
            "research_goal": research_goal
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the result
        research_config = self.results[task_id].get("research_config", {})
        
        # Store the research configuration
        self.context_memory.set("research_config", research_config)
        
        # Generate initial hypotheses
        task_id = self.context_memory.create_id("generate_initial")
        await self.task_queue.put({
            "id": task_id,
            "agent": "GenerationAgent",
            "action": "generate_initial_hypotheses",
            "params": research_config
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the result
        hypotheses = self.results[task_id].get("hypotheses", [])
        
        # Store the hypotheses
        self.context_memory.set("hypotheses", hypotheses)
        
        # Review the hypotheses
        reviews = {}
        review_tasks = []
        
        for hypothesis in hypotheses:
            task_id = self.context_memory.create_id(f"review_{hypothesis.get('id', 'unknown')}")
            review_tasks.append(task_id)
            await self.task_queue.put({
                "id": task_id,
                "agent": "ReflectionAgent",
                "action": "initial_review",
                "hypothesis": hypothesis,
                "research_config": research_config
            })
        
        # Wait for all review tasks to complete
        await self.task_queue.join()
        
        # Collect reviews
        for task_id in review_tasks:
            review_result = self.results.get(task_id, {})
            review = review_result.get("review", {})
            hypothesis_id = review.get("hypothesis_id")
            if hypothesis_id:
                reviews[hypothesis_id] = review
        
        # Store the reviews
        self.context_memory.set("reviews", reviews)
        
        # Conduct tournament
        elo_ratings = {h.get("id", f"unknown_{i}"): 1200 for i, h in enumerate(hypotheses)}
        
        for i in range(len(hypotheses)):
            for j in range(i+1, len(hypotheses)):
                hypothesis1 = hypotheses[i]
                hypothesis2 = hypotheses[j]
                
                # Skip if any hypothesis doesn't have a valid ID
                if not hypothesis1.get("id") or not hypothesis2.get("id"):
                    continue
                
                task_id = self.context_memory.create_id(f"debate_{hypothesis1.get('id')}_{hypothesis2.get('id')}")
                await self.task_queue.put({
                    "id": task_id,
                    "agent": "RankingAgent",
                    "action": "scientific_debate",
                    "hypothesis1": hypothesis1,
                    "hypothesis2": hypothesis2,
                    "research_config": research_config,
                    "reviews": reviews
                })
                
                # Wait for the task to complete
                await self.task_queue.join()
                
                # Get the debate result
                debate_result = self.results.get(task_id, {}).get("debate_result", {})
                
                # Update Elo ratings
                task_id = self.context_memory.create_id("update_elo")
                await self.task_queue.put({
                    "id": task_id,
                    "agent": "RankingAgent",
                    "action": "update_elo_ratings",
                    "match_result": debate_result,
                    "elo_ratings": elo_ratings
                })
                
                # Wait for the task to complete
                await self.task_queue.join()
                
                # Get the updated Elo ratings
                updated_ratings = self.results.get(task_id, {}).get("updated_elo_ratings", {})
                if updated_ratings:
                    elo_ratings = updated_ratings
        
        # Store the Elo ratings
        self.context_memory.set("elo_ratings", elo_ratings)
        
        # Sort hypotheses by Elo rating
        sorted_hypotheses = sorted(
            [(h, elo_ratings.get(h.get("id", "unknown"), 1200)) for h in hypotheses],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top hypotheses for improvement
        top_hypotheses = [h[0] for h in sorted_hypotheses[:min(3, len(sorted_hypotheses))]]
        
        # Improve top hypotheses
        improved_hypotheses = []
        improve_tasks = []
        
        for hypothesis in top_hypotheses:
            task_id = self.context_memory.create_id(f"improve_{hypothesis.get('id', 'unknown')}")
            improve_tasks.append(task_id)
            await self.task_queue.put({
                "id": task_id,
                "agent": "EvolutionAgent",
                "action": "improve_hypothesis",
                "hypothesis": hypothesis,
                "research_config": research_config,
                "reviews": reviews
            })
        
        # Wait for all improvement tasks to complete
        await self.task_queue.join()
        
        # Collect improved hypotheses
        for task_id in improve_tasks:
            improve_result = self.results.get(task_id, {})
            improved = improve_result.get("improved_hypothesis")
            if improved:
                improved_hypotheses.append(improved)
        
        # Combine the improved hypotheses if there are more than one
        combined = None
        if len(improved_hypotheses) > 1:
            task_id = self.context_memory.create_id("combine")
            await self.task_queue.put({
                "id": task_id,
                "agent": "EvolutionAgent",
                "action": "combine_hypotheses",
                "hypotheses": improved_hypotheses,
                "research_config": research_config
            })
            
            # Wait for the task to complete
            await self.task_queue.join()
            
            # Get the combined hypothesis
            combined = self.results.get(task_id, {}).get("combined_hypothesis")
        elif len(improved_hypotheses) == 1:
            # If there's only one improved hypothesis, use it as the combined result
            combined = improved_hypotheses[0]
        
        # Generate a research overview
        all_top_hypotheses = top_hypotheses + improved_hypotheses
        if combined:
            all_top_hypotheses.append(combined)
        
        task_id = self.context_memory.create_id("overview")
        await self.task_queue.put({
            "id": task_id,
            "agent": "MetaReviewAgent",
            "action": "generate_research_overview",
            "top_hypotheses": all_top_hypotheses,
            "research_config": research_config,
            "elo_ratings": elo_ratings
        })
        
        # Wait for the task to complete
        await self.task_queue.join()
        
        # Get the research overview
        overview = self.results.get(task_id, {}).get("research_overview", {})
        
        # Return the final results
        result = {
            "research_config": research_config,
            "hypotheses": hypotheses,
            "reviews": reviews,
            "elo_ratings": elo_ratings,
            "improved_hypotheses": improved_hypotheses,
            "combined_hypothesis": combined,
            "research_overview": overview
        }
        
        # Store the result in context memory
        self.context_memory.set("latest_result", result)
        
        return result
    
    async def add_scientist_feedback(self, hypothesis_content: str) -> Dict[str, Any]:
        """
        Add a hypothesis from a scientist.
        
        Args:
            hypothesis_content: Hypothesis content text
            
        Returns:
            Dictionary with scientist contribution results
        """
        # Start workers if not already started
        await self.start_workers()
        
        # Get existing hypotheses
        hypotheses = self.context_memory.get("hypotheses", [])
        
        # Create a new hypothesis
        scientist_hypothesis = {
            "id": self.context_memory.create_id("scientist"),
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
        
        task_id = self.context_memory.create_id(f"review_{scientist_hypothesis.get('id')}")
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
        review_result = self.results.get(task_id, {})
        review = review_result.get("review", {})
        
        # Update reviews in context memory
        if review and "hypothesis_id" in review:
            reviews[review["hypothesis_id"]] = review
            self.context_memory.set("reviews", reviews)
        
        return {
            "hypothesis": scientist_hypothesis,
            "review": review
        }
    
    def get_latest_results(self) -> Dict[str, Any]:
        """
        Get the latest research results.
        
        Returns:
            Dictionary with latest research results
        """
        return self.context_memory.get("latest_result", {})
    
    def clear_memory(self):
        """Clear context memory."""
        self.context_memory.clear()
    
    async def close(self):
        """Clean up resources."""
        await self.stop_workers()


# Example usage function
async def example_usage():
    # Initialize the AI Co-scientist
    co_scientist = AICoScientist()
    
    try:
        # Process a research goal
        research_goal = """
        Develop a novel hypothesis for the key factor or process which contributes to ALS progression,
        focusing on protein homeostasis. We can use iPSC cells for the in-vitro experiments.
        """
        
        print("Processing research goal...")
        results = await co_scientist.process_research_goal(research_goal)
        
        # Print the research overview
        print("\n=== Research Overview ===\n")
        if "research_overview" in results and "content" in results["research_overview"]:
            print(results["research_overview"]["content"])
        else:
            print("No research overview generated.")
        
        # Add scientist feedback
        feedback = """
        I have a hypothesis that TDP-43 aggregation in motor neurons might be driven by impaired
        autophagy specifically in response to ER stress, causing accumulation of misfolded proteins.
        """
        
        print("\nAdding scientist feedback...")
        scientist_contribution = await co_scientist.add_scientist_feedback(feedback)
        
        print("\n=== Scientist Contribution Review ===\n")
        if "review" in scientist_contribution and "content" in scientist_contribution["review"]:
            print(scientist_contribution["review"]["content"])
        else:
            print("No review generated for scientist contribution.")
    
    finally:
        # Clean up resources
        await co_scientist.close()


if __name__ == "__main__":
    asyncio.run(example_usage()) 