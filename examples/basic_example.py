#!/usr/bin/env python3
"""
Basic example of using the Einstein AI Co-scientist system.
"""
import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv

# Add parent directory to path to allow running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from einstein_pkg import AICoScientist

async def main():
    """Run a basic example of the AI Co-scientist system."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Einstein AI Co-scientist example")
    parser.add_argument("--api-key", help="OpenAI API key (can also be set as OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4-turbo", help="Language model to use")
    parser.add_argument("--goal", help="Research goal to process")
    parser.add_argument("--feedback", help="Scientist feedback to add")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key is required. Set it with --api-key or OPENAI_API_KEY environment variable.")
        return 1
    
    # Default research goal
    default_goal = """
    Develop a novel hypothesis for the key factor or process which contributes to ALS progression,
    focusing on protein homeostasis. We can use iPSC cells for the in-vitro experiments.
    """
    
    # Default scientist feedback
    default_feedback = """
    I have a hypothesis that TDP-43 aggregation in motor neurons might be driven by impaired
    autophagy specifically in response to ER stress, causing accumulation of misfolded proteins.
    """
    
    # Use provided or default goal and feedback
    research_goal = args.goal or default_goal
    scientist_feedback = args.feedback or default_feedback
    
    # Initialize the AI Co-scientist
    co_scientist = AICoScientist(api_key=api_key, model=args.model)
    
    try:
        # Process research goal
        print("\n=== Processing Research Goal ===\n")
        print(f"Goal: {research_goal.strip()}")
        print("\nGenerating hypotheses and evaluating...\n")
        
        results = await co_scientist.process_research_goal(research_goal)
        
        # Print generated hypotheses
        print("\n=== Generated Hypotheses ===\n")
        for i, h in enumerate(results.get("hypotheses", []), 1):
            print(f"Hypothesis {i}:")
            print(f"{h.get('content', 'No content')}".strip())
            print()
        
        # Print research overview
        print("\n=== Research Overview ===\n")
        if "research_overview" in results and "content" in results["research_overview"]:
            print(results["research_overview"]["content"])
        else:
            print("No research overview generated.")
        
        # Add and evaluate scientist feedback
        print("\n=== Adding Scientist Feedback ===\n")
        print(f"Feedback: {scientist_feedback.strip()}\n")
        
        scientist_contribution = await co_scientist.add_scientist_feedback(scientist_feedback)
        
        print("\n=== Scientist Contribution Review ===\n")
        if "review" in scientist_contribution and "content" in scientist_contribution["review"]:
            print(scientist_contribution["review"]["content"])
        else:
            print("No review generated for scientist contribution.")
    
    finally:
        # Clean up resources
        await co_scientist.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 