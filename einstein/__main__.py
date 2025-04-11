"""Main entry point for running the Einstein AI Co-scientist package directly."""
import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv

from einstein_pkg import AICoScientist

async def main():
    """Run the Einstein AI Co-scientist system."""
    parser = argparse.ArgumentParser(description="Einstein AI Co-scientist")
    parser.add_argument("--api-key", help="OpenAI API key (can also be set as OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4-turbo", help="Language model to use")
    parser.add_argument("--goal", help="Research goal to process")
    parser.add_argument("--feedback", help="Scientist feedback to add")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    args = parser.parse_args()
    
    if args.version:
        from einstein_pkg import __version__
        print(f"Einstein AI Co-scientist v{__version__}")
        return 0
    
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
    
    # Use provided or default goal
    research_goal = args.goal or default_goal
    
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
        
        # Process scientist feedback if provided
        if args.feedback:
            print("\n=== Adding Scientist Feedback ===\n")
            print(f"Feedback: {args.feedback.strip()}\n")
            
            scientist_contribution = await co_scientist.add_scientist_feedback(args.feedback)
            
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