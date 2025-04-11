# AI Co-scientist System

An implementation of a multi-agent scientific hypothesis generation and evaluation system using Microsoft's AutoGen framework.

## Overview

This system uses a team of specialized AI agents to generate, evaluate, and refine scientific hypotheses. It's built on top of AutoGen, a multi-agent conversation framework developed by Microsoft.

The system follows a structured workflow:

1. Parse research goals into structured configurations
2. Generate initial hypotheses
3. Review and critique each hypothesis
4. Conduct a tournament to rank hypotheses (using an Elo-based rating system)
5. Improve the top-ranking hypotheses
6. Combine improved hypotheses into a unified hypothesis
7. Generate a comprehensive research overview

## Agents

The system includes the following specialized agents:

- **SupervisorAgent**: Orchestrates the overall workflow and manages task assignments
- **GenerationAgent**: Generates initial research hypotheses
- **ReflectionAgent**: Critically reviews hypotheses for correctness, quality, novelty, and safety
- **RankingAgent**: Conducts simulated scientific debates to compare and rank hypotheses
- **EvolutionAgent**: Improves existing hypotheses based on reviews
- **MetaReviewAgent**: Synthesizes insights across multiple hypotheses and reviews

## Installation

Requirements:

- Python 3.8+
- AutoGen library

Install the necessary packages:

```bash
pip install autogen-agentchat autogen-ext
```

## Usage

```python
import asyncio
from einstein import AICoScientist

async def main():
    # Initialize the AI Co-scientist
    co_scientist = AICoScientist()
    
    # Process a research goal
    research_goal = """
    Develop a novel hypothesis for the key factor or process which contributes to 
    ALS progression, focusing on protein homeostasis. We can use iPSC cells for 
    the in-vitro experiments.
    """
    
    results = await co_scientist.process_research_goal(research_goal)
    
    # Print the research overview
    print("\n=== Research Overview ===\n")
    print(results["research_overview"]["content"])
    
    # You can also add your own hypothesis for review
    feedback = """
    I have a hypothesis that TDP-43 aggregation in motor neurons might be driven by 
    impaired autophagy specifically in response to ER stress, causing accumulation 
    of misfolded proteins.
    """
    
    scientist_contribution = await co_scientist.add_scientist_feedback(feedback)
    
    print("\n=== Scientist Contribution Review ===\n")
    print(scientist_contribution["review"]["content"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

The system requires an OpenAI API key to be set as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Extending the System

The system can be extended by:
1. Adding new specialized agents for specific scientific domains
2. Implementing additional tools for literature search, data analysis, etc.
3. Customizing the evaluation criteria for different research domains

## Credits

This implementation is based on the AI Co-scientist concept and uses Microsoft's AutoGen framework for agent orchestration.

- [AutoGen GitHub Repository](https://github.com/microsoft/autogen)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
