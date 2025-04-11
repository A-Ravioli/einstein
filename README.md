# Einstein: AI Co-scientist System

A multi-agent scientific hypothesis generation and evaluation system built on Microsoft's AutoGen framework.

## Overview

Einstein is an implementation of an AI Co-scientist system, designed to help researchers generate, evaluate, and refine scientific hypotheses. The system uses a team of specialized AI agents, each focusing on different aspects of the scientific process:

- **Supervisor**: Orchestrates the research workflow and coordinates other agents
- **Generation**: Creates novel, testable scientific hypotheses
- **Reflection**: Reviews and critiques hypotheses with scientific rigor
- **Ranking**: Compares hypotheses through simulated scientific debates
- **Evolution**: Improves hypotheses based on feedback and combines strengths
- **Meta-Review**: Synthesizes insights and provides research overviews

The system follows a structured workflow:
1. Parse research goals into structured configurations
2. Generate initial hypotheses
3. Review and critique each hypothesis
4. Conduct a tournament to rank hypotheses (using an Elo-based rating system)
5. Improve the top-ranking hypotheses
6. Combine improved hypotheses into a unified hypothesis
7. Generate a comprehensive research overview

## Installation

Install the package with pip:

```
pip install -e .
```

Or install the dependencies separately:

```
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
import asyncio
from einstein_pkg import AICoScientist

async def main():
    # Initialize the AI Co-scientist with your OpenAI API key
    co_scientist = AICoScientist(api_key="your-api-key")
    
    # Process a research goal
    research_goal = "Develop a novel hypothesis for the key factor contributing to ALS progression."
    results = await co_scientist.process_research_goal(research_goal)
    
    # Print the research overview
    print(results["research_overview"]["content"])
    
    # Add scientist feedback
    feedback = "I have a hypothesis that TDP-43 aggregation might be driven by impaired autophagy."
    contribution = await co_scientist.add_scientist_feedback(feedback)
    
    # Clean up resources
    await co_scientist.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Command Line Example

Run the included example script:

```
python examples/basic_example.py --api-key your-api-key --model gpt-4-turbo
```

Or use a .env file with your API key and run without the --api-key parameter:

```
echo "OPENAI_API_KEY=your-api-key" > .env
python examples/basic_example.py
```

## Package Structure

```
einstein_pkg/
├── einstein_pkg/           # Main package
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration handling
│   ├── main.py             # Main AICoScientist class
│   ├── memory.py           # Context memory persistence
│   ├── agents/             # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py         # Base agent class
│   │   ├── supervisor.py   # Research supervisor agent
│   │   ├── generation.py   # Hypothesis generation agent
│   │   ├── reflection.py   # Hypothesis review agent
│   │   ├── ranking.py      # Hypothesis ranking agent
│   │   ├── evolution.py    # Hypothesis improvement agent
│   │   └── meta_review.py  # Research synthesis agent
│   ├── models/             # Domain models
│   │   ├── __init__.py
│   │   └── research_config.py  # Research goal configuration
│   └── tools/              # AutoGen tools
│       ├── __init__.py
│       ├── literature_search.py  # Literature search tool
│       └── research_goal.py      # Research goal parsing tool
├── examples/               # Example scripts
│   └── basic_example.py    # Basic usage example
├── setup.py                # Package setup
└── requirements.txt        # Dependencies
```

## Configuration

Einstein can be configured through:

1. **Constructor parameters**: Pass configuration directly when creating the AICoScientist instance
2. **Environment variables**: Set configuration in environment or .env file
3. **Configuration dictionary**: Use the configure() function to set global configuration

Common configuration options:
- `openai_api_key`: Your OpenAI API key
- `default_model`: Model to use (default: "gpt-4-turbo")
- `context_memory_path`: Path to store context memory (default: "context_memory.json")

## Advanced Usage

### Custom Memory Path

```python
co_scientist = AICoScientist(context_memory_path="my_research_memory.json")
```

### Clearing Memory

```python
co_scientist.clear_memory()
```

### Getting Latest Results

```python
latest_results = co_scientist.get_latest_results()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project was inspired by and implements concepts from the paper "Towards an AI Co-scientist for Accelerating Scientific Discoveries" and uses Microsoft's AutoGen framework for multi-agent communication. 