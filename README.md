# Einstein AI Co-Scientist

A simplified implementation of a multi-agent AI system inspired by Google's AI co-scientist. This project demonstrates how multiple specialized AI agents can collaborate to generate, evaluate, and refine scientific hypotheses.

## Overview

Einstein AI Co-Scientist is a prototype system that mimics the architecture and functionality of Google's AI co-scientist. It uses a collection of specialized agents to:

1. Gather relevant research information from the web
2. Generate novel research hypotheses
3. Evaluate and refine these hypotheses
4. Rank and prioritize the most promising ideas
5. Assess alignment with research goals
6. Provide comprehensive evaluation of the process

## System Architecture

The system consists of several specialized agents:

- **Research Agent**: Gathers information from the web to support hypothesis generation
- **Supervisor Agent**: Coordinates the overall process and manages resources
- **Generation Agent**: Creates initial hypotheses based on research goals and findings
- **Reflection Agent**: Evaluates hypotheses and provides feedback based on research findings
- **Evolution Agent**: Refines hypotheses through iterative improvement
- **Ranking Agent**: Compares and prioritizes different hypotheses
- **Proximity Agent**: Assesses alignment with research goals
- **Meta-review Agent**: Provides comprehensive evaluation of the entire process

## Getting Started

### Prerequisites

- Python 3.9+
- Required packages (see requirements.txt)
- OpenAI API key
- Serper.dev API key (for web search functionality)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/einstein.git
cd einstein

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env to add your API keys
```

### Usage

```bash
python main.py --research_goal "Your research goal here"
```

#### Command Line Options

- `--research_goal`: The research goal to explore (required)
- `--output`: Output file path for the results (default: results.json)
- `--verbose`: Enable verbose output
- `--disable_web_search`: Disable web search functionality

## Examples

### Basic Example

```bash
python main.py --research_goal "Investigate potential mechanisms for improving drug delivery across the blood-brain barrier"
```

### Research-Enhanced Example

```bash
# Run the research-enhanced example script
python examples/research_example.py
```

This example demonstrates how the Research Agent gathers information from the web to enhance hypothesis generation for quantum computing error correction.

## Web Search Functionality

The system uses the Serper.dev API to perform web searches. To use this functionality:

1. Sign up for a Serper.dev API key at https://serper.dev/
2. Add your API key to the `.env` file:
   ```
   SERPER_API_KEY=your_serper_api_key_here
   ```

If you don't want to use web search, you can disable it with the `--disable_web_search` flag.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
