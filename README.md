# AI Co-Scientist Platform

> Advanced AI-powered scientific research platform with multi-agent system for literature review, hypothesis generation, and experiment design

## Overview

The AI Co-Scientist Platform is a comprehensive web application that accelerates scientific discovery through artificial intelligence. Built on our extensive research and competitive analysis, it addresses key pain points identified in the 2025 Zendy survey of 1,502 researchers and incorporates insights from platforms like Google's AI Co-Scientist and Potato AI.

### Key Features

🔬 **Multi-Agent AI System** - 6 specialized agents working collaboratively
📚 **Literature Review** - AI-powered analysis across 50+ academic databases  
💡 **Hypothesis Generation** - Novel, testable hypotheses with scoring
🧪 **Experiment Design** - Detailed protocols and automated execution
📊 **Data Analysis** - Advanced statistical analysis and insights
✍️ **Writing Support** - AI assistance for scientific documents
📡 **Research Updates** - Personalized notifications on latest developments

## Architecture

### Backend (FastAPI)
- **Multi-Agent System**: Literature, Hypothesis, Experiment, Execution, Analysis, and Writing agents
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Vector Database**: ChromaDB for semantic search
- **AI Integration**: OpenAI GPT-4 with specialized prompts
- **External APIs**: arXiv, PubMed, Semantic Scholar integration
- **Authentication**: JWT-based with user management

### Frontend (Next.js)
- **Modern UI**: React with Tailwind CSS and Radix UI components
- **Real-time Updates**: React Query for state management
- **Responsive Design**: Mobile-first approach
- **Interactive Components**: Dynamic workflows and visualizations

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL 13+
- Redis 6+
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd einstein
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your configuration:
# - OPENAI_API_KEY: Your OpenAI API key
# - DATABASE_URL: PostgreSQL connection string
# - REDIS_URL: Redis connection string
```

4. **Database Setup**
```bash
# Ensure PostgreSQL is running
createdb ai_coscientist
python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
```

5. **Start Backend**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

6. **Frontend Setup** (in new terminal)
```bash
cd frontend
npm install
npm run dev
```

7. **Access the Application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Usage

### Starting a Research Workflow

1. **Navigate to the platform**: Open http://localhost:3000
2. **Enter Research Goal**: Describe your research objective
3. **Start AI Workflow**: Click "Start AI Research Workflow"
4. **Monitor Progress**: Watch as agents complete each step:
   - Literature Review
   - Hypothesis Generation
   - Experiment Design
   - Analysis & Synthesis

### Individual Features

#### Literature Review
```bash
curl -X POST "http://localhost:8000/api/v1/research/literature-review" \
  -H "Content-Type: application/json" \
  -d '{"query": "protein folding mechanisms", "max_papers": 50}'
```

#### Hypothesis Generation
```bash
curl -X POST "http://localhost:8000/api/v1/research/generate-hypotheses" \
  -H "Content-Type: application/json" \
  -d '{"research_goal": "Investigate protein aggregation in ALS", "num_hypotheses": 5}'
```

#### Experiment Design
```bash
curl -X POST "http://localhost:8000/api/v1/experiments/design" \
  -H "Content-Type: application/json" \
  -d '{"hypothesis": {"statement": "Protein X aggregation is mediated by pathway Y"}}'
```

## Configuration

### Backend Environment Variables

```bash
# API Configuration
DEBUG=true
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your-secret-key

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ai_coscientist
REDIS_URL=redis://localhost:6379

# AI Services
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7

# External APIs
ARXIV_API_URL=http://export.arxiv.org/api/query
PUBMED_API_URL=https://eutils.ncbi.nlm.nih.gov/entrez/eutils
SEMANTIC_SCHOLAR_API_URL=https://api.semanticscholar.org/graph/v1
```

### Frontend Environment Variables

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development

### Project Structure

```
einstein/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Core functionality
│   │   ├── models/         # Database models
│   │   ├── schemas/        # Pydantic schemas
│   │   └── services/       # AI agents and services
│   ├── main.py             # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # App router pages
│   │   ├── components/    # React components
│   │   └── lib/           # Utilities and API client
│   └── package.json       # Node dependencies
├── docs/                  # Research documentation
└── README.md
```

### Adding New AI Agents

1. **Create Agent Class**
```python
# backend/app/services/agents/my_agent.py
from .base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, openai_client, settings):
        super().__init__(openai_client, settings, "MyAgent")
    
    async def _setup(self):
        # Agent-specific setup
        pass
    
    async def my_functionality(self, input_data):
        # Implement agent functionality
        return result
```

2. **Register in AIServicesManager**
```python
# backend/app/services/ai_services.py
self.agents['my_agent'] = MyAgent(
    openai_client=self.openai_client,
    settings=self.settings
)
```

3. **Add API Endpoints**
```python
# backend/app/api/v1/endpoints/my_endpoints.py
@router.post("/my-endpoint")
async def my_endpoint(
    request: MyRequest,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    result = await ai_manager.agents['my_agent'].my_functionality(request.data)
    return {"result": result}
```

### Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# Integration tests
npm run test:e2e
```

## Research Foundation

This platform is built on extensive research documented in the `docs/` folder:

- **research-overview.md**: Market analysis and competitive landscape
- **platform-architecture.md**: Technical architecture and design decisions
- **feature-specifications.md**: Detailed feature requirements and specifications

### Key Research Insights

- **73.6% of scientists** already use AI tools (Zendy 2025 Survey)
- **Primary pain points**: Time consumption, information overload, fragmented workflows
- **Market gap**: No end-to-end research workflow platform exists
- **Opportunity**: Bridge AI tools with real laboratory execution

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on the existing Einstein AutoGen-based system
- Inspired by research in "Towards an AI Co-scientist for Accelerating Scientific Discoveries"
- Incorporates insights from Google's AI Co-Scientist and Potato AI platforms
- Designed based on needs identified in the 2025 Zendy researcher survey

## Support

- **Documentation**: `/docs` folder
- **API Reference**: http://localhost:8000/docs
- **Issues**: GitHub Issues
- **Community**: Research Community Forum

---

**Built for the scientific community** 🧬🔬🧪 