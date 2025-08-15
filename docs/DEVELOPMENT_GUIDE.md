# Einstein Workflow Builder - Development Guide

## Setup Instructions

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Redis 6+
- Docker (optional)

### Backend Setup

1. **Create virtual environment:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment configuration:**
```bash
cp ../.env.example .env
# Edit .env with your configuration
```

4. **Database setup:**
```bash
# Make sure PostgreSQL is running
createdb einstein_db

# Run migrations (when implemented)
alembic upgrade head
```

5. **Start development server:**
```bash
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Install additional scientific packages:**
```bash
npm install @reactflow/core @reactflow/background @reactflow/controls @reactflow/minimap
npm install @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-select
npm install lucide-react recharts
npm install socket.io-client
```

3. **Start development server:**
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Docker Setup (Alternative)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Project Structure

```
einstein/
├── backend/
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Core configuration
│   │   ├── models/        # Database models
│   │   ├── schemas/       # Pydantic schemas
│   │   └── services/      # Business logic
│   ├── tests/             # Backend tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/           # Next.js app directory
│   │   ├── components/    # React components
│   │   ├── lib/           # Utilities
│   │   └── types/         # TypeScript types
│   └── package.json
├── docs/                  # Documentation
└── docker-compose.yml
```

## Development Workflow

### 1. Backend Development

**Adding a new API endpoint:**

1. Create/update model in `app/models/`
2. Create/update schema in `app/schemas/`
3. Add business logic in `app/services/`
4. Create endpoint in `app/api/api_v1/endpoints/`
5. Add tests in `tests/`

**Database migrations:**
```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

**Running tests:**
```bash
pytest tests/ -v
```

### 2. Frontend Development

**Component structure:**
```tsx
// components/workflow/WorkflowEditor.tsx
import { useCallback } from 'react';
import ReactFlow, { 
  Node, 
  Edge, 
  addEdge, 
  Connection 
} from '@reactflow/core';

export function WorkflowEditor() {
  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onConnect={onConnect}
      nodeTypes={nodeTypes}
    />
  );
}
```

**API integration:**
```tsx
// lib/api.ts
import { WorkflowResponse, WorkflowCreate } from '@/types/workflow';

export class WorkflowAPI {
  private baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

  async getWorkflows(): Promise<WorkflowResponse[]> {
    const response = await fetch(`${this.baseURL}/workflows`);
    return response.json();
  }

  async createWorkflow(data: WorkflowCreate): Promise<WorkflowResponse> {
    const response = await fetch(`${this.baseURL}/workflows`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return response.json();
  }
}
```

### 3. Adding Platform Adapters

To add support for a new scientific platform:

1. **Create adapter class:**
```python
# app/services/platforms/my_platform_adapter.py
from app.services.platform_service import PlatformAdapter

class MyPlatformAdapter(PlatformAdapter):
    async def authenticate(self, credentials):
        # Implementation
        pass
    
    async def submit_workflow(self, workflow_definition):
        # Convert internal format to platform format
        # Submit to platform API
        # Return external job ID
        pass
```

2. **Register adapter:**
```python
# app/services/platform_service.py
class PlatformService:
    def __init__(self):
        self.adapters = {
            "my_platform": MyPlatformAdapter(),
            # ... other adapters
        }
```

3. **Add configuration:**
```python
# app/core/config.py
class Settings(BaseSettings):
    MY_PLATFORM_API_KEY: Optional[str] = None
    MY_PLATFORM_BASE_URL: str = "https://api.myplatform.com"
```

### 4. Creating Scientific Node Types

**Backend node definition:**
```python
# app/models/scientific_nodes.py
SCIENTIFIC_NODE_TYPES = {
    "alphafold": {
        "name": "AlphaFold",
        "description": "Protein structure prediction",
        "inputs": ["sequence"],
        "outputs": ["structure"],
        "parameters": {
            "model": {"type": "select", "options": ["alphafold2", "alphafold3"]},
            "max_template_date": {"type": "date"},
            "use_gpu": {"type": "boolean", "default": True}
        }
    }
}
```

**Frontend node component:**
```tsx
// components/nodes/AlphaFoldNode.tsx
import { Handle, Position } from '@reactflow/core';

export function AlphaFoldNode({ data }: { data: any }) {
  return (
    <div className="alphafold-node">
      <Handle type="target" position={Position.Left} id="sequence" />
      <div>
        <h3>AlphaFold</h3>
        <p>Model: {data.model}</p>
      </div>
      <Handle type="source" position={Position.Right} id="structure" />
    </div>
  );
}
```

## Testing

### Backend Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_workflows.py -v
```

**Test structure:**
```python
# tests/test_workflows.py
import pytest
from app.services.workflow_service import WorkflowService

@pytest.mark.asyncio
async def test_create_workflow(db_session):
    service = WorkflowService(db_session)
    workflow_data = WorkflowCreate(
        name="Test Workflow",
        definition={"nodes": [], "edges": []}
    )
    workflow = await service.create_workflow(workflow_data)
    assert workflow.name == "Test Workflow"
```

### Frontend Testing

```bash
# Run tests
npm test

# Run with watch mode
npm run test:watch

# Run e2e tests
npm run test:e2e
```

## Code Style

### Backend (Python)

- Use Black for formatting: `black app/`
- Use isort for imports: `isort app/`
- Use mypy for type checking: `mypy app/`
- Follow PEP 8 guidelines

### Frontend (TypeScript)

- Use Prettier for formatting
- Use ESLint for linting
- Follow React/Next.js best practices
- Use TypeScript strictly

## Environment Variables

### Backend (.env)
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/einstein_db

# Redis
REDIS_URL=redis://localhost:6379

# API Keys
GALAXY_API_KEY=your_galaxy_api_key
NEXTFLOW_TOWER_TOKEN=your_nextflow_token

# File Storage
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=einstein-workflows

# Security
SECRET_KEY=your_secret_key_here
```

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## Debugging

### Backend Debugging

1. **Enable debug logging:**
```python
# app/core/config.py
LOG_LEVEL = "DEBUG"
```

2. **Use debugger:**
```python
import pdb; pdb.set_trace()  # Insert breakpoint
```

3. **Check logs:**
```bash
tail -f logs/einstein_$(date +%Y-%m-%d).log
```

### Frontend Debugging

1. **React Developer Tools**
2. **Browser DevTools Network tab**
3. **Console logging:**
```tsx
console.log('Debug info:', data);
```

## Performance Optimization

### Backend

- Use async/await for database operations
- Implement database connection pooling
- Add Redis caching for frequently accessed data
- Use background tasks for long-running operations

### Frontend

- Implement React.memo for expensive components
- Use React.useCallback and useMemo appropriately
- Lazy load components and routes
- Optimize bundle size with code splitting

## Deployment

### Production Environment

1. **Backend deployment:**
   - Use Gunicorn/Uvicorn with multiple workers
   - Set up proper logging and monitoring
   - Configure environment variables securely
   - Use a reverse proxy (nginx)

2. **Frontend deployment:**
   - Build optimized production bundle: `npm run build`
   - Deploy to Vercel, Netlify, or similar platform
   - Configure environment variables

3. **Database:**
   - Use managed PostgreSQL service
   - Set up automated backups
   - Configure connection pooling

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run linting and tests
5. Commit changes: `git commit -m "Add my feature"`
6. Push to branch: `git push origin feature/my-feature`
7. Create pull request
