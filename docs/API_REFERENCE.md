# Einstein Workflow Builder - API Reference

## Overview

The Einstein Scientific Workflow Builder provides a REST API for creating, managing, and executing scientific workflows across multiple cloud platforms.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

The API uses JWT tokens for authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer <your-jwt-token>
```

## Workflows API

### List Workflows

```http
GET /workflows
```

**Query Parameters:**
- `skip` (int): Number of records to skip (default: 0)
- `limit` (int): Maximum number of records to return (default: 100)
- `status` (string): Filter by workflow status (`draft`, `active`, `archived`)
- `tags` (array): Filter by tags

**Response:**
```json
{
  "workflows": [
    {
      "id": 1,
      "name": "AlphaFold Protein Prediction",
      "description": "Predict protein structure using AlphaFold",
      "definition": {...},
      "status": "active",
      "version": "1.0.0",
      "tags": ["protein", "structure", "alphafold"],
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 100
}
```

### Create Workflow

```http
POST /workflows
```

**Request Body:**
```json
{
  "name": "My Workflow",
  "description": "Description of my workflow",
  "definition": {
    "nodes": [
      {
        "id": "input1",
        "type": "file_input",
        "data": {
          "file_type": "fasta",
          "label": "Protein Sequence"
        },
        "position": {"x": 100, "y": 100}
      },
      {
        "id": "alphafold1",
        "type": "alphafold",
        "data": {
          "model": "alphafold2",
          "max_template_date": "2022-01-01"
        },
        "position": {"x": 300, "y": 100}
      }
    ],
    "edges": [
      {
        "id": "edge1",
        "source": "input1",
        "target": "alphafold1",
        "sourceHandle": "output",
        "targetHandle": "sequence"
      }
    ]
  },
  "tags": ["protein", "alphafold"],
  "status": "draft"
}
```

### Execute Workflow

```http
POST /workflows/{workflow_id}/execute
```

**Request Body:**
```json
{
  "platform": "galaxy",
  "parameters": {
    "input_files": ["file_id_123"],
    "custom_param": "value"
  }
}
```

**Response:**
```json
{
  "job_id": 456,
  "status": "pending",
  "platform": "galaxy"
}
```

### Validate Workflow

```http
POST /workflows/{workflow_id}/validate
```

**Response:**
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": ["Node 'output1' has no connections"],
  "node_count": 3,
  "edge_count": 2,
  "estimated_runtime_minutes": 120.5
}
```

## Jobs API

### List Jobs

```http
GET /jobs
```

**Query Parameters:**
- `workflow_id` (int): Filter by workflow ID
- `status` (string): Filter by job status
- `platform` (string): Filter by execution platform
- `skip`, `limit`: Pagination parameters

### Get Job Details

```http
GET /jobs/{job_id}
```

**Response:**
```json
{
  "id": 456,
  "workflow_id": 123,
  "platform": "galaxy",
  "status": "running",
  "external_job_id": "galaxy_job_789",
  "parameters": {...},
  "started_at": "2024-01-01T10:00:00Z",
  "results": {...},
  "job_steps": [
    {
      "id": 1,
      "step_name": "sequence_input",
      "status": "completed",
      "duration_seconds": 5.2
    }
  ]
}
```

### Cancel Job

```http
POST /jobs/{job_id}/cancel
```

### Get Job Logs

```http
GET /jobs/{job_id}/logs
```

**Response:**
```json
{
  "logs": "Job started at 10:00:00\nProcessing input files...\n"
}
```

## Platforms API

### List Available Platforms

```http
GET /platforms
```

**Response:**
```json
[
  {
    "name": "galaxy",
    "status": "available",
    "authenticated": true,
    "description": "Open-source computational biology platform"
  },
  {
    "name": "nextflow",
    "status": "available",
    "authenticated": false,
    "description": "Workflow management system for bioinformatics"
  }
]
```

### Check Platform Status

```http
GET /platforms/{platform_name}/status
```

### Authenticate with Platform

```http
POST /platforms/{platform_name}/authenticate
```

**Request Body:**
```json
{
  "api_key": "your-platform-api-key",
  "base_url": "https://custom-galaxy-instance.org" // optional
}
```

### List Platform Tools

```http
GET /platforms/{platform_name}/tools
```

## Files API

### Upload File

```http
POST /files/upload
```

**Request:** Multipart form data with file upload

**Response:**
```json
{
  "file_id": 789,
  "filename": "protein.fasta",
  "file_size": 1024,
  "file_type": "fasta",
  "message": "File uploaded successfully"
}
```

### List Files

```http
GET /files
```

**Query Parameters:**
- `workflow_id` (int): Filter by workflow
- `file_type` (string): Filter by file type
- `skip`, `limit`: Pagination

### Download File

```http
GET /files/{file_id}/download
```

**Response:**
```json
{
  "download_url": "https://signed-s3-url.amazonaws.com/...",
  "filename": "protein.fasta",
  "expires_at": "2024-01-01T15:00:00Z"
}
```

## Scientific Node Types

### Supported Node Types

1. **Input Nodes**
   - `file_input`: File upload input
   - `text_input`: Text parameter input
   - `number_input`: Numeric parameter input

2. **Processing Nodes**
   - `alphafold`: AlphaFold protein structure prediction
   - `blast`: BLAST sequence search
   - `pymol`: PyMOL visualization
   - `custom_script`: Custom Python/R/shell script

3. **Output Nodes**
   - `file_output`: File download output
   - `visualization`: Results visualization
   - `report`: Analysis report generation

### Node Configuration Schema

Each node type has specific configuration parameters:

```json
{
  "alphafold": {
    "model": "alphafold2|alphafold3",
    "max_template_date": "YYYY-MM-DD",
    "num_multimer_predictions_per_model": 5,
    "use_gpu": true
  },
  "blast": {
    "database": "nr|nt|pdb|swissprot",
    "evalue": 0.001,
    "max_target_seqs": 100
  },
  "pymol": {
    "representation": "cartoon|surface|sticks",
    "color_scheme": "spectrum|rainbow|element",
    "output_format": "png|pdf|pse"
  }
}
```

## Error Handling

All API endpoints return standard HTTP status codes:

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

Error response format:
```json
{
  "detail": "Error message",
  "error_code": "VALIDATION_ERROR",
  "field_errors": {
    "field_name": ["Field-specific error message"]
  }
}
```

## Rate Limiting

API requests are limited to:
- 1000 requests per hour per user
- 100 workflow executions per day per user
- 10 GB file uploads per day per user
