from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any
import structlog

from app.services.ai_services import AIServicesManager

router = APIRouter()
logger = structlog.get_logger()


async def get_ai_manager() -> AIServicesManager:
    """Dependency to get AI services manager"""
    from app.main import app
    return app.state.ai_manager


@router.post("/design")
async def design_experiments(
    hypothesis: Dict[str, Any],
    constraints: Dict[str, Any] = None,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Design experiments for a hypothesis"""
    try:
        experiments = await ai_manager.design_experiments(
            hypothesis=hypothesis,
            constraints=constraints
        )
        
        return {
            "hypothesis_id": hypothesis.get("id"),
            "experiments": experiments,
            "count": len(experiments),
            "types": list(set(exp.get("type") for exp in experiments))
        }
        
    except Exception as e:
        logger.error("Experiment design failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/execute")
async def execute_experiment(
    experiment_id: str,
    experiment: Dict[str, Any],
    background_tasks: BackgroundTasks,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Execute a computational experiment"""
    try:
        if experiment.get("type") != "computational":
            raise HTTPException(
                status_code=400, 
                detail="Only computational experiments can be executed automatically"
            )
        
        result = await ai_manager.execute_computational_experiment(experiment)
        
        return {
            "experiment_id": experiment_id,
            "execution_result": result,
            "status": result.get("status"),
            "can_analyze": result.get("status") == "completed"
        }
        
    except Exception as e:
        logger.error("Experiment execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{experiment_id}/analyze")
async def analyze_experiment(
    experiment_id: str,
    experiment_data: Dict[str, Any],
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Analyze experiment results"""
    try:
        analysis = await ai_manager.analyze_results(experiment_data)
        
        return {
            "experiment_id": experiment_id,
            "analysis": analysis,
            "key_findings": analysis.get("key_findings", []),
            "next_steps": analysis.get("next_steps", [])
        }
        
    except Exception as e:
        logger.error("Experiment analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_experiment_templates():
    """Get experiment design templates"""
    return {
        "templates": [
            {
                "id": "computational_analysis",
                "name": "Computational Data Analysis",
                "type": "computational",
                "description": "Template for computational data analysis experiments",
                "required_inputs": ["dataset", "analysis_method", "variables"],
                "typical_duration": "1-4 hours"
            },
            {
                "id": "simulation_study", 
                "name": "Simulation Study",
                "type": "computational",
                "description": "Template for simulation-based research",
                "required_inputs": ["model_parameters", "sample_size", "iterations"],
                "typical_duration": "2-8 hours"
            },
            {
                "id": "wet_lab_protocol",
                "name": "Wet Lab Protocol",
                "type": "wet_lab",
                "description": "Template for laboratory experiments",
                "required_inputs": ["materials", "procedures", "controls"],
                "typical_duration": "1-7 days"
            }
        ]
    }


@router.get("/{experiment_id}/status")
async def get_experiment_status(experiment_id: str):
    """Get experiment execution status"""
    # Mock implementation for demo
    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "progress": 100,
        "started_at": "2024-01-15T10:00:00Z",
        "completed_at": "2024-01-15T10:05:00Z",
        "execution_time": 300,
        "has_results": True
    } 