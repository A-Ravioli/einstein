from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import structlog
from datetime import datetime

from app.schemas.research import (
    WorkflowRequest, WorkflowResponse, 
    LiteratureReviewRequest, HypothesisGenerationRequest,
    ExperimentDesignRequest, ResearchUpdateRequest
)
from app.services.ai_services import AIServicesManager

logger = structlog.get_logger()
router = APIRouter()


async def get_ai_manager() -> AIServicesManager:
    """Dependency to get AI services manager"""
    # In a real app, this would be injected from the app state
    # For now, we'll create a mock manager
    from app.main import app
    return app.state.ai_manager


@router.post("/workflow", response_model=Dict[str, Any])
async def start_research_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """
    Start a complete AI-powered research workflow.
    This is the main endpoint that orchestrates the multi-agent system.
    """
    try:
        logger.info("Starting research workflow", goal=request.research_goal)
        
        # Start collaborative workflow
        workflow_result = await ai_manager.collaborative_workflow(
            research_goal=request.research_goal,
            user_preferences=request.user_preferences
        )
        
        return {
            "workflow_id": workflow_result.get("workflow_id"),
            "status": workflow_result.get("status", "started"),
            "message": "Research workflow initiated successfully",
            "research_goal": request.research_goal,
            "estimated_completion": "5-10 minutes",
            "steps_planned": [
                "Literature Review",
                "Hypothesis Generation", 
                "Experiment Design",
                "Analysis & Synthesis"
            ],
            "results": workflow_result if workflow_result.get("status") == "completed" else None
        }
        
    except Exception as e:
        logger.error("Workflow initiation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")


@router.post("/literature-review", response_model=Dict[str, Any])
async def conduct_literature_review(
    request: LiteratureReviewRequest,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """
    Conduct AI-powered literature review on a research topic.
    Searches multiple databases and provides analysis.
    """
    try:
        logger.info("Starting literature review", query=request.query)
        
        review_result = await ai_manager.conduct_literature_review(
            query=request.query,
            filters=request.filters
        )
        
        return {
            "status": "completed",
            "query": request.query,
            "review": review_result,
            "summary": {
                "papers_found": review_result.get("total_papers_found", 0),
                "papers_analyzed": review_result.get("papers_analyzed", 0),
                "key_findings_count": len(review_result.get("key_findings", [])),
                "research_gaps_identified": len(review_result.get("research_gaps", []))
            }
        }
        
    except Exception as e:
        logger.error("Literature review failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Literature review failed: {str(e)}")


@router.post("/generate-hypotheses", response_model=Dict[str, Any])
async def generate_research_hypotheses(
    request: HypothesisGenerationRequest,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """
    Generate research hypotheses based on research goal and literature context.
    Uses AI to create testable, novel hypotheses.
    """
    try:
        logger.info("Generating hypotheses", goal=request.research_goal)
        
        hypotheses = await ai_manager.generate_hypotheses(
            research_goal=request.research_goal,
            literature_context=request.literature_context
        )
        
        return {
            "status": "completed",
            "research_goal": request.research_goal,
            "hypotheses_generated": len(hypotheses),
            "hypotheses": hypotheses,
            "top_hypothesis": hypotheses[0] if hypotheses else None,
            "average_score": sum(h.get("overall_score", 0) for h in hypotheses) / len(hypotheses) if hypotheses else 0
        }
        
    except Exception as e:
        logger.error("Hypothesis generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Hypothesis generation failed: {str(e)}")


@router.post("/design-experiments", response_model=Dict[str, Any])
async def design_experiments(
    request: ExperimentDesignRequest,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """
    Design experiments to test a specific hypothesis.
    Creates detailed experimental protocols and procedures.
    """
    try:
        logger.info("Designing experiments", hypothesis_id=request.hypothesis_id)
        
        experiments = await ai_manager.design_experiments(
            hypothesis=request.hypothesis,
            constraints=request.constraints
        )
        
        return {
            "status": "completed",
            "hypothesis_id": request.hypothesis_id,
            "experiments_designed": len(experiments),
            "experiments": experiments,
            "computational_experiments": [e for e in experiments if e.get("type") == "computational"],
            "lab_experiments": [e for e in experiments if e.get("type") in ["wet_lab", "field_study"]]
        }
        
    except Exception as e:
        logger.error("Experiment design failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Experiment design failed: {str(e)}")


@router.post("/execute-experiment", response_model=Dict[str, Any])
async def execute_computational_experiment(
    experiment_id: str,
    experiment_data: Dict[str, Any],
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """
    Execute a computational experiment.
    Only works for computational experiments that can be automated.
    """
    try:
        logger.info("Executing experiment", experiment_id=experiment_id)
        
        execution_result = await ai_manager.execute_computational_experiment(experiment_data)
        
        return {
            "status": "completed",
            "experiment_id": experiment_id,
            "execution": execution_result,
            "can_analyze": execution_result.get("status") == "completed"
        }
        
    except Exception as e:
        logger.error("Experiment execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Experiment execution failed: {str(e)}")


@router.post("/analyze-results", response_model=Dict[str, Any])
async def analyze_experiment_results(
    experiment_data: Dict[str, Any],
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """
    Analyze experimental results using AI.
    Provides statistical analysis and interpretation.
    """
    try:
        experiment_id = experiment_data.get("experiment_id", "unknown")
        logger.info("Analyzing results", experiment_id=experiment_id)
        
        analysis_result = await ai_manager.analyze_results(experiment_data)
        
        return {
            "status": "completed",
            "experiment_id": experiment_id,
            "analysis": analysis_result
        }
        
    except Exception as e:
        logger.error("Results analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Results analysis failed: {str(e)}")


@router.post("/research-updates", response_model=Dict[str, Any])
async def get_research_updates(
    request: ResearchUpdateRequest,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """
    Get personalized research updates based on user interests.
    Keeps researchers informed about recent developments.
    """
    try:
        logger.info("Generating research updates", interests=request.interests)
        
        updates = await ai_manager.generate_research_update(
            user_interests=request.interests,
            timeframe=request.timeframe
        )
        
        return {
            "status": "completed",
            "timeframe": request.timeframe,
            "interests": request.interests,
            "updates": updates,
            "total_new_papers": sum(u.get("new_papers_count", 0) for u in updates.get("updates", [])),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Research updates failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Research updates failed: {str(e)}")


@router.post("/writing-assistance", response_model=Dict[str, Any])
async def get_writing_assistance(
    content_type: str,
    context: Dict[str, Any],
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """
    Get AI assistance with scientific writing.
    Supports various content types like abstracts, introductions, etc.
    """
    try:
        logger.info("Providing writing assistance", content_type=content_type)
        
        writing_result = await ai_manager.assist_with_writing(
            content_type=content_type,
            context=context
        )
        
        return {
            "status": "completed",
            "content_type": content_type,
            "assistance": writing_result
        }
        
    except Exception as e:
        logger.error("Writing assistance failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Writing assistance failed: {str(e)}")


@router.get("/workflow-status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    Get the status of a running research workflow.
    """
    # In a real implementation, this would check the actual workflow status
    # For demo purposes, return a mock status
    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "progress": 100,
        "current_step": "Analysis Complete",
        "estimated_remaining": "0 minutes"
    } 