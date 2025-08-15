from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import structlog

from app.services.ai_services import AIServicesManager

router = APIRouter()
logger = structlog.get_logger()


async def get_ai_manager() -> AIServicesManager:
    """Dependency to get AI services manager"""
    from app.main import app
    return app.state.ai_manager


@router.post("/generate")
async def generate_hypotheses(
    research_goal: str,
    literature_context: Dict[str, Any] = None,
    num_hypotheses: int = 5,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Generate research hypotheses"""
    try:
        hypotheses = await ai_manager.generate_hypotheses(
            research_goal=research_goal,
            literature_context=literature_context
        )
        
        return {
            "research_goal": research_goal,
            "hypotheses": hypotheses,
            "count": len(hypotheses),
            "top_scored": max(hypotheses, key=lambda x: x.get("overall_score", 0)) if hypotheses else None
        }
        
    except Exception as e:
        logger.error("Hypothesis generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{hypothesis_id}/refine")
async def refine_hypothesis(
    hypothesis_id: str,
    hypothesis: Dict[str, Any],
    feedback: str,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Refine a hypothesis based on feedback"""
    try:
        # In a real implementation, we'd retrieve the hypothesis from the database
        # For demo, we'll use the provided hypothesis data
        
        refined = await ai_manager.agents['hypothesis'].refine_hypothesis(
            hypothesis=hypothesis,
            feedback=feedback
        )
        
        return {
            "hypothesis_id": hypothesis_id,
            "original": hypothesis,
            "refined": refined,
            "feedback_applied": feedback
        }
        
    except Exception as e:
        logger.error("Hypothesis refinement failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_hypotheses(
    hypotheses: List[Dict[str, Any]],
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Compare multiple hypotheses"""
    try:
        comparison = await ai_manager.agents['hypothesis'].compare_hypotheses(hypotheses)
        
        return {
            "hypotheses_compared": len(hypotheses),
            "comparison": comparison,
            "recommendation": comparison.get("recommended_prioritization", [])
        }
        
    except Exception as e:
        logger.error("Hypothesis comparison failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{hypothesis_id}/questions")
async def generate_research_questions(
    hypothesis_id: str,
    hypothesis: Dict[str, Any],
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Generate research questions for a hypothesis"""
    try:
        questions = await ai_manager.agents['hypothesis'].generate_research_questions(hypothesis)
        
        return {
            "hypothesis_id": hypothesis_id,
            "research_questions": questions,
            "count": len(questions)
        }
        
    except Exception as e:
        logger.error("Research questions generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) 