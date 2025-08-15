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


@router.get("/search")
async def search_literature(
    query: str,
    databases: List[str] = None,
    max_results: int = 50,
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Search literature across multiple databases"""
    try:
        filters = {"databases": databases} if databases else None
        
        results = await ai_manager.conduct_literature_review(
            query=query,
            filters=filters
        )
        
        return {
            "query": query,
            "databases_searched": databases or ["arxiv", "pubmed", "semantic_scholar"],
            "results": results
        }
        
    except Exception as e:
        logger.error("Literature search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending")
async def get_trending_topics():
    """Get trending research topics"""
    # Mock implementation for demo
    return {
        "trending_topics": [
            {
                "topic": "Large Language Models",
                "growth_rate": "+245%",
                "recent_papers": 1234,
                "key_institutions": ["OpenAI", "Google", "Anthropic"]
            },
            {
                "topic": "Protein Folding Prediction",
                "growth_rate": "+89%", 
                "recent_papers": 567,
                "key_institutions": ["DeepMind", "University of Washington"]
            },
            {
                "topic": "Quantum Computing",
                "growth_rate": "+67%",
                "recent_papers": 890,
                "key_institutions": ["IBM", "Google", "MIT"]
            }
        ],
        "timeframe": "last_30_days"
    }


@router.get("/recommendations")
async def get_paper_recommendations(
    user_interests: List[str],
    ai_manager: AIServicesManager = Depends(get_ai_manager)
):
    """Get personalized paper recommendations"""
    try:
        recommendations = await ai_manager.generate_research_update(
            user_interests=user_interests,
            timeframe="week"
        )
        
        return {
            "interests": user_interests,
            "recommendations": recommendations,
            "generated_at": recommendations.get("generated_at")
        }
        
    except Exception as e:
        logger.error("Paper recommendations failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) 