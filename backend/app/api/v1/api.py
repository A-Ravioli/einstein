from fastapi import APIRouter
from app.api.v1.endpoints import research, literature, hypotheses, experiments, auth

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(research.router, prefix="/research", tags=["research"])
api_router.include_router(literature.router, prefix="/literature", tags=["literature"])
api_router.include_router(hypotheses.router, prefix="/hypotheses", tags=["hypotheses"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["experiments"]) 