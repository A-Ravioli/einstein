"""
Configuration settings for Einstein Scientific Workflow Builder
"""

from typing import List, Optional
from pydantic import BaseSettings, AnyHttpUrl, validator
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Einstein Scientific Workflow Builder"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # Next.js development
        "http://localhost:3001",  # Alternative frontend port
        "https://localhost:3000",
    ]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost/einstein_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Authentication
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Storage
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "einstein-workflows"
    
    # External Platform APIs
    GALAXY_API_KEY: Optional[str] = None
    GALAXY_BASE_URL: str = "https://usegalaxy.org"
    
    NEXTFLOW_TOWER_TOKEN: Optional[str] = None
    NEXTFLOW_TOWER_URL: str = "https://api.tower.nf"
    
    SEVEN_BRIDGES_TOKEN: Optional[str] = None
    SEVEN_BRIDGES_URL: str = "https://api.sbgenomics.com/v2"
    
    DNA_NEXUS_TOKEN: Optional[str] = None
    DNA_NEXUS_URL: str = "https://api.dnanexus.com"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
