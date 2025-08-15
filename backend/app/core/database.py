from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import event
import structlog
from app.core.config import get_settings

logger = structlog.get_logger()

class Base(DeclarativeBase):
    """Base class for all database models"""
    pass

# Global variables for database
engine = None
AsyncSessionLocal = None

async def init_db():
    """Initialize database connection and create tables"""
    global engine, AsyncSessionLocal
    
    settings = get_settings()
    
    logger.info("Initializing database connection", url=settings.async_database_url.split('@')[1])
    
    engine = create_async_engine(
        settings.async_database_url,
        echo=settings.DEBUG,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    
    AsyncSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized successfully")

async def get_db() -> AsyncSession:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def close_db():
    """Close database connection"""
    global engine
    if engine:
        await engine.dispose()
        logger.info("Database connection closed") 