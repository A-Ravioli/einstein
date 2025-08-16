"""
Celery worker configuration for background tasks
"""

from celery import Celery
from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "einstein_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.workflow_tasks", "app.tasks.platform_tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Task routing
    task_routes={
        "app.tasks.workflow_tasks.*": {"queue": "workflow_queue"},
        "app.tasks.platform_tasks.*": {"queue": "platform_queue"},
    },
    # Result expiration
    result_expires=3600,
)
