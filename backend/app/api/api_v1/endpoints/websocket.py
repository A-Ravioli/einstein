"""
WebSocket endpoints for real-time monitoring
"""

import uuid
from fastapi import APIRouter, WebSocket, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.websockets.monitoring_websocket import monitoring_websocket
from app.websockets.connection_manager import connection_manager

router = APIRouter()


@router.websocket("/monitor")
async def websocket_monitor_endpoint(
    websocket: WebSocket,
    client_id: str = None,
    db: AsyncSession = Depends(get_db)
):
    """WebSocket endpoint for real-time workflow and job monitoring"""
    # Generate client ID if not provided
    if not client_id:
        client_id = str(uuid.uuid4())
    
    await monitoring_websocket.handle_connection(websocket, client_id, db)


@router.get("/connections/stats")
async def get_connection_stats():
    """Get WebSocket connection statistics"""
    return {
        "active_connections": connection_manager.get_connection_count(),
        "subscription_stats": connection_manager.get_subscription_stats(),
        "total_topics": len(connection_manager.subscriptions)
    }


@router.post("/broadcast/test")
async def test_broadcast(message: str, topic: str = None):
    """Test endpoint for broadcasting messages (development only)"""
    test_message = {
        "type": "test_broadcast",
        "message": message,
        "timestamp": "now"
    }
    
    if topic:
        await connection_manager.broadcast_to_topic(test_message, topic)
        return {"status": "Message broadcasted to topic", "topic": topic}
    else:
        await connection_manager.broadcast_to_all(test_message)
        return {"status": "Message broadcasted to all clients"}
