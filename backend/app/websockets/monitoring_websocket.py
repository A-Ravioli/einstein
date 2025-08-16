"""
WebSocket endpoint for real-time workflow and job monitoring
"""

import json
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.websockets.connection_manager import connection_manager
from app.core.database import get_db
from app.services.workflow_service import WorkflowService
from app.services.job_service import JobService
from app.services.workflow_execution_service import WorkflowExecutionService
from loguru import logger


class MonitoringWebSocket:
    """WebSocket handler for monitoring workflow and job updates"""
    
    def __init__(self):
        self.connection_manager = connection_manager
    
    async def handle_connection(self, websocket: WebSocket, client_id: str, db: AsyncSession):
        """Handle WebSocket connection for monitoring"""
        try:
            await self.connection_manager.connect(
                websocket, 
                client_id,
                metadata={"type": "monitoring", "connected_at": "now"}
            )
            
            # Main message loop
            while True:
                try:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    await self._handle_message(message, client_id, db)
                
                except WebSocketDisconnect:
                    logger.info(f"Client {client_id} disconnected")
                    break
                except json.JSONDecodeError:
                    await self._send_error(client_id, "Invalid JSON message")
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
                    await self._send_error(client_id, f"Error processing message: {str(e)}")
        
        except Exception as e:
            logger.error(f"WebSocket connection error for {client_id}: {e}")
        finally:
            self.connection_manager.disconnect(client_id)
    
    async def _handle_message(self, message: Dict[str, Any], client_id: str, db: AsyncSession):
        """Handle incoming WebSocket message"""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            await self._handle_subscribe(message, client_id)
        elif message_type == "unsubscribe":
            await self._handle_unsubscribe(message, client_id)
        elif message_type == "get_status":
            await self._handle_get_status(message, client_id, db)
        elif message_type == "ping":
            await self._handle_ping(client_id)
        else:
            await self._send_error(client_id, f"Unknown message type: {message_type}")
    
    async def _handle_subscribe(self, message: Dict[str, Any], client_id: str):
        """Handle subscription request"""
        try:
            topic = message.get("topic")
            if not topic:
                await self._send_error(client_id, "Topic is required for subscription")
                return
            
            self.connection_manager.subscribe(client_id, topic)
            
            await self.connection_manager.send_personal_message({
                "type": "subscription_confirmed",
                "topic": topic,
                "message": f"Subscribed to {topic}"
            }, client_id)
        
        except Exception as e:
            await self._send_error(client_id, f"Subscription failed: {str(e)}")
    
    async def _handle_unsubscribe(self, message: Dict[str, Any], client_id: str):
        """Handle unsubscription request"""
        try:
            topic = message.get("topic")
            if not topic:
                await self._send_error(client_id, "Topic is required for unsubscription")
                return
            
            self.connection_manager.unsubscribe(client_id, topic)
            
            await self.connection_manager.send_personal_message({
                "type": "unsubscription_confirmed",
                "topic": topic,
                "message": f"Unsubscribed from {topic}"
            }, client_id)
        
        except Exception as e:
            await self._send_error(client_id, f"Unsubscription failed: {str(e)}")
    
    async def _handle_get_status(self, message: Dict[str, Any], client_id: str, db: AsyncSession):
        """Handle status request"""
        try:
            request_type = message.get("request_type")
            
            if request_type == "workflow":
                workflow_id = message.get("workflow_id")
                if workflow_id:
                    await self._send_workflow_status(workflow_id, client_id, db)
                else:
                    await self._send_error(client_id, "workflow_id is required")
            
            elif request_type == "job":
                job_id = message.get("job_id")
                if job_id:
                    await self._send_job_status(job_id, client_id, db)
                else:
                    await self._send_error(client_id, "job_id is required")
            
            elif request_type == "task":
                task_id = message.get("task_id")
                if task_id:
                    await self._send_task_status(task_id, client_id, db)
                else:
                    await self._send_error(client_id, "task_id is required")
            
            else:
                await self._send_error(client_id, f"Unknown request type: {request_type}")
        
        except Exception as e:
            await self._send_error(client_id, f"Status request failed: {str(e)}")
    
    async def _handle_ping(self, client_id: str):
        """Handle ping message"""
        await self.connection_manager.send_personal_message({
            "type": "pong",
            "timestamp": "now"
        }, client_id)
    
    async def _send_workflow_status(self, workflow_id: int, client_id: str, db: AsyncSession):
        """Send workflow status"""
        try:
            workflow_service = WorkflowService(db)
            workflow = await workflow_service.get_workflow(workflow_id)
            
            if workflow:
                await self.connection_manager.send_personal_message({
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "status": workflow.status,
                    "name": workflow.name,
                    "updated_at": workflow.updated_at
                }, client_id)
            else:
                await self._send_error(client_id, f"Workflow {workflow_id} not found")
        
        except Exception as e:
            await self._send_error(client_id, f"Error getting workflow status: {str(e)}")
    
    async def _send_job_status(self, job_id: int, client_id: str, db: AsyncSession):
        """Send job status"""
        try:
            job_service = JobService(db)
            job = await job_service.get_job(job_id)
            
            if job:
                await self.connection_manager.send_personal_message({
                    "type": "job_status",
                    "job_id": job_id,
                    "status": job.status,
                    "workflow_id": job.workflow_id,
                    "platform": job.platform,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "duration_seconds": job.duration_seconds,
                    "error_message": job.error_message
                }, client_id)
            else:
                await self._send_error(client_id, f"Job {job_id} not found")
        
        except Exception as e:
            await self._send_error(client_id, f"Error getting job status: {str(e)}")
    
    async def _send_task_status(self, task_id: str, client_id: str, db: AsyncSession):
        """Send Celery task status"""
        try:
            execution_service = WorkflowExecutionService(db)
            status = await execution_service.get_execution_status(task_id)
            
            await self.connection_manager.send_personal_message({
                "type": "task_status",
                "task_id": task_id,
                "state": status.get("state"),
                "info": status.get("info", {}),
                "ready": status.get("ready"),
                "successful": status.get("successful"),
                "failed": status.get("failed")
            }, client_id)
        
        except Exception as e:
            await self._send_error(client_id, f"Error getting task status: {str(e)}")
    
    async def _send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        await self.connection_manager.send_personal_message({
            "type": "error",
            "message": error_message
        }, client_id)


# Global monitoring WebSocket handler
monitoring_websocket = MonitoringWebSocket()


# Utility functions for broadcasting updates
async def broadcast_workflow_update(workflow_id: int, status: str, message: str = None):
    """Broadcast workflow status update"""
    await connection_manager.broadcast_to_topic({
        "type": "workflow_update",
        "workflow_id": workflow_id,
        "status": status,
        "message": message,
        "timestamp": "now"
    }, f"workflow_{workflow_id}")


async def broadcast_job_update(job_id: int, status: str, workflow_id: int = None, message: str = None):
    """Broadcast job status update"""
    update_message = {
        "type": "job_update",
        "job_id": job_id,
        "status": status,
        "message": message,
        "timestamp": "now"
    }
    
    if workflow_id:
        update_message["workflow_id"] = workflow_id
    
    # Broadcast to job-specific topic
    await connection_manager.broadcast_to_topic(update_message, f"job_{job_id}")
    
    # Also broadcast to workflow topic if available
    if workflow_id:
        await connection_manager.broadcast_to_topic(update_message, f"workflow_{workflow_id}")


async def broadcast_task_update(task_id: str, state: str, info: Dict[str, Any] = None):
    """Broadcast Celery task update"""
    await connection_manager.broadcast_to_topic({
        "type": "task_update",
        "task_id": task_id,
        "state": state,
        "info": info or {},
        "timestamp": "now"
    }, f"task_{task_id}")


async def broadcast_platform_update(platform: str, status: str, message: str = None):
    """Broadcast platform status update"""
    await connection_manager.broadcast_to_topic({
        "type": "platform_update",
        "platform": platform,
        "status": status,
        "message": message,
        "timestamp": "now"
    }, f"platform_{platform}")


async def broadcast_system_notification(notification_type: str, message: str, level: str = "info"):
    """Broadcast system-wide notification"""
    await connection_manager.broadcast_to_all({
        "type": "system_notification",
        "notification_type": notification_type,
        "message": message,
        "level": level,
        "timestamp": "now"
    })
