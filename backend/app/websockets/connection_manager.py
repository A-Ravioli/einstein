"""
WebSocket connection manager for real-time updates
"""

import json
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        # Store active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Store subscriptions: topic -> set of client IDs
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Store client metadata
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Connect a client"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = metadata or {}
        
        logger.info(f"Client {client_id} connected via WebSocket")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "message": "WebSocket connection established"
        }, client_id)
    
    def disconnect(self, client_id: str):
        """Disconnect a client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from all subscriptions
        for topic in list(self.subscriptions.keys()):
            if client_id in self.subscriptions[topic]:
                self.subscriptions[topic].remove(client_id)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
        
        # Remove metadata
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
        
        logger.info(f"Client {client_id} disconnected from WebSocket")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                # Remove disconnected client
                self.disconnect(client_id)
    
    async def broadcast_to_topic(self, message: Dict[str, Any], topic: str):
        """Broadcast a message to all clients subscribed to a topic"""
        if topic in self.subscriptions:
            disconnected_clients = []
            
            for client_id in self.subscriptions[topic]:
                try:
                    websocket = self.active_connections[client_id]
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            # Remove disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(client_id)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe(self, client_id: str, topic: str):
        """Subscribe a client to a topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        
        self.subscriptions[topic].add(client_id)
        logger.info(f"Client {client_id} subscribed to topic: {topic}")
    
    def unsubscribe(self, client_id: str, topic: str):
        """Unsubscribe a client from a topic"""
        if topic in self.subscriptions and client_id in self.subscriptions[topic]:
            self.subscriptions[topic].remove(client_id)
            
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
            
            logger.info(f"Client {client_id} unsubscribed from topic: {topic}")
    
    def get_topic_subscribers(self, topic: str) -> List[str]:
        """Get list of clients subscribed to a topic"""
        return list(self.subscriptions.get(topic, set()))
    
    def get_client_subscriptions(self, client_id: str) -> List[str]:
        """Get list of topics a client is subscribed to"""
        return [topic for topic, clients in self.subscriptions.items() if client_id in clients]
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_subscription_stats(self) -> Dict[str, int]:
        """Get subscription statistics"""
        return {topic: len(clients) for topic, clients in self.subscriptions.items()}


# Global connection manager instance
connection_manager = ConnectionManager()
