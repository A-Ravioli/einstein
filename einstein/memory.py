"""Persistent storage for maintaining state across operations."""
import os
import json
import time
from typing import Dict, Any, Optional

from einstein_pkg.config import get_config

class ContextMemory:
    """Persistent storage for maintaining state across operations."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize context memory with optional storage path.
        
        Args:
            storage_path: Path to storage file. If None, use default from config.
        """
        config = get_config()
        self.storage_path = storage_path or config.context_memory_path
        self.memory = self._load_memory() if os.path.exists(self.storage_path) else {}
    
    def _load_memory(self) -> Dict:
        """
        Load memory from disk.
        
        Returns:
            Dictionary with memory contents
        """
        with open(self.storage_path, 'r') as f:
            return json.load(f)
    
    def _save_memory(self):
        """Save memory to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def set(self, key: str, value: Any):
        """
        Store value in memory.
        
        Args:
            key: Memory key
            value: Value to store
        """
        self.memory[key] = value
        self._save_memory()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from memory.
        
        Args:
            key: Memory key
            default: Default value if key is not found
            
        Returns:
            Stored value or default
        """
        return self.memory.get(key, default)
    
    def update_state(self, state_updates: Dict):
        """
        Update multiple state values at once.
        
        Args:
            state_updates: Dictionary with updates
        """
        self.memory.update(state_updates)
        self._save_memory()
    
    def clear(self):
        """Clear all memory."""
        self.memory = {}
        self._save_memory()
    
    def get_timestamp(self) -> float:
        """
        Get current timestamp.
        
        Returns:
            Current timestamp
        """
        return time.time()
    
    def create_id(self, prefix: str) -> str:
        """
        Create a unique ID with prefix and timestamp.
        
        Args:
            prefix: ID prefix
            
        Returns:
            Unique ID string
        """
        return f"{prefix}_{int(self.get_timestamp())}" 