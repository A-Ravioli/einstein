"""Configuration handling for the Einstein system."""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration handler for Einstein system."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional dictionary.
        
        Args:
            config_dict: Optional dictionary with configuration values
        """
        self._config = config_dict or {}
        
        # Load default settings from environment
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration from environment variables."""
        # API keys
        self._config.setdefault("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        
        # Model settings
        self._config.setdefault("default_model", os.environ.get("EINSTEIN_DEFAULT_MODEL", "gpt-4-turbo"))
        
        # System settings
        self._config.setdefault("context_memory_path", os.environ.get("EINSTEIN_CONTEXT_MEMORY_PATH", "context_memory.json"))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
        """
        self._config.update(config_dict)
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.get("openai_api_key")
    
    @property
    def default_model(self) -> str:
        """Get default model name."""
        return self.get("default_model")
    
    @property
    def context_memory_path(self) -> str:
        """Get context memory file path."""
        return self.get("context_memory_path")


# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance."""
    return config

def configure(config_dict: Dict[str, Any]) -> None:
    """
    Update global configuration.
    
    Args:
        config_dict: Dictionary with configuration values
    """
    config.update(config_dict) 