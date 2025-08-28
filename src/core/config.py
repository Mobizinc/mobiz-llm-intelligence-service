"""
Configuration settings for the LLM Intelligence Service
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    app_name: str = "Mobiz LLM Intelligence Service"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Azure settings
    azure_key_vault_url: Optional[str] = None
    applicationinsights_connection_string: Optional[str] = None
    
    # API keys (will be overridden by Key Vault in production)
    openai_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    slack_signing_secret: Optional[str] = None
    slack_bot_token: Optional[str] = None
    
    # AI settings
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 500
    openai_timeout: int = 10
    default_llm_provider: str = "openai"
    default_model: str = "gpt-4"
    enable_fallback: bool = True
    request_timeout: int = 300
    max_retries: int = 3
    
    # CORS settings
    cors_origins: list = ["*"]  # Configure appropriately for production
    cors_methods: list = ["GET", "POST"]
    cors_headers: list = ["*"]
    
    # Logging settings
    log_level: str = "INFO"
    enable_slack_debug: bool = False
    enable_openai_debug: bool = False
    enable_langchain_debug: bool = False
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "mobiz-llm-intelligence"
    langsmith_tracing: bool = False
    
    # LangGraph/LangSmith
    langchain_project: str = "mobiz-llm-intelligence"
    environment: str = "development"
    service_port: int = 8000
    service_host: str = "0.0.0.0"
    
    # MCP (Model Context Protocol) settings
    mcp_config_path: str = "config/mcp_servers.json"
    mcp_global_enabled: bool = True
    mcp_tool_refresh_interval: int = 3600  # 1 hour
    mcp_dynamic_discovery: bool = True
    mcp_max_concurrent_requests: int = 10
    mcp_default_timeout: int = 30
    mcp_debug_logging: bool = False
    mcp_refresh_interval: int = 300
    
    # Azure Services (optional)
    azure_keyvault_url: Optional[str] = None
    azure_table_storage_account: Optional[str] = None
    azure_table_storage_key: Optional[str] = None
    
    # Caching
    redis_url: str = "redis://localhost:6379"
    cache_ttl_hours: int = 2
    
    # Additional settings
    max_conversation_turns: int = 20
    conversation_timeout_minutes: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra environment variables


# Global settings instance
settings = Settings()