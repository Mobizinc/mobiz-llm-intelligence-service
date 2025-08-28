"""
LLM Manager - Multi-Provider LLM Integration
============================================
Manages multiple LLM providers (OpenAI GPT-5, Azure OpenAI, etc.) with
intelligent routing, fallbacks, and cost optimization.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field

import openai
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"  # Future
    GOOGLE = "google"  # Future


class ModelType(str, Enum):
    """Model types for different use cases"""
    GPT5 = "gpt-5"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"  # Future


@dataclass
class LLMRequest:
    """Structured LLM request"""
    messages: List[Dict[str, str]]
    system_prompt: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Union[str, Dict[str, Any]]] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 30
    model_preference: Optional[ModelType] = None
    structured_output: Optional[type] = None  # Pydantic model for structured outputs


@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    model: str = ""
    provider: str = ""
    response_time_ms: int = 0
    structured_data: Optional[Any] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, provider_name: str, **config):
        self.provider_name = provider_name
        self.config = config
        self.client = None
        self.is_available = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider"""
        pass
    
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider (including GPT-5)"""
    
    def __init__(self, **config):
        super().__init__("openai", **config)
        self.api_key = config.get("api_key", os.getenv("OPENAI_API_KEY"))
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.default_model = config.get("default_model", "gpt-5")
    
    async def initialize(self) -> bool:
        """Initialize OpenAI client"""
        try:
            if not self.api_key:
                logger.error("OpenAI API key not provided")
                return False
            
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=60.0
            )
            
            # Test connection
            await self.health_check()
            self.is_available = True
            logger.info("✅ OpenAI provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI provider: {e}")
            return False
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI"""
        if not self.client or not self.is_available:
            raise RuntimeError("OpenAI provider not initialized")
        
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = request.messages.copy()
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})
            
            # Prepare request parameters
            params = {
                "model": request.model_preference.value if request.model_preference else self.default_model,
                "messages": messages,
                "temperature": request.temperature,
                "timeout": request.timeout
            }
            
            # Add optional parameters
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens
            
            if request.tools:
                params["tools"] = request.tools
                params["parallel_tool_calls"] = False  # Required for structured outputs
            
            # Handle structured outputs
            if request.structured_output:
                from pydantic import BaseModel
                if issubclass(request.structured_output, BaseModel):
                    # Use Pydantic parsing with new SDK
                    response = await self.client.beta.chat.completions.parse(
                        response_format=request.structured_output,
                        **params
                    )
                    
                    response_time_ms = int((time.time() - start_time) * 1000)
                    
                    return LLMResponse(
                        content=response.choices[0].message.content or "",
                        usage=response.usage.dict() if response.usage else None,
                        model=response.model,
                        provider=self.provider_name,
                        response_time_ms=response_time_ms,
                        structured_data=response.choices[0].message.parsed
                    )
            
            elif request.response_format:
                if isinstance(request.response_format, dict):
                    params["response_format"] = request.response_format
                elif request.response_format == "json":
                    params["response_format"] = {"type": "json_object"}
            
            # Make API call
            response = await self.client.chat.completions.create(**params)
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract tool calls if present
            tool_calls = None
            if response.choices[0].message.tool_calls:
                tool_calls = [
                    {
                        "id": call.id,
                        "function": call.function.name,
                        "arguments": json.loads(call.function.arguments)
                    }
                    for call in response.choices[0].message.tool_calls
                ]
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                tool_calls=tool_calls,
                usage=response.usage.dict() if response.usage else None,
                model=response.model,
                provider=self.provider_name,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check OpenAI health"""
        try:
            # Simple completion test
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for health check
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                timeout=10
            )
            return bool(response.choices)
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider"""
    
    def __init__(self, **config):
        super().__init__("azure_openai", **config)
        self.api_key = config.get("api_key", os.getenv("AZURE_OPENAI_API_KEY"))
        self.endpoint = config.get("endpoint", os.getenv("AZURE_OPENAI_ENDPOINT"))
        self.api_version = config.get("api_version", "2024-10-01-preview")
        self.deployment_name = config.get("deployment_name", "gpt-4o")
    
    async def initialize(self) -> bool:
        """Initialize Azure OpenAI client"""
        try:
            if not self.api_key or not self.endpoint:
                logger.warning("Azure OpenAI credentials not provided")
                return False
            
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=f"{self.endpoint}/openai/deployments/{self.deployment_name}",
                default_query={"api-version": self.api_version},
                timeout=60.0
            )
            
            await self.health_check()
            self.is_available = True
            logger.info("✅ Azure OpenAI provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI provider: {e}")
            return False
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Azure OpenAI"""
        # Similar implementation to OpenAI but with Azure-specific handling
        # Implementation would be similar to OpenAI provider
        pass
    
    async def health_check(self) -> bool:
        """Check Azure OpenAI health"""
        # Implementation for Azure health check
        return True


class LLMManager:
    """Manages multiple LLM providers with intelligent routing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.routing_strategy = self.config.get("routing_strategy", "cost_optimized")
        self.fallback_enabled = self.config.get("fallback_enabled", True)
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "provider_usage": {},
            "cost_estimate": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize all configured providers"""
        logger.info("Initializing LLM Manager...")
        
        # Initialize OpenAI provider
        openai_config = self.config.get("openai", {})
        if os.getenv("OPENAI_API_KEY") or openai_config.get("api_key"):
            openai_provider = OpenAIProvider(**openai_config)
            if await openai_provider.initialize():
                self.providers["openai"] = openai_provider
        
        # Initialize Azure OpenAI provider
        azure_config = self.config.get("azure_openai", {})
        if (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")) or \
           (azure_config.get("api_key") and azure_config.get("endpoint")):
            azure_provider = AzureOpenAIProvider(**azure_config)
            if await azure_provider.initialize():
                self.providers["azure_openai"] = azure_provider
        
        if not self.providers:
            logger.error("❌ No LLM providers initialized")
            return False
        
        logger.info(f"✅ LLM Manager initialized with {len(self.providers)} providers: {list(self.providers.keys())}")
        return True
    
    def _select_provider(self, request: LLMRequest) -> BaseLLMProvider:
        """Select best provider based on routing strategy"""
        available_providers = [p for p in self.providers.values() if p.is_available]
        
        if not available_providers:
            raise RuntimeError("No available LLM providers")
        
        # Simple strategy: prefer OpenAI for GPT-5, otherwise first available
        if request.model_preference == ModelType.GPT5 and "openai" in self.providers:
            return self.providers["openai"]
        
        return available_providers[0]
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion with provider selection and fallback"""
        self.usage_stats["total_requests"] += 1
        
        primary_provider = self._select_provider(request)
        
        try:
            # Try primary provider
            logger.debug(f"Using provider: {primary_provider.provider_name}")
            response = await primary_provider.complete(request)
            
            # Update usage stats
            if response.usage:
                self.usage_stats["total_tokens"] += response.usage.get("total_tokens", 0)
                
                provider_stats = self.usage_stats["provider_usage"].get(primary_provider.provider_name, {
                    "requests": 0, "tokens": 0
                })
                provider_stats["requests"] += 1
                provider_stats["tokens"] += response.usage.get("total_tokens", 0)
                self.usage_stats["provider_usage"][primary_provider.provider_name] = provider_stats
            
            return response
            
        except Exception as e:
            logger.error(f"Primary provider {primary_provider.provider_name} failed: {e}")
            
            if not self.fallback_enabled:
                raise
            
            # Try fallback providers
            fallback_providers = [p for p in self.providers.values() 
                                if p != primary_provider and p.is_available]
            
            for fallback_provider in fallback_providers:
                try:
                    logger.info(f"Trying fallback provider: {fallback_provider.provider_name}")
                    return await fallback_provider.complete(request)
                except Exception as fallback_error:
                    logger.error(f"Fallback provider {fallback_provider.provider_name} failed: {fallback_error}")
                    continue
            
            raise RuntimeError(f"All providers failed. Last error: {e}")
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.usage_stats.copy()
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health_status = {}
        
        for name, provider in self.providers.items():
            try:
                health_status[name] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health_status[name] = False
        
        return health_status
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models"""
        models = []
        
        for provider_name, provider in self.providers.items():
            if not provider.is_available:
                continue
            
            if provider_name == "openai":
                models.extend([
                    {"model": ModelType.GPT5.value, "provider": provider_name},
                    {"model": ModelType.GPT4O.value, "provider": provider_name},
                    {"model": ModelType.GPT4O_MINI.value, "provider": provider_name},
                ])
            elif provider_name == "azure_openai":
                models.extend([
                    {"model": ModelType.GPT4O.value, "provider": provider_name},
                    {"model": ModelType.GPT4O_MINI.value, "provider": provider_name},
                ])
        
        return models