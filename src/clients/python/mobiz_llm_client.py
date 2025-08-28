"""
Mobiz LLM Intelligence Service - Python Client Library
======================================================
Client library for integrating with the LLM Intelligence microservice.
Provides typed interfaces for all service capabilities.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field

import httpx
import asyncio_compat  # For backward compatibility

logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """Supported platforms"""
    POWER_PLATFORM = "power_platform"
    SERVICENOW = "servicenow"
    SALESFORCE = "salesforce"
    DYNAMICS365 = "dynamics365"
    GENERIC = "generic"


class AnalysisType(str, Enum):
    """Types of analysis"""
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_LEARNING = "pattern_learning"
    REMEDIATION_GENERATION = "remediation_generation"
    CROSS_PLATFORM_CORRELATION = "cross_platform_correlation"
    BUSINESS_IMPACT_ANALYSIS = "business_impact_analysis"


@dataclass
class AnalysisResult:
    """Result from analysis operation"""
    analysis_id: str
    conversation_id: str
    status: str
    platform: str
    analysis_type: str
    anomalies_found: int
    patterns_applied: int
    remediation_options: int
    confidence_score: float
    processing_time_ms: int
    response: str
    metadata: Dict[str, Any]
    
    @property
    def success(self) -> bool:
        return self.status == "completed"
    
    @property
    def has_anomalies(self) -> bool:
        return self.anomalies_found > 0
    
    @property
    def has_remediations(self) -> bool:
        return self.remediation_options > 0


@dataclass
class PatternSearchResult:
    """Result from pattern search"""
    query: str
    results: List[Dict[str, Any]]
    total_found: int
    search_time_ms: int


class MobizLLMClientError(Exception):
    """Base exception for client errors"""
    pass


class MobizLLMServiceError(Exception):
    """Exception for service-side errors"""
    def __init__(self, message: str, status_code: int = None, details: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class MobizLLMClient:
    """
    Python client for Mobiz LLM Intelligence Service.
    
    Provides async methods for all service capabilities:
    - Power Platform analysis
    - Cross-platform correlation
    - Pattern search and learning
    - Remediation generation
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        timeout: int = 300,
        max_retries: int = 3,
        client_id: str = None
    ):
        """
        Initialize client.
        
        Args:
            endpoint: Service endpoint URL
            api_key: Authentication API key  
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            client_id: Client identifier for tracking
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.client_id = client_id or "mobiz-platform-intelligence"
        
        # HTTP client configuration
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"mobiz-llm-client/1.0.0 ({self.client_id})"
            },
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            response = await self.http_client.get(f"{self.endpoint}/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise MobizLLMClientError(f"Health check failed: {e}")
    
    async def analyze_power_platform(
        self,
        context_graph: Dict[str, Any],
        analysis_type: AnalysisType = AnalysisType.ANOMALY_DETECTION,
        query: str = None,
        user_id: str = "system",
        channel_id: str = "api",
        streaming: bool = False
    ) -> AnalysisResult:
        """
        Analyze Power Platform assets for anomalies and issues.
        
        Args:
            context_graph: Power Platform context data (from ContextGraph.to_dict())
            analysis_type: Type of analysis to perform
            query: Natural language query describing the issue
            user_id: User identifier
            channel_id: Channel identifier
            streaming: Enable streaming response
            
        Returns:
            AnalysisResult with detected anomalies and recommendations
        """
        request_data = {
            "client_id": self.client_id,
            "platform": PlatformType.POWER_PLATFORM.value,
            "analysis_type": analysis_type.value,
            "query": query or "Analyze this Power Platform asset for anomalies",
            "context": {
                "context_graph": context_graph,
                "entity_type": context_graph.get("entity_type", "unknown"),
                "entity_id": context_graph.get("entity_id"),
                "app_name": context_graph.get("app_name"),
                "app_id": context_graph.get("app_id")
            },
            "user_id": user_id,
            "channel_id": channel_id,
            "streaming": streaming
        }
        
        return await self._analyze(request_data)
    
    async def analyze_canvas_app(
        self,
        app_id: str,
        app_name: str,
        context_graph: Dict[str, Any],
        query: str = None
    ) -> AnalysisResult:
        """
        Specialized Canvas App analysis.
        
        Args:
            app_id: Canvas App ID
            app_name: Canvas App name
            context_graph: App structure and metadata
            query: Specific query about the app
            
        Returns:
            Analysis result with Canvas App specific insights
        """
        # Enhance context for Canvas Apps
        enhanced_context = {
            **context_graph,
            "entity_type": "canvas_app",
            "app_id": app_id,
            "app_name": app_name
        }
        
        return await self.analyze_power_platform(
            context_graph=enhanced_context,
            analysis_type=AnalysisType.ANOMALY_DETECTION,
            query=query or f"Analyze Canvas App '{app_name}' for performance and usability issues"
        )
    
    async def analyze_power_automate_flow(
        self,
        flow_id: str,
        flow_name: str,
        flow_data: Dict[str, Any],
        stuck_entities: List[Dict] = None,
        query: str = None
    ) -> AnalysisResult:
        """
        Specialized Power Automate Flow analysis.
        
        Args:
            flow_id: Flow identifier
            flow_name: Flow name
            flow_data: Flow definition and metadata
            stuck_entities: List of stuck work orders or processes
            query: Specific query about the flow
            
        Returns:
            Analysis result with Flow specific insights
        """
        context_graph = {
            "entity_type": "flow",
            "flow_id": flow_id,
            "flow_name": flow_name,
            "flow_data": flow_data,
            "stuck_orders": stuck_entities or [],
            "additional_context": {}
        }
        
        return await self.analyze_power_platform(
            context_graph=context_graph,
            analysis_type=AnalysisType.ANOMALY_DETECTION,
            query=query or f"Analyze Power Automate Flow '{flow_name}' for stuck processes and bottlenecks"
        )
    
    async def generate_remediation(
        self,
        issue_description: Dict[str, Any],
        platform: PlatformType = PlatformType.POWER_PLATFORM,
        user_id: str = "system"
    ) -> AnalysisResult:
        """
        Generate remediation scripts and actions for identified issues.
        
        Args:
            issue_description: Description of the issue to remediate
            platform: Target platform
            user_id: User requesting remediation
            
        Returns:
            Analysis result with remediation options
        """
        request_data = {
            "client_id": self.client_id,
            "platform": platform.value,
            "analysis_type": AnalysisType.REMEDIATION_GENERATION.value,
            "query": "Generate remediation for the described issue",
            "context": {
                "issue_data": issue_description,
                "remediation_request": True
            },
            "user_id": user_id,
            "channel_id": "remediation_api"
        }
        
        return await self._analyze(request_data)
    
    async def search_patterns(
        self,
        query: str,
        platform: PlatformType = None,
        pattern_type: str = None,
        limit: int = 10
    ) -> PatternSearchResult:
        """
        Search learned patterns across platforms.
        
        Args:
            query: Search query
            platform: Filter by platform
            pattern_type: Filter by pattern type
            limit: Maximum results
            
        Returns:
            Pattern search results
        """
        params = {
            "query": query,
            "limit": limit
        }
        
        if platform:
            params["platform"] = platform.value
        if pattern_type:
            params["pattern_type"] = pattern_type
        
        try:
            response = await self.http_client.get(
                f"{self.endpoint}/v1/patterns/search",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            return PatternSearchResult(
                query=query,
                results=data.get("results", []),
                total_found=data.get("total_found", 0),
                search_time_ms=data.get("search_time_ms", 0)
            )
            
        except httpx.HTTPError as e:
            raise MobizLLMClientError(f"Pattern search failed: {e}")
    
    async def get_supported_platforms(self) -> List[Dict[str, str]]:
        """Get list of supported platforms"""
        try:
            response = await self.http_client.get(f"{self.endpoint}/v1/platforms")
            response.raise_for_status()
            return response.json().get("platforms", [])
        except httpx.HTTPError as e:
            raise MobizLLMClientError(f"Failed to get platforms: {e}")
    
    async def get_analysis_types(self) -> List[Dict[str, str]]:
        """Get list of supported analysis types"""
        try:
            response = await self.http_client.get(f"{self.endpoint}/v1/analysis-types")
            response.raise_for_status()
            return response.json().get("analysis_types", [])
        except httpx.HTTPError as e:
            raise MobizLLMClientError(f"Failed to get analysis types: {e}")
    
    async def _analyze(self, request_data: Dict[str, Any]) -> AnalysisResult:
        """Internal method to perform analysis with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.http_client.post(
                    f"{self.endpoint}/v1/analyze",
                    json=request_data
                )
                response.raise_for_status()
                
                data = response.json()
                return AnalysisResult(
                    analysis_id=data["analysis_id"],
                    conversation_id=data["conversation_id"],
                    status=data["status"],
                    platform=data["platform"],
                    analysis_type=data["analysis_type"],
                    anomalies_found=data["anomalies_found"],
                    patterns_applied=data["patterns_applied"],
                    remediation_options=data["remediation_options"],
                    confidence_score=data["confidence_score"],
                    processing_time_ms=data["processing_time_ms"],
                    response=data["response"],
                    metadata=data["metadata"]
                )
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    # Server error, retry
                    last_exception = e
                    if attempt < self.max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Server error, retrying in {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Client error or final attempt
                try:
                    error_data = e.response.json()
                except:
                    error_data = {"detail": str(e)}
                
                raise MobizLLMServiceError(
                    f"Analysis failed: {error_data.get('detail', 'Unknown error')}",
                    status_code=e.response.status_code,
                    details=error_data
                )
                
            except httpx.RequestError as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request error, retrying in {wait_time}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                    continue
                
                raise MobizLLMClientError(f"Request failed after {self.max_retries} retries: {e}")
        
        # If we get here, all retries failed
        raise MobizLLMClientError(f"Analysis failed after {self.max_retries} retries: {last_exception}")


# Convenience functions for common use cases
async def analyze_power_platform_quick(
    endpoint: str,
    api_key: str,
    context_graph: Dict[str, Any],
    query: str = None
) -> AnalysisResult:
    """
    Quick analysis function for Power Platform assets.
    
    Usage:
        result = await analyze_power_platform_quick(
            endpoint="http://localhost:8000",
            api_key="your-api-key",
            context_graph=your_context_graph
        )
    """
    async with MobizLLMClient(endpoint, api_key) as client:
        return await client.analyze_power_platform(context_graph, query=query)


async def search_patterns_quick(
    endpoint: str,
    api_key: str,
    query: str,
    platform: str = None
) -> PatternSearchResult:
    """
    Quick pattern search function.
    
    Usage:
        patterns = await search_patterns_quick(
            endpoint="http://localhost:8000", 
            api_key="your-api-key",
            query="stuck approval processes"
        )
    """
    async with MobizLLMClient(endpoint, api_key) as client:
        platform_enum = PlatformType(platform) if platform else None
        return await client.search_patterns(query, platform=platform_enum)