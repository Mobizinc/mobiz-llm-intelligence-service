"""
Mobiz LLM Intelligence Service - Main FastAPI Application
=========================================================
Universal LLM Intelligence microservice providing anomaly detection,
pattern learning, and automated remediation across enterprise platforms.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import httpx

# Import core components
from ...models.universal_state import (
    UniversalIntelligenceState,
    PlatformType, 
    AnalysisType,
    UniversalStateManager
)

# Import services (we'll create these)
from ...services.llm_manager import LLMManager
from ...services.mcp_manager import MCPManager
from ...core.conversation_graph import ConversationGraph

# Import models for API
from ...models.conversation import ConversationRequest, ConversationResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
llm_manager: Optional[LLMManager] = None
mcp_manager: Optional[MCPManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global llm_manager, mcp_manager
    
    # Startup
    logger.info("🚀 Starting Mobiz LLM Intelligence Service")
    
    try:
        # Initialize LLM Manager
        logger.info("Initializing LLM Manager...")
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        # Initialize MCP Manager
        logger.info("Initializing MCP Manager...")
        mcp_manager = MCPManager()
        await mcp_manager.initialize()
        
        logger.info("✅ Service initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Mobiz LLM Intelligence Service")
    
    if mcp_manager:
        await mcp_manager.shutdown()
    
    logger.info("✅ Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Mobiz LLM Intelligence Service",
    description="Universal LLM Intelligence microservice for enterprise platform analysis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# API Models
class AnalysisRequest(BaseModel):
    """Request model for analysis operations"""
    client_id: str = Field(..., description="Client identifier")
    platform: PlatformType = Field(..., description="Target platform")
    analysis_type: AnalysisType = Field(..., description="Type of analysis")
    query: str = Field(..., description="Natural language query")
    context: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific context")
    user_id: str = Field(..., description="User identifier")
    channel_id: str = Field(default="api", description="Channel identifier")
    tools: Optional[List[str]] = Field(default=None, description="Available tools")
    streaming: bool = Field(default=False, description="Enable streaming response")


class AnalysisResponse(BaseModel):
    """Response model for analysis operations"""
    analysis_id: str
    conversation_id: str
    status: str
    platform: PlatformType
    analysis_type: AnalysisType
    anomalies_found: int
    patterns_applied: int
    remediation_options: int
    confidence_score: float
    processing_time_ms: int
    response: str
    metadata: Dict[str, Any]


class PatternSearchRequest(BaseModel):
    """Request model for pattern search"""
    query: str = Field(..., description="Search query")
    platform: Optional[PlatformType] = Field(default=None, description="Filter by platform")
    pattern_type: Optional[str] = Field(default=None, description="Filter by pattern type")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]


# Dependency functions
async def get_llm_manager() -> LLMManager:
    """Get LLM manager instance"""
    if llm_manager is None:
        raise HTTPException(status_code=503, detail="LLM Manager not initialized")
    return llm_manager


async def get_mcp_manager() -> MCPManager:
    """Get MCP manager instance"""
    if mcp_manager is None:
        raise HTTPException(status_code=503, detail="MCP Manager not initialized")
    return mcp_manager


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        services={
            "llm_manager": "healthy" if llm_manager else "unavailable",
            "mcp_manager": "healthy" if mcp_manager else "unavailable"
        }
    )


@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    llm_mgr: LLMManager = Depends(get_llm_manager)
) -> AnalysisResponse:
    """
    Main analysis endpoint for universal intelligence operations.
    
    Supports:
    - Anomaly detection
    - Pattern learning
    - Remediation generation
    - Cross-platform correlation
    - Business impact analysis
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting analysis: {request.analysis_type.value} for {request.platform.value}")
        
        # Generate conversation ID
        conversation_id = f"conv_{request.client_id}_{int(time.time())}"
        
        # Create initial state
        initial_state = UniversalStateManager.create_initial_state(
            conversation_id=conversation_id,
            platform=request.platform,
            analysis_type=request.analysis_type,
            initial_query=request.query,
            user_id=request.user_id,
            channel_id=request.channel_id,
            platform_context=request.context
        )
        
        # Create conversation graph for the platform
        graph = ConversationGraph(
            domain=request.platform.value,
            enable_parallel_execution=True,
            enable_supervisor_mode=True,
            timeout=300  # 5 minute timeout
        )
        
        # Execute analysis workflow
        result = await graph.app.ainvoke(initial_state)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Create response
        response = AnalysisResponse(
            analysis_id=result['analysis_id'],
            conversation_id=conversation_id,
            status="completed",
            platform=request.platform,
            analysis_type=request.analysis_type,
            anomalies_found=len(result.get('detected_anomalies', [])),
            patterns_applied=len(result.get('applied_patterns', [])),
            remediation_options=len(result.get('remediation_options', [])),
            confidence_score=result.get('anomaly_confidence', 0.0),
            processing_time_ms=processing_time_ms,
            response=result.get('response', 'Analysis completed'),
            metadata={
                'business_impact': result.get('business_impact', {}),
                'quality_score': result.get('quality_score'),
                'token_usage': result.get('token_usage', {})
            }
        )
        
        logger.info(f"Analysis completed: {conversation_id} in {processing_time_ms}ms")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/v1/workflows/{workflow_id}")
async def execute_workflow(
    workflow_id: str,
    request: Dict[str, Any],
    llm_mgr: LLMManager = Depends(get_llm_manager)
):
    """Execute predefined LangGraph workflow"""
    
    # This will be implemented based on specific workflow requirements
    return {"message": f"Workflow {workflow_id} executed", "request": request}


@app.get("/v1/patterns/search")
async def search_patterns(
    query: str,
    platform: Optional[PlatformType] = None,
    pattern_type: Optional[str] = None,
    limit: int = 10,
    mcp_mgr: MCPManager = Depends(get_mcp_manager)
):
    """Search learned patterns"""
    
    try:
        # Use MCP to search patterns
        search_params = {
            "query": query,
            "limit": limit
        }
        
        if platform:
            search_params["platform"] = platform.value
        if pattern_type:
            search_params["pattern_type"] = pattern_type
        
        # This would call our pattern search MCP tool
        # For now, return placeholder
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "search_time_ms": 0
        }
        
    except Exception as e:
        logger.error(f"Pattern search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern search failed: {str(e)}")


@app.get("/v1/platforms")
async def list_platforms():
    """List supported platforms"""
    return {
        "platforms": [
            {
                "id": platform.value,
                "name": platform.value.replace("_", " ").title(),
                "description": f"{platform.value.replace('_', ' ').title()} platform support"
            }
            for platform in PlatformType
        ]
    }


@app.get("/v1/analysis-types")
async def list_analysis_types():
    """List supported analysis types"""
    return {
        "analysis_types": [
            {
                "id": analysis_type.value,
                "name": analysis_type.value.replace("_", " ").title(),
                "description": f"{analysis_type.value.replace('_', ' ')} capabilities"
            }
            for analysis_type in AnalysisType
        ]
    }


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Mobiz LLM Intelligence Service",
        version="1.0.0",
        description="Universal LLM Intelligence microservice for enterprise platform analysis",
        routes=app.routes,
    )
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://mobizinc.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )