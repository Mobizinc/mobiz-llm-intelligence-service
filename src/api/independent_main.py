"""
Independent LLM Intelligence Service API
=======================================
Self-contained FastAPI service without external dependencies.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.independent_graph import IndependentAnalysisGraph
from ..models.independent_state import (
    PlatformType, AnalysisType, DetectedAnomaly, AnalysisInsight,
    AnalysisState, BusinessImpact
)
from ..services.llm_manager import LLMManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
llm_manager: LLMManager = None
analysis_graph: IndependentAnalysisGraph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global llm_manager, analysis_graph
    
    logger.info("🚀 Starting LLM Intelligence Service (Independent)")
    
    # Initialize LLM Manager
    llm_manager = LLMManager()
    await llm_manager.initialize()
    
    # Initialize Analysis Graph
    analysis_graph = IndependentAnalysisGraph(llm_manager)
    
    logger.info("✅ Service initialization complete")
    
    yield
    
    logger.info("🔄 Shutting down LLM Intelligence Service")


# Create FastAPI app
app = FastAPI(
    title="Mobiz LLM Intelligence Service",
    description="Independent LLM-powered analysis service for enterprise platforms",
    version="2.0.0-independent",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AnalysisRequest(BaseModel):
    """Analysis request model"""
    platform: str = Field(..., description="Platform type (power_platform, servicenow, etc)")
    data_type: str = Field(..., description="Type of data being analyzed")
    data: Dict[str, Any] = Field(..., description="Data to analyze")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    user_id: str = Field(default="anonymous", description="User ID for tracking")


class AnomalyResponse(BaseModel):
    """Anomaly response model"""
    id: str
    type: str
    severity: str
    title: str
    description: str
    location: str
    recommendation: str
    confidence: float


class InsightResponse(BaseModel):
    """Insight response model"""
    category: str
    title: str
    description: str
    confidence: float
    actionable: bool


class AnalysisResponse(BaseModel):
    """Analysis response model"""
    analysis_id: str
    platform: str
    data_type: str
    status: str
    anomalies: List[AnomalyResponse] = []
    insights: List[InsightResponse] = []
    metadata: Dict[str, Any] = {}


# Dependency functions
def get_analysis_graph() -> IndependentAnalysisGraph:
    """Get the analysis graph instance"""
    if analysis_graph is None:
        raise HTTPException(status_code=500, detail="Analysis graph not initialized")
    return analysis_graph


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mobiz-llm-intelligence",
        "version": "2.0.0-independent",
        "message": "Independent service running without external dependencies"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mobiz LLM Intelligence Service (Independent)",
        "status": "running",
        "version": "2.0.0-independent",
        "docs": "/docs"
    }


@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(
    request: AnalysisRequest,
    graph: IndependentAnalysisGraph = Depends(get_analysis_graph)
) -> AnalysisResponse:
    """Analyze data for anomalies and insights"""
    
    # Validate platform
    try:
        platform = PlatformType(request.platform.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform: {request.platform}"
        )
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    conversation_id = request.context.get("conversation_id", analysis_id)
    
    logger.info(f"Starting analysis {analysis_id} for {platform.value} {request.data_type}")
    
    try:
        # Run analysis
        result = await graph.analyze(
            analysis_id=analysis_id,
            conversation_id=conversation_id,
            user_id=request.user_id,
            platform=platform,
            data_type=request.data_type,
            raw_data=request.data,
            context=request.context
        )
        
        # Convert to response format
        return AnalysisResponse(
            analysis_id=result.analysis_id,
            platform=result.platform.value,
            data_type=result.data_type,
            status=result.status,
            anomalies=[
                AnomalyResponse(
                    id=anomaly.id,
                    type=anomaly.type,
                    severity=anomaly.severity,
                    title=anomaly.title,
                    description=anomaly.description,
                    location=anomaly.location,
                    recommendation=anomaly.recommendation,
                    confidence=anomaly.confidence
                )
                for anomaly in result.anomalies
            ],
            insights=[
                InsightResponse(
                    category=insight.category,
                    title=insight.title,
                    description=insight.description,
                    confidence=insight.confidence,
                    actionable=insight.actionable
                )
                for insight in result.insights
            ],
            metadata={
                "overall_confidence": result.overall_confidence,
                "processing_time_ms": result.processing_time_ms,
                "rules_applied": result.rules_applied,
                "created_at": result.created_at.isoformat(),
                "updated_at": result.updated_at.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/v1/platforms")
async def list_platforms():
    """List supported platforms"""
    return {
        "platforms": [
            {
                "id": platform.value,
                "name": platform.value.replace("_", " ").title(),
                "supported_data_types": ["canvas_app", "power_automate_flow", "dataverse_entity"]
                if platform == PlatformType.POWER_PLATFORM
                else ["generic_data"]
            }
            for platform in PlatformType
        ]
    }


@app.get("/v1/status")
async def service_status():
    """Get detailed service status"""
    global llm_manager, analysis_graph
    
    # Check LLM manager health
    llm_health = {}
    if llm_manager:
        try:
            llm_health = await llm_manager.health_check()
        except Exception as e:
            llm_health = {"error": str(e)}
    
    return {
        "service": "mobiz-llm-intelligence",
        "version": "2.0.0-independent",
        "status": "operational" if analysis_graph else "initializing",
        "components": {
            "llm_manager": "healthy" if llm_manager else "not_initialized",
            "analysis_graph": "ready" if analysis_graph else "not_initialized",
            "llm_providers": llm_health
        },
        "capabilities": [
            "anomaly_detection",
            "insight_generation",
            "performance_analysis",
            "business_impact_assessment"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)