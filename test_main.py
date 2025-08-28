"""
Simple test version of LLM Intelligence Service for integration testing
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mobiz LLM Intelligence Service (Test)",
    description="Test version for integration testing",
    version="1.0.0-test"
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
    platform: str = Field(..., description="Platform type (power_platform, etc)")
    data_type: str = Field(..., description="Type of data being analyzed")
    data: Dict[str, Any] = Field(..., description="Data to analyze")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")

class DetectedAnomaly(BaseModel):
    type: str
    severity: str
    title: str
    description: str
    location: str
    recommendation: str
    confidence: float = 0.85

class AnalysisResponse(BaseModel):
    platform: str
    data_type: str
    status: str = "completed"
    anomalies: list[DetectedAnomaly] = []
    insights: list[str] = []
    metadata: Dict[str, Any] = {}

# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mobiz-llm-intelligence",
        "version": "1.0.0-test",
        "timestamp": "2025-08-27T20:45:00Z"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mobiz LLM Intelligence Service (Test Version)",
        "status": "running",
        "docs": "/docs"
    }

# Analysis endpoint
@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Analyze data for anomalies and patterns"""
    logger.info(f"Analyzing {request.platform} data of type {request.data_type}")
    
    # Mock analysis for testing
    mock_anomalies = []
    
    if request.platform == "power_platform":
        if request.data_type == "canvas_app":
            mock_anomalies = [
                DetectedAnomaly(
                    type="performance",
                    severity="medium", 
                    title="Complex Formula in OnVisible",
                    description="OnVisible property contains complex calculations that may impact app load time",
                    location="Screen1.OnVisible",
                    recommendation="Move complex calculations to OnStart or use context variables",
                    confidence=0.87
                ),
                DetectedAnomaly(
                    type="usability",
                    severity="low",
                    title="Missing Error Handling",
                    description="Submit button lacks error handling for API failures",
                    location="SubmitButton.OnSelect",
                    recommendation="Add error handling with Notify() for user feedback",
                    confidence=0.75
                )
            ]
    
    # Mock insights
    insights = [
        f"Analyzed {len(request.data.get('controls', []))} controls in Canvas App",
        "Performance optimization opportunities identified",
        "User experience improvements suggested"
    ]
    
    return AnalysisResponse(
        platform=request.platform,
        data_type=request.data_type,
        status="completed",
        anomalies=mock_anomalies,
        insights=insights,
        metadata={
            "analysis_time_ms": 250,
            "rules_applied": 15,
            "confidence_avg": 0.81
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)