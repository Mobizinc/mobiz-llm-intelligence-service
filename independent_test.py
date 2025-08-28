#!/usr/bin/env python3
"""
Independent test runner for LLM Intelligence Service
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Simple mock implementations for testing
class MockLLMManager:
    def __init__(self):
        self.is_available = True
    
    async def initialize(self):
        return True
    
    async def complete(self, request):
        class MockResponse:
            content = "Analysis completed: Performance issues detected in complex formulas. Error handling needs improvement for better user experience."
        return MockResponse()
    
    async def health_check(self):
        return {"openai": True}

class MockAnalysisGraph:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
    
    async def analyze(self, analysis_id, conversation_id, user_id, platform, data_type, raw_data, context=None):
        from models.independent_state import AnalysisState, DetectedAnomaly, AnalysisInsight, PlatformType, AnalysisType
        from datetime import datetime, timezone
        
        # Create state
        state = AnalysisState(
            analysis_id=analysis_id,
            conversation_id=conversation_id, 
            user_id=user_id,
            platform=platform,
            data_type=data_type,
            raw_data=raw_data,
            context=context or {}
        )
        
        # Add mock anomalies
        state.add_anomaly(DetectedAnomaly(
            id=f"{analysis_id}_perf",
            type="performance",
            severity="medium",
            title="Complex Formula Performance",
            description="Complex calculations detected that may impact performance",
            location="Screen formulas",
            recommendation="Consider optimizing formula complexity",
            confidence=0.85
        ))
        
        state.add_anomaly(DetectedAnomaly(
            id=f"{analysis_id}_usability", 
            type="usability",
            severity="low",
            title="Error Handling Enhancement",
            description="Missing error handling for user interactions",
            location="Button controls",
            recommendation="Add error handling and user feedback",
            confidence=0.70
        ))
        
        # Add insights
        state.add_insight(AnalysisInsight(
            category="performance",
            title="Performance Optimization Available",
            description="Multiple performance improvement opportunities identified",
            confidence=0.80
        ))
        
        state.add_insight(AnalysisInsight(
            category="usability", 
            title="UX Enhancement Suggestions",
            description="User experience can be improved with better error handling",
            confidence=0.75
        ))
        
        state.mark_completed(250)
        return state

# FastAPI app with mock components
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List

app = FastAPI(
    title="Mobiz LLM Intelligence Service (Independent)",
    description="Independent LLM-powered analysis service",
    version="2.0.0-independent"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
llm_manager = MockLLMManager()
analysis_graph = MockAnalysisGraph(llm_manager)

# Models
class AnalysisRequest(BaseModel):
    platform: str
    data_type: str
    data: Dict[str, Any]
    context: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = "anonymous"

class AnomalyResponse(BaseModel):
    id: str
    type: str
    severity: str
    title: str
    description: str
    location: str
    recommendation: str
    confidence: float

class InsightResponse(BaseModel):
    category: str
    title: str
    description: str
    confidence: float
    actionable: bool = True

class AnalysisResponse(BaseModel):
    analysis_id: str
    platform: str
    data_type: str
    status: str
    anomalies: List[AnomalyResponse] = []
    insights: List[InsightResponse] = []
    metadata: Dict[str, Any] = {}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "mobiz-llm-intelligence",
        "version": "2.0.0-independent",
        "message": "Independent service running without external dependencies"
    }

@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    import uuid
    from models.independent_state import PlatformType
    
    # Validate platform
    try:
        platform = PlatformType(request.platform.lower())
    except ValueError:
        platform = PlatformType.GENERIC
    
    analysis_id = str(uuid.uuid4())
    
    # Run analysis
    result = await analysis_graph.analyze(
        analysis_id=analysis_id,
        conversation_id=request.context.get("conversation_id", analysis_id),
        user_id=request.user_id,
        platform=platform,
        data_type=request.data_type,
        raw_data=request.data,
        context=request.context
    )
    
    return AnalysisResponse(
        analysis_id=result.analysis_id,
        platform=result.platform.value,
        data_type=result.data_type,
        status=result.status,
        anomalies=[
            AnomalyResponse(
                id=a.id,
                type=a.type,
                severity=a.severity,
                title=a.title,
                description=a.description,
                location=a.location,
                recommendation=a.recommendation,
                confidence=a.confidence
            ) for a in result.anomalies
        ],
        insights=[
            InsightResponse(
                category=i.category,
                title=i.title,
                description=i.description,
                confidence=i.confidence,
                actionable=i.actionable
            ) for i in result.insights
        ],
        metadata={
            "overall_confidence": result.overall_confidence,
            "processing_time_ms": result.processing_time_ms,
            "rules_applied": result.rules_applied
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Independent LLM Intelligence Service...")
    uvicorn.run(app, host="0.0.0.0", port=8001)