"""
Independent State Schema for LLM Intelligence Service
====================================================
Completely self-contained state management without external dependencies.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum


class PlatformType(str, Enum):
    """Supported enterprise platforms"""
    POWER_PLATFORM = "power_platform"
    SERVICENOW = "servicenow"
    SALESFORCE = "salesforce"
    DYNAMICS365 = "dynamics365"
    GENERIC = "generic"


class AnalysisType(str, Enum):
    """Types of analysis that can be performed"""
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_LEARNING = "pattern_learning"
    REMEDIATION_GENERATION = "remediation_generation"
    BUSINESS_IMPACT_ANALYSIS = "business_impact_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected"""
    PERFORMANCE = "performance"
    USABILITY = "usability"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    DATA_INTEGRITY = "data_integrity"
    WORKFLOW = "workflow"


class SeverityLevel(str, Enum):
    """Severity levels for anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BusinessImpact(BaseModel):
    """Business impact assessment"""
    revenue_at_risk: Optional[float] = Field(default=None, description="Revenue impact in USD")
    affected_users: Optional[int] = Field(default=None, description="Number of affected users")
    affected_processes: Optional[int] = Field(default=None, description="Number of affected processes")
    downtime_hours: Optional[float] = Field(default=None, description="Estimated downtime hours")
    priority_score: float = Field(ge=0.0, le=1.0, description="Priority score 0-1")
    urgency_level: SeverityLevel = SeverityLevel.MEDIUM


class DetectedAnomaly(BaseModel):
    """Structure for detected anomalies"""
    id: str
    type: str  # More flexible than enum for extensibility
    severity: str
    title: str
    description: str
    location: str
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    impact: Optional[BusinessImpact] = None
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnalysisInsight(BaseModel):
    """Analysis insights"""
    category: str
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    actionable: bool = True


class AnalysisState(BaseModel):
    """Independent analysis state for LLM Intelligence Service"""
    
    # Core identification
    analysis_id: str
    conversation_id: str
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Platform context
    platform: PlatformType
    data_type: str
    analysis_type: AnalysisType = AnalysisType.ANOMALY_DETECTION
    
    # Input data
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis results
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    anomalies: List[DetectedAnomaly] = Field(default_factory=list)
    insights: List[AnalysisInsight] = Field(default_factory=list)
    
    # Quality metrics
    overall_confidence: float = 0.0
    processing_time_ms: Optional[int] = None
    rules_applied: int = 0
    
    # Business context
    business_impact: Optional[BusinessImpact] = None
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_anomaly(self, anomaly: DetectedAnomaly) -> None:
        """Add a detected anomaly"""
        self.anomalies.append(anomaly)
        self._update_confidence()
        self.updated_at = datetime.now(timezone.utc)
    
    def add_insight(self, insight: AnalysisInsight) -> None:
        """Add an analysis insight"""
        self.insights.append(insight)
        self.updated_at = datetime.now(timezone.utc)
    
    def mark_completed(self, processing_time_ms: int = None) -> None:
        """Mark analysis as completed"""
        self.status = "completed"
        if processing_time_ms:
            self.processing_time_ms = processing_time_ms
        self.updated_at = datetime.now(timezone.utc)
        self._update_confidence()
    
    def mark_failed(self, error: str) -> None:
        """Mark analysis as failed"""
        self.status = "failed"
        self.errors.append(error)
        self.updated_at = datetime.now(timezone.utc)
    
    def _update_confidence(self) -> None:
        """Update overall confidence based on anomalies and insights"""
        if not self.anomalies and not self.insights:
            self.overall_confidence = 0.0
            return
        
        total_confidence = 0.0
        total_items = 0
        
        for anomaly in self.anomalies:
            total_confidence += anomaly.confidence
            total_items += 1
        
        for insight in self.insights:
            total_confidence += insight.confidence
            total_items += 1
        
        if total_items > 0:
            self.overall_confidence = total_confidence / total_items


class StateManager:
    """Manager for analysis state operations"""
    
    @staticmethod
    def create_analysis_state(
        analysis_id: str,
        conversation_id: str,
        user_id: str,
        platform: PlatformType,
        data_type: str,
        raw_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> AnalysisState:
        """Create a new analysis state"""
        return AnalysisState(
            analysis_id=analysis_id,
            conversation_id=conversation_id,
            user_id=user_id,
            platform=platform,
            data_type=data_type,
            raw_data=raw_data,
            context=context or {}
        )
    
    @staticmethod
    def serialize_state(state: AnalysisState) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return state.dict()
    
    @staticmethod
    def deserialize_state(data: Dict[str, Any]) -> AnalysisState:
        """Deserialize state from dictionary"""
        return AnalysisState(**data)