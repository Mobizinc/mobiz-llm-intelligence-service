"""
Universal Intelligence State Schema
==================================
Extends KITT's conversation state for universal platform intelligence analysis.
Supports anomaly detection, pattern learning, and cross-platform correlation.
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal, Union
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum

# Import base types from KITT
from .conversation_state import (
    ConversationStage, 
    IntentType, 
    ResponseType, 
    TechnicalConversationState
)


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
    CROSS_PLATFORM_CORRELATION = "cross_platform_correlation"
    BUSINESS_IMPACT_ANALYSIS = "business_impact_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected"""
    STUCK_PROCESS = "stuck_process"
    MISSING_APPROVER = "missing_approver"
    PERMISSION_ISSUE = "permission_issue"
    CONFIGURATION_ERROR = "configuration_error"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"
    DATA_INTEGRITY = "data_integrity"
    WORKFLOW_LOOP = "workflow_loop"
    SECURITY_VIOLATION = "security_violation"


class RemediationType(str, Enum):
    """Types of remediation that can be generated"""
    SCRIPT_GENERATION = "script_generation"
    CONFIGURATION_FIX = "configuration_fix"
    PERMISSION_RESTORATION = "permission_restoration"
    DATA_CORRECTION = "data_correction"
    PROCESS_RESTART = "process_restart"
    BULK_OPERATION = "bulk_operation"


class BusinessImpact(BaseModel):
    """Business impact assessment"""
    revenue_at_risk: Optional[float] = Field(default=None, description="Revenue impact in USD")
    affected_users: Optional[int] = Field(default=None, description="Number of affected users")
    affected_processes: Optional[int] = Field(default=None, description="Number of affected processes")
    downtime_hours: Optional[float] = Field(default=None, description="Estimated downtime hours")
    priority_score: float = Field(ge=0.0, le=1.0, description="Priority score 0-1")
    urgency_level: Literal["low", "medium", "high", "critical"] = "medium"


class DetectedAnomaly(BaseModel):
    """Structure for detected anomalies"""
    id: str
    type: AnomalyType
    title: str
    description: str
    affected_entities: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    root_cause: Optional[str] = None
    impact: BusinessImpact
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LearnedPattern(BaseModel):
    """Structure for abstracted patterns"""
    pattern_id: str
    pattern_type: str
    description: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    applicable_platforms: List[PlatformType]
    metadata: Dict[str, Any]
    learned_from: List[str]  # Source analysis IDs
    usage_count: int = 0


class RemediationOption(BaseModel):
    """Structure for remediation options"""
    id: str
    type: RemediationType
    title: str
    description: str
    script_content: Optional[str] = None
    estimated_time_minutes: Optional[int] = None
    risk_level: Literal["low", "medium", "high"] = "medium"
    requires_approval: bool = False
    rollback_available: bool = False
    rollback_script: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)


class CrossPlatformCorrelation(BaseModel):
    """Cross-platform entity correlations"""
    correlation_id: str
    source_platform: PlatformType
    target_platform: PlatformType
    source_entity_id: str
    target_entity_id: str
    correlation_type: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any]


class UniversalIntelligenceState(TechnicalConversationState):
    """
    Extended state schema for Universal Intelligence analysis.
    
    Extends KITT's TechnicalConversationState with platform-agnostic
    intelligence capabilities for anomaly detection, pattern learning,
    and automated remediation.
    """
    
    # === PLATFORM CONTEXT ===
    platform: PlatformType                    # Source platform being analyzed
    platform_context: Dict[str, Any]          # Platform-specific context data
    entity_type: NotRequired[str]             # Type of entity (workorder, case, etc.)
    entity_id: NotRequired[str]               # Specific entity identifier
    
    # === ANALYSIS CONTEXT ===
    analysis_type: AnalysisType               # Type of analysis being performed
    analysis_id: str                          # Unique analysis identifier
    parent_analysis_id: NotRequired[str]      # Parent analysis if this is a sub-analysis
    
    # === ANOMALY DETECTION ===
    detected_anomalies: List[DetectedAnomaly] # Anomalies found in current analysis
    anomaly_patterns: NotRequired[List[str]]  # Pattern IDs that matched
    anomaly_confidence: float                 # Overall confidence in anomaly detection
    
    # === PATTERN LEARNING ===
    learned_patterns: NotRequired[List[LearnedPattern]]  # Patterns learned from this analysis
    applied_patterns: NotRequired[List[str]]             # Pattern IDs applied during analysis
    pattern_feedback: NotRequired[Dict[str, Any]]        # Feedback on pattern effectiveness
    
    # === REMEDIATION ===
    remediation_options: List[RemediationOption]         # Generated remediation options
    selected_remediation: NotRequired[RemediationOption] # User-selected remediation
    remediation_status: NotRequired[str]                 # Status of remediation execution
    remediation_results: NotRequired[Dict[str, Any]]     # Results of executed remediation
    
    # === CROSS-PLATFORM ANALYSIS ===
    cross_platform_correlations: NotRequired[List[CrossPlatformCorrelation]]
    related_analyses: NotRequired[List[str]]              # Related analysis IDs
    cascade_impact: NotRequired[Dict[str, Any]]           # Cascade effects to other systems
    
    # === BUSINESS INTELLIGENCE ===
    business_impact: BusinessImpact           # Business impact assessment
    risk_assessment: NotRequired[Dict[str, Any]]          # Risk analysis
    compliance_impact: NotRequired[Dict[str, Any]]        # Compliance implications
    
    # === PERFORMANCE METRICS ===
    analysis_start_time: str                  # ISO timestamp when analysis started
    analysis_end_time: NotRequired[str]       # ISO timestamp when analysis completed
    processing_time_ms: NotRequired[int]      # Total processing time
    token_usage: NotRequired[Dict[str, int]]  # LLM token consumption
    
    # === QUALITY METRICS ===
    quality_score: NotRequired[float]         # Overall quality score
    completeness_score: NotRequired[float]    # How complete the analysis is
    accuracy_validation: NotRequired[Dict[str, Any]]  # Validation of accuracy


class UniversalStateManager:
    """Manager for Universal Intelligence State operations"""
    
    @staticmethod
    def create_initial_state(
        conversation_id: str,
        platform: PlatformType,
        analysis_type: AnalysisType,
        initial_query: str,
        user_id: str,
        channel_id: str,
        platform_context: Dict[str, Any] = None
    ) -> UniversalIntelligenceState:
        """Create initial state for a new analysis"""
        
        now = datetime.now(timezone.utc).isoformat()
        analysis_id = f"{platform.value}_{analysis_type.value}_{conversation_id[:8]}"
        
        return UniversalIntelligenceState(
            # Base conversation fields
            conversation_id=conversation_id,
            initial_query=initial_query,
            current_input=initial_query,
            domain="universal",  # Use universal domain
            thread_id=f"thread_{conversation_id}",
            user_id=user_id,
            channel_id=channel_id,
            correlation_id=conversation_id,
            
            # Intent and classification (initialized)
            intent=IntentType.NEW_QUERY,
            confidence_score=0.0,
            
            # Information management (initialized)
            extracted_info={},
            accumulated_info={},
            required_fields=[],
            missing_fields=[],
            
            # Flow control (initialized)
            stage=ConversationStage.INITIAL,
            questions_asked=[],
            response="",
            response_type=ResponseType.ACKNOWLEDGMENT,
            
            # Timing
            created_at=now,
            updated_at=now,
            version=1,
            
            # Universal Intelligence fields
            platform=platform,
            platform_context=platform_context or {},
            analysis_type=analysis_type,
            analysis_id=analysis_id,
            detected_anomalies=[],
            anomaly_confidence=0.0,
            remediation_options=[],
            business_impact=BusinessImpact(priority_score=0.5),
            analysis_start_time=now
        )
    
    @staticmethod
    def transition_to_stage(
        state: UniversalIntelligenceState, 
        new_stage: ConversationStage,
        update_timestamp: bool = True
    ) -> UniversalIntelligenceState:
        """Transition state to new stage with validation"""
        
        # Create new state with updated stage
        new_state = state.copy()
        new_state['stage'] = new_stage
        
        if update_timestamp:
            new_state['updated_at'] = datetime.now(timezone.utc).isoformat()
            new_state['version'] = new_state.get('version', 1) + 1
        
        return new_state
    
    @staticmethod
    def add_anomaly(
        state: UniversalIntelligenceState,
        anomaly: DetectedAnomaly
    ) -> UniversalIntelligenceState:
        """Add detected anomaly to state"""
        
        new_state = state.copy()
        new_state['detected_anomalies'] = state['detected_anomalies'] + [anomaly]
        
        # Update overall confidence
        if new_state['detected_anomalies']:
            avg_confidence = sum(a.confidence_score for a in new_state['detected_anomalies']) / len(new_state['detected_anomalies'])
            new_state['anomaly_confidence'] = avg_confidence
        
        new_state['updated_at'] = datetime.now(timezone.utc).isoformat()
        new_state['version'] = new_state.get('version', 1) + 1
        
        return new_state
    
    @staticmethod
    def add_remediation(
        state: UniversalIntelligenceState,
        remediation: RemediationOption
    ) -> UniversalIntelligenceState:
        """Add remediation option to state"""
        
        new_state = state.copy()
        new_state['remediation_options'] = state['remediation_options'] + [remediation]
        new_state['updated_at'] = datetime.now(timezone.utc).isoformat()
        new_state['version'] = new_state.get('version', 1) + 1
        
        return new_state
    
    @staticmethod
    def complete_analysis(
        state: UniversalIntelligenceState,
        final_response: str,
        quality_score: float = None
    ) -> UniversalIntelligenceState:
        """Mark analysis as complete with final results"""
        
        now = datetime.now(timezone.utc).isoformat()
        new_state = state.copy()
        
        new_state['stage'] = ConversationStage.SUFFICIENT
        new_state['response'] = final_response
        new_state['response_type'] = ResponseType.IMPLEMENTATION
        new_state['analysis_end_time'] = now
        new_state['updated_at'] = now
        new_state['version'] = new_state.get('version', 1) + 1
        
        if quality_score is not None:
            new_state['quality_score'] = quality_score
        
        # Calculate processing time if start time available
        if 'analysis_start_time' in state:
            start_time = datetime.fromisoformat(state['analysis_start_time'].replace('Z', '+00:00'))
            end_time = datetime.now(timezone.utc)
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            new_state['processing_time_ms'] = processing_time
        
        return new_state