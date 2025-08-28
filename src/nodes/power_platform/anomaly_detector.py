"""
Power Platform Anomaly Detector Node
====================================
Specialized LangGraph node for detecting anomalies in Power Platform assets
including Canvas Apps, Power Automate Flows, and Dataverse entities.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from ..base.base import BaseNode
from ...models.universal_state import (
    UniversalIntelligenceState,
    DetectedAnomaly,
    AnomalyType,
    BusinessImpact,
    UniversalStateManager
)
from ...services.llm_manager import LLMRequest, ModelType

logger = logging.getLogger(__name__)


class PowerPlatformAnomalyAnalysis(BaseModel):
    """Structured output for Power Platform anomaly analysis"""
    
    anomalies_detected: List[Dict[str, Any]] = Field(description="List of detected anomalies")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence in detection")
    analysis_summary: str = Field(description="Summary of the analysis performed")
    risk_assessment: Dict[str, Any] = Field(description="Risk assessment of identified issues")
    recommended_actions: List[str] = Field(description="Recommended immediate actions")


class PowerPlatformAnomalyDetector(BaseNode):
    """
    Specialized node for detecting anomalies in Power Platform assets.
    
    Capabilities:
    - Canvas App anomaly detection (performance, UI/UX, data flow)
    - Power Automate Flow issues (stuck processes, approval bottlenecks)
    - Dataverse anomalies (schema issues, permission problems)
    - Cross-component dependency analysis
    """
    
    def __init__(self, **kwargs):
        super().__init__(node_name="PowerPlatformAnomalyDetector", **kwargs)
        
        # Power Platform specific patterns
        self.known_anomaly_patterns = {
            "stuck_approval": {
                "indicators": ["missing_approver", "null_approver", "deleted_user"],
                "severity": "high",
                "business_impact": "revenue_blocking"
            },
            "performance_bottleneck": {
                "indicators": ["slow_response", "timeout", "large_dataset"],
                "severity": "medium",
                "business_impact": "user_experience"
            },
            "permission_issue": {
                "indicators": ["access_denied", "unauthorized", "missing_role"],
                "severity": "high",
                "business_impact": "process_blocking"
            },
            "data_integrity": {
                "indicators": ["null_values", "schema_mismatch", "orphaned_records"],
                "severity": "high",
                "business_impact": "data_quality"
            },
            "configuration_error": {
                "indicators": ["invalid_config", "missing_connection", "wrong_environment"],
                "severity": "medium",
                "business_impact": "functionality"
            }
        }
    
    async def process(self, state: UniversalIntelligenceState) -> UniversalIntelligenceState:
        """
        Process Power Platform assets for anomaly detection.
        
        Args:
            state: Current universal intelligence state
            
        Returns:
            Updated state with detected anomalies
        """
        logger.info(f"🔍 Starting Power Platform anomaly detection for {state['analysis_id']}")
        
        try:
            # Extract platform context
            platform_context = state.get('platform_context', {})
            
            # Perform different types of analysis based on entity type
            entity_type = platform_context.get('entity_type', 'unknown')
            
            if entity_type == 'canvas_app':
                anomalies = await self._detect_canvas_app_anomalies(state)
            elif entity_type == 'flow':
                anomalies = await self._detect_flow_anomalies(state)
            elif entity_type == 'dataverse_entity':
                anomalies = await self._detect_dataverse_anomalies(state)
            else:
                # Generic Power Platform analysis
                anomalies = await self._detect_generic_anomalies(state)
            
            # Update state with detected anomalies
            updated_state = state.copy()
            for anomaly in anomalies:
                updated_state = UniversalStateManager.add_anomaly(updated_state, anomaly)
            
            # Update analysis stage
            updated_state = UniversalStateManager.transition_to_stage(
                updated_state, 
                "analyzing"  # ConversationStage.ANALYZING
            )
            
            logger.info(f"✅ Detected {len(anomalies)} anomalies in {state['analysis_id']}")
            return updated_state
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed for {state['analysis_id']}: {e}")
            # Return error state
            error_state = state.copy()
            error_state['stage'] = "error"  # ConversationStage.ERROR
            error_state['response'] = f"Anomaly detection failed: {str(e)}"
            return error_state
    
    async def _detect_canvas_app_anomalies(self, state: UniversalIntelligenceState) -> List[DetectedAnomaly]:
        """Detect anomalies in Canvas Apps"""
        platform_context = state['platform_context']
        
        # Build analysis prompt for Canvas Apps
        system_prompt = """You are an expert Power Platform analyst specializing in Canvas App anomaly detection.
        
Analyze the provided Canvas App data for anomalies including:
- Performance issues (slow loading, large controls, inefficient formulas)
- UI/UX problems (accessibility, navigation, user experience)
- Data flow issues (broken connections, missing data sources)
- Security vulnerabilities (overprivileged access, data exposure)
- Architectural problems (code smells, maintainability issues)

Focus on business impact and provide actionable insights."""

        user_message = f"""
Analyze this Canvas App for anomalies:

App Name: {platform_context.get('app_name', 'Unknown')}
App ID: {platform_context.get('app_id', 'Unknown')}

Context Graph Data:
{json.dumps(platform_context.get('context_graph', {}), indent=2)}

Identify specific anomalies with confidence scores and business impact assessment.
"""

        # Call LLM for analysis
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            structured_output=PowerPlatformAnomalyAnalysis,
            model_preference=ModelType.GPT5,
            temperature=0.1
        )
        
        response = await self.call_llm_structured(llm_request)
        
        # Convert to DetectedAnomaly objects
        anomalies = []
        for anomaly_data in response.structured_data.anomalies_detected:
            anomaly = DetectedAnomaly(
                id=f"canvasapp_{anomaly_data.get('id', len(anomalies))}",
                type=AnomalyType(anomaly_data.get('type', 'configuration_error')),
                title=anomaly_data.get('title', 'Canvas App Issue'),
                description=anomaly_data.get('description', ''),
                affected_entities=[platform_context.get('app_id', 'unknown')],
                confidence_score=anomaly_data.get('confidence_score', 0.5),
                root_cause=anomaly_data.get('root_cause'),
                impact=self._calculate_business_impact(anomaly_data)
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_flow_anomalies(self, state: UniversalIntelligenceState) -> List[DetectedAnomaly]:
        """Detect anomalies in Power Automate Flows"""
        platform_context = state['platform_context']
        
        # Leverage existing flow analysis capabilities
        system_prompt = """You are an expert Power Automate flow analyst specialized in detecting stuck processes, approval bottlenecks, and workflow anomalies.

Analyze flows for:
- Stuck approval processes (missing approvers, deleted users)
- Performance bottlenecks (long-running actions, timeouts)
- Logic errors (infinite loops, unreachable branches)
- Connection failures (expired credentials, broken connectors)
- Security issues (overprivileged flows, data exposure)

Focus on identifying the root cause and business impact of issues."""

        # Extract flow data
        flow_data = platform_context.get('flow_data', {})
        stuck_orders = platform_context.get('stuck_orders', [])
        
        user_message = f"""
Analyze this Power Automate Flow for anomalies:

Flow Details:
{json.dumps(flow_data, indent=2)}

Stuck Work Orders (if applicable):
{json.dumps(stuck_orders[:5], indent=2)}  # Limit for prompt size

Additional Context:
{json.dumps(platform_context.get('additional_context', {}), indent=2)}

Identify specific anomalies with confidence scores, affected entities, and business impact.
"""

        llm_request = LLMRequest(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            structured_output=PowerPlatformAnomalyAnalysis,
            model_preference=ModelType.GPT5,
            temperature=0.1
        )
        
        response = await self.call_llm_structured(llm_request)
        
        # Convert to DetectedAnomaly objects with specific focus on stuck processes
        anomalies = []
        for anomaly_data in response.structured_data.anomalies_detected:
            # Special handling for stuck approval pattern
            anomaly_type = AnomalyType.STUCK_PROCESS
            if "approver" in anomaly_data.get('description', '').lower():
                anomaly_type = AnomalyType.MISSING_APPROVER
            elif "permission" in anomaly_data.get('description', '').lower():
                anomaly_type = AnomalyType.PERMISSION_ISSUE
            
            anomaly = DetectedAnomaly(
                id=f"flow_{anomaly_data.get('id', len(anomalies))}",
                type=anomaly_type,
                title=anomaly_data.get('title', 'Flow Issue'),
                description=anomaly_data.get('description', ''),
                affected_entities=anomaly_data.get('affected_entities', []),
                confidence_score=anomaly_data.get('confidence_score', 0.7),
                root_cause=anomaly_data.get('root_cause'),
                impact=self._calculate_business_impact(anomaly_data, flow_context=True)
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_dataverse_anomalies(self, state: UniversalIntelligenceState) -> List[DetectedAnomaly]:
        """Detect anomalies in Dataverse entities"""
        platform_context = state['platform_context']
        
        system_prompt = """You are a Dataverse expert specializing in data integrity, schema, and permission anomalies.

Analyze Dataverse entities for:
- Data integrity issues (null values, orphaned records, constraint violations)
- Schema problems (missing fields, type mismatches, relationship issues)
- Permission anomalies (unauthorized access, missing security roles)
- Performance issues (inefficient queries, missing indexes)
- Configuration errors (wrong environments, missing connections)

Provide detailed analysis with impact assessment."""

        entity_data = platform_context.get('entity_data', {})
        
        user_message = f"""
Analyze this Dataverse entity for anomalies:

Entity Data:
{json.dumps(entity_data, indent=2)}

Schema Information:
{json.dumps(platform_context.get('schema_info', {}), indent=2)}

Access Patterns:
{json.dumps(platform_context.get('access_patterns', {}), indent=2)}

Identify specific data integrity, schema, and permission anomalies.
"""

        llm_request = LLMRequest(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            structured_output=PowerPlatformAnomalyAnalysis,
            model_preference=ModelType.GPT5,
            temperature=0.1
        )
        
        response = await self.call_llm_structured(llm_request)
        
        # Convert to DetectedAnomaly objects
        anomalies = []
        for anomaly_data in response.structured_data.anomalies_detected:
            anomaly = DetectedAnomaly(
                id=f"dataverse_{anomaly_data.get('id', len(anomalies))}",
                type=AnomalyType(anomaly_data.get('type', 'data_integrity')),
                title=anomaly_data.get('title', 'Dataverse Issue'),
                description=anomaly_data.get('description', ''),
                affected_entities=anomaly_data.get('affected_entities', []),
                confidence_score=anomaly_data.get('confidence_score', 0.6),
                root_cause=anomaly_data.get('root_cause'),
                impact=self._calculate_business_impact(anomaly_data)
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_generic_anomalies(self, state: UniversalIntelligenceState) -> List[DetectedAnomaly]:
        """Generic Power Platform anomaly detection"""
        platform_context = state['platform_context']
        
        system_prompt = """You are a Power Platform expert capable of analyzing any component for anomalies.

Perform comprehensive analysis for:
- Configuration issues
- Performance problems  
- Security vulnerabilities
- Integration failures
- User experience issues

Provide specific, actionable insights with business impact assessment."""

        user_message = f"""
Analyze this Power Platform component for anomalies:

Query: {state['initial_query']}

Platform Context:
{json.dumps(platform_context, indent=2)}

Identify any anomalies with confidence scores and business impact.
"""

        llm_request = LLMRequest(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            structured_output=PowerPlatformAnomalyAnalysis,
            model_preference=ModelType.GPT5,
            temperature=0.1
        )
        
        response = await self.call_llm_structured(llm_request)
        
        # Convert to DetectedAnomaly objects
        anomalies = []
        for anomaly_data in response.structured_data.anomalies_detected:
            anomaly = DetectedAnomaly(
                id=f"generic_{anomaly_data.get('id', len(anomalies))}",
                type=AnomalyType(anomaly_data.get('type', 'configuration_error')),
                title=anomaly_data.get('title', 'Power Platform Issue'),
                description=anomaly_data.get('description', ''),
                affected_entities=anomaly_data.get('affected_entities', ['unknown']),
                confidence_score=anomaly_data.get('confidence_score', 0.5),
                root_cause=anomaly_data.get('root_cause'),
                impact=self._calculate_business_impact(anomaly_data)
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_business_impact(self, anomaly_data: Dict[str, Any], flow_context: bool = False) -> BusinessImpact:
        """Calculate business impact from anomaly data"""
        
        # Extract impact indicators
        severity = anomaly_data.get('severity', 'medium')
        affected_count = len(anomaly_data.get('affected_entities', []))
        
        # Calculate priority score based on severity and scale
        priority_score = 0.3  # Default
        if severity == 'critical':
            priority_score = 0.9
        elif severity == 'high':
            priority_score = 0.7
        elif severity == 'medium':
            priority_score = 0.5
        elif severity == 'low':
            priority_score = 0.3
        
        # Adjust for scale
        if affected_count > 10:
            priority_score = min(1.0, priority_score + 0.2)
        elif affected_count > 5:
            priority_score = min(1.0, priority_score + 0.1)
        
        # Special handling for flow context (revenue impact)
        revenue_at_risk = None
        if flow_context and "stuck" in anomaly_data.get('description', '').lower():
            # Estimate revenue impact for stuck processes
            # This could be enhanced with actual business data
            estimated_order_value = 30000  # Average work order value
            revenue_at_risk = affected_count * estimated_order_value
        
        return BusinessImpact(
            revenue_at_risk=revenue_at_risk,
            affected_users=anomaly_data.get('affected_users'),
            affected_processes=affected_count,
            downtime_hours=anomaly_data.get('estimated_downtime_hours'),
            priority_score=priority_score,
            urgency_level=severity if severity in ['low', 'medium', 'high', 'critical'] else 'medium'
        )
    
    async def call_llm_structured(self, llm_request: LLMRequest):
        """Call LLM with structured output handling"""
        try:
            # This would use the LLMManager from the parent application
            # For now, return a mock response
            from ...services.llm_manager import LLMManager
            
            # In actual implementation, this would be injected
            llm_manager = LLMManager()
            return await llm_manager.complete(llm_request)
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Return fallback structured response
            fallback_analysis = PowerPlatformAnomalyAnalysis(
                anomalies_detected=[{
                    "id": "fallback_001",
                    "type": "configuration_error",
                    "title": "Analysis Error",
                    "description": f"Unable to complete analysis: {str(e)}",
                    "confidence_score": 0.1,
                    "severity": "low",
                    "affected_entities": ["unknown"]
                }],
                confidence_score=0.1,
                analysis_summary="Analysis failed due to technical error",
                risk_assessment={"overall_risk": "unknown"},
                recommended_actions=["Review system logs", "Retry analysis"]
            )
            
            class MockResponse:
                structured_data = fallback_analysis
            
            return MockResponse()