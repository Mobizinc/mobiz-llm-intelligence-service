"""
Independent LangGraph Implementation for LLM Intelligence Service
================================================================
Self-contained graph orchestration without external dependencies.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledGraph

from ..models.independent_state import AnalysisState, PlatformType, AnalysisType, StateManager
from ..services.llm_manager import LLMManager, LLMRequest

logger = logging.getLogger(__name__)


class AnalysisNode:
    """Base class for analysis nodes"""
    
    def __init__(self, name: str, llm_manager: LLMManager):
        self.name = name
        self.llm_manager = llm_manager
    
    async def process(self, state: AnalysisState) -> AnalysisState:
        """Process the analysis state"""
        raise NotImplementedError("Subclasses must implement process method")


class IntentAnalysisNode(AnalysisNode):
    """Analyzes the intent and determines analysis type"""
    
    async def process(self, state: AnalysisState) -> AnalysisState:
        """Analyze intent from raw data"""
        logger.info(f"Processing intent analysis for {state.analysis_id}")
        
        # Simple intent analysis based on data type and platform
        if state.platform == PlatformType.POWER_PLATFORM:
            if state.data_type == "canvas_app":
                state.analysis_type = AnalysisType.ANOMALY_DETECTION
            elif state.data_type == "power_automate_flow":
                state.analysis_type = AnalysisType.PERFORMANCE_OPTIMIZATION
        
        state.status = "processing"
        state.updated_at = datetime.now(timezone.utc)
        
        return state


class AnomalyDetectionNode(AnalysisNode):
    """Detects anomalies in the analyzed data"""
    
    async def process(self, state: AnalysisState) -> AnalysisState:
        """Detect anomalies using LLM analysis"""
        logger.info(f"Processing anomaly detection for {state.analysis_id}")
        
        # Build LLM request for anomaly detection
        request = LLMRequest(
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt(state.platform, state.data_type)
                },
                {
                    "role": "user", 
                    "content": self._build_analysis_prompt(state)
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        try:
            # Get LLM response
            response = await self.llm_manager.complete(request)
            
            # Parse response and extract anomalies (simplified for independence)
            anomalies = self._parse_anomalies_from_response(response.content, state)
            
            # Add anomalies to state
            for anomaly in anomalies:
                state.add_anomaly(anomaly)
            
            state.rules_applied = len(anomalies)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            state.mark_failed(f"Anomaly detection error: {str(e)}")
        
        return state
    
    def _get_system_prompt(self, platform: PlatformType, data_type: str) -> str:
        """Get platform-specific system prompt"""
        if platform == PlatformType.POWER_PLATFORM and data_type == "canvas_app":
            return """You are an expert Power Platform analyst. Analyze Canvas App data for common issues:
- Performance problems (complex formulas, inefficient data loading)
- Usability issues (missing error handling, poor UX patterns)
- Security concerns (improper data access, missing validation)
- Configuration problems (incorrect settings, missing required fields)

Return your analysis in a structured format with specific recommendations."""
        
        return "You are an expert system analyst. Identify potential issues and provide recommendations."
    
    def _build_analysis_prompt(self, state: AnalysisState) -> str:
        """Build analysis prompt from state data"""
        prompt = f"Analyze this {state.platform.value} {state.data_type} for potential issues:\n\n"
        
        # Add raw data
        if "name" in state.raw_data:
            prompt += f"Name: {state.raw_data['name']}\n"
        
        if "controls" in state.raw_data:
            prompt += f"Controls: {len(state.raw_data['controls'])} controls found\n"
            for control in state.raw_data.get("controls", [])[:5]:  # Limit for prompt size
                prompt += f"- {control.get('type', 'Unknown')}: {control.get('name', 'Unnamed')}\n"
        
        if "formulas" in state.raw_data:
            prompt += f"\nFormulas: {len(state.raw_data['formulas'])} formulas found\n"
            for formula in state.raw_data.get("formulas", [])[:3]:  # Limit for prompt size
                prompt += f"- {formula}\n"
        
        # Add context
        if state.context:
            prompt += f"\nContext: {state.context}\n"
        
        prompt += "\nProvide specific anomalies found with recommendations for improvement."
        
        return prompt
    
    def _parse_anomalies_from_response(self, response: str, state: AnalysisState) -> List[Any]:
        """Parse anomalies from LLM response (simplified implementation)"""
        from ..models.independent_state import DetectedAnomaly, BusinessImpact
        
        anomalies = []
        
        # Simple parsing - in production this would be more sophisticated
        # For now, create mock anomalies based on the response content
        if "performance" in response.lower():
            anomalies.append(DetectedAnomaly(
                id=f"{state.analysis_id}_perf_1",
                type="performance",
                severity="medium",
                title="Potential Performance Issue",
                description="Performance concerns identified in the analysis",
                location="Multiple locations",
                recommendation="Review and optimize identified areas",
                confidence=0.75
            ))
        
        if "error" in response.lower() or "handling" in response.lower():
            anomalies.append(DetectedAnomaly(
                id=f"{state.analysis_id}_error_1", 
                type="usability",
                severity="low",
                title="Error Handling Improvement",
                description="Error handling could be improved",
                location="User interaction points",
                recommendation="Add proper error handling and user feedback",
                confidence=0.65
            ))
        
        # If no specific anomalies detected, create a general one
        if not anomalies and len(response) > 50:
            anomalies.append(DetectedAnomaly(
                id=f"{state.analysis_id}_general_1",
                type="configuration",
                severity="low", 
                title="General Recommendation",
                description="Areas for improvement identified",
                location="System configuration",
                recommendation=response[:200] + "..." if len(response) > 200 else response,
                confidence=0.60
            ))
        
        return anomalies


class InsightGenerationNode(AnalysisNode):
    """Generates actionable insights from analysis"""
    
    async def process(self, state: AnalysisState) -> AnalysisState:
        """Generate insights from detected anomalies"""
        logger.info(f"Processing insight generation for {state.analysis_id}")
        
        from ..models.independent_state import AnalysisInsight
        
        # Generate insights based on anomalies found
        insights = []
        
        if state.anomalies:
            # Performance insights
            perf_anomalies = [a for a in state.anomalies if a.type == "performance"]
            if perf_anomalies:
                insights.append(AnalysisInsight(
                    category="performance",
                    title="Performance Optimization Opportunities",
                    description=f"Found {len(perf_anomalies)} performance-related issues that can be optimized",
                    confidence=0.80
                ))
            
            # Usability insights
            usability_anomalies = [a for a in state.anomalies if a.type == "usability"]
            if usability_anomalies:
                insights.append(AnalysisInsight(
                    category="usability",
                    title="User Experience Improvements",
                    description=f"Identified {len(usability_anomalies)} usability improvements",
                    confidence=0.75
                ))
            
            # General insight
            insights.append(AnalysisInsight(
                category="general",
                title="Overall Analysis Summary",
                description=f"Completed analysis of {state.data_type} with {len(state.anomalies)} findings",
                confidence=state.overall_confidence
            ))
        else:
            # No anomalies found
            insights.append(AnalysisInsight(
                category="general",
                title="Clean Analysis Result",
                description="No significant issues detected in the current analysis",
                confidence=0.90
            ))
        
        # Add insights to state
        for insight in insights:
            state.add_insight(insight)
        
        return state


class CompletionNode(AnalysisNode):
    """Completes the analysis and finalizes results"""
    
    async def process(self, state: AnalysisState) -> AnalysisState:
        """Complete the analysis"""
        logger.info(f"Completing analysis for {state.analysis_id}")
        
        # Calculate processing time
        processing_time = int((datetime.now(timezone.utc) - state.created_at).total_seconds() * 1000)
        
        # Mark as completed
        state.mark_completed(processing_time)
        
        logger.info(f"Analysis {state.analysis_id} completed with {len(state.anomalies)} anomalies and {len(state.insights)} insights")
        
        return state


class IndependentAnalysisGraph:
    """Independent analysis graph orchestrator"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.graph = self._build_graph()
    
    def _build_graph(self) -> CompiledGraph:
        """Build the analysis graph"""
        
        # Create nodes
        intent_node = IntentAnalysisNode("intent_analysis", self.llm_manager)
        anomaly_node = AnomalyDetectionNode("anomaly_detection", self.llm_manager) 
        insight_node = InsightGenerationNode("insight_generation", self.llm_manager)
        completion_node = CompletionNode("completion", self.llm_manager)
        
        # Build workflow
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("intent_analysis", intent_node.process)
        workflow.add_node("anomaly_detection", anomaly_node.process)
        workflow.add_node("insight_generation", insight_node.process)
        workflow.add_node("completion", completion_node.process)
        
        # Add edges
        workflow.add_edge("intent_analysis", "anomaly_detection")
        workflow.add_edge("anomaly_detection", "insight_generation")
        workflow.add_edge("insight_generation", "completion")
        workflow.add_edge("completion", END)
        
        # Set entry point
        workflow.set_entry_point("intent_analysis")
        
        return workflow.compile()
    
    async def analyze(
        self,
        analysis_id: str,
        conversation_id: str,
        user_id: str,
        platform: PlatformType,
        data_type: str,
        raw_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> AnalysisState:
        """Run complete analysis"""
        
        # Create initial state
        initial_state = StateManager.create_analysis_state(
            analysis_id=analysis_id,
            conversation_id=conversation_id,
            user_id=user_id,
            platform=platform,
            data_type=data_type,
            raw_data=raw_data,
            context=context
        )
        
        logger.info(f"Starting analysis {analysis_id} for platform {platform.value}")
        
        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            logger.info(f"Analysis {analysis_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            initial_state.mark_failed(str(e))
            return initial_state