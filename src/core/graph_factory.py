"""
Graph Factory for LangGraph Studio
==================================
This module provides factory functions that create graphs dynamically,
avoiding the need to export static graph instances.

The factory pattern enables LangGraph Studio to instantiate graphs on-demand,
allowing for hot reloading and easy development without constant exports.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def create_conversation_graph(domain: str = "power_platform") -> CompiledStateGraph:
    """
    Factory function for main conversation graph.
    LangGraph Studio will call this function to create the graph dynamically.
    
    Args:
        domain: Domain type (power_platform, servicenow, salesforce, generic)
        
    Returns:
        CompiledStateGraph: Ready-to-use conversation graph
    """
    logger.info(f"Creating conversation graph for domain: {domain}")
    
    try:
        from src.core.conversation_graph import ConversationGraph
        from src.models.conversation_state import DomainType
        
        # Map string to enum
        domain_map = {
            "power_platform": DomainType.POWER_PLATFORM,
            "servicenow": DomainType.SERVICENOW, 
            "salesforce": DomainType.SALESFORCE,
            "generic": DomainType.GENERIC
        }
        
        domain_type = domain_map.get(domain.lower(), DomainType.POWER_PLATFORM)
        
        # Create graph instance with your existing logic
        graph = ConversationGraph(
            domain=domain_type,
            enable_parallel_execution=True,
            enable_supervisor_mode=True,
            timeout=300  # 5 minute timeout for development
        )
        
        logger.info(f"Conversation graph created successfully for {domain_type}")
        return graph.app
        
    except Exception as e:
        logger.error(f"Failed to create conversation graph: {e}")
        # Return a simple fallback graph
        return _create_fallback_graph("conversation_graph_error")


def create_independent_graph() -> CompiledStateGraph:
    """
    Factory function for independent analysis graph.
    Creates a new graph instance each time it's called.
    
    Returns:
        CompiledStateGraph: Independent analysis graph
    """
    logger.info("Creating independent analysis graph")
    
    try:
        from src.core.independent_graph import IndependentAnalysisGraph
        from src.services.llm_manager import LLMManager
        
        # Create LLM manager
        llm_manager = LLMManager()
        
        # Initialize it (handle async in sync context)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run async initialization
        if loop.is_running():
            # If loop is already running, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, llm_manager.initialize())
                future.result()
        else:
            loop.run_until_complete(llm_manager.initialize())
        
        # Create and return the graph
        analysis_graph = IndependentAnalysisGraph(llm_manager)
        logger.info("Independent analysis graph created successfully")
        return analysis_graph.graph
        
    except Exception as e:
        logger.error(f"Failed to create independent graph: {e}")
        return _create_fallback_graph("independent_graph_error")


def create_development_graph(test_scenario: str = "basic") -> CompiledStateGraph:
    """
    Development graph for testing new nodes and edges.
    You can modify this without affecting production graphs.
    
    Args:
        test_scenario: Type of test scenario to create
        
    Returns:
        CompiledStateGraph: Development testing graph
    """
    logger.info(f"Creating development graph for scenario: {test_scenario}")
    
    try:
        from langgraph.graph import StateGraph, END
        from src.models.conversation_state import TechnicalConversationState, ConversationStage
        
        # Create a test workflow based on scenario
        workflow = StateGraph(TechnicalConversationState)
        
        if test_scenario == "basic":
            # Simple linear flow for basic testing
            async def start_node(state: TechnicalConversationState) -> TechnicalConversationState:
                state['stage'] = ConversationStage.PROCESSING
                state['response'] = "Development graph - Start node executed"
                state['current_node'] = "start"
                return state
            
            async def middle_node(state: TechnicalConversationState) -> TechnicalConversationState:
                state['stage'] = ConversationStage.PROCESSING
                state['response'] = "Development graph - Middle node executed"
                state['current_node'] = "middle"
                return state
            
            async def end_node(state: TechnicalConversationState) -> TechnicalConversationState:
                state['stage'] = ConversationStage.COMPLETED
                state['response'] = "Development graph - Completed successfully"
                state['current_node'] = "end"
                return state
            
            workflow.add_node("start", start_node)
            workflow.add_node("middle", middle_node)
            workflow.add_node("end", end_node)
            
            workflow.set_entry_point("start")
            workflow.add_edge("start", "middle")
            workflow.add_edge("middle", "end")
            workflow.add_edge("end", END)
            
        elif test_scenario == "conditional":
            # Conditional routing test
            async def decision_node(state: TechnicalConversationState) -> TechnicalConversationState:
                state['stage'] = ConversationStage.PROCESSING
                state['confidence_score'] = 0.8
                return state
            
            async def high_confidence_node(state: TechnicalConversationState) -> TechnicalConversationState:
                state['response'] = "High confidence path taken"
                return state
            
            async def low_confidence_node(state: TechnicalConversationState) -> TechnicalConversationState:
                state['response'] = "Low confidence path taken"
                return state
            
            def route_by_confidence(state: TechnicalConversationState) -> str:
                confidence = state.get('confidence_score', 0.0)
                return "high_confidence" if confidence > 0.7 else "low_confidence"
            
            workflow.add_node("decision", decision_node)
            workflow.add_node("high_confidence", high_confidence_node)
            workflow.add_node("low_confidence", low_confidence_node)
            
            workflow.set_entry_point("decision")
            workflow.add_conditional_edges(
                "decision",
                route_by_confidence,
                {
                    "high_confidence": "high_confidence",
                    "low_confidence": "low_confidence"
                }
            )
            workflow.add_edge("high_confidence", END)
            workflow.add_edge("low_confidence", END)
            
        else:
            # Default simple test
            async def test_node(state: TechnicalConversationState) -> TechnicalConversationState:
                state['response'] = f"Test node executed for scenario: {test_scenario}"
                state['stage'] = ConversationStage.COMPLETED
                return state
            
            workflow.add_node("test", test_node)
            workflow.set_entry_point("test")
            workflow.add_edge("test", END)
        
        compiled_graph = workflow.compile()
        logger.info(f"Development graph created successfully for scenario: {test_scenario}")
        return compiled_graph
        
    except Exception as e:
        logger.error(f"Failed to create development graph: {e}")
        return _create_fallback_graph("development_graph_error")


def create_power_platform_graph() -> CompiledStateGraph:
    """
    Specialized graph for Power Platform analysis.
    This demonstrates domain-specific graph creation with actual domain logic.
    
    Returns:
        CompiledStateGraph: Power Platform specific graph
    """
    logger.info("Creating Power Platform specialized graph")
    
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        class PowerPlatformState(TypedDict):
            message: str
            platform_type: str
            app_analysis: dict
            flow_analysis: dict
            security_check: dict
            recommendations: list
            stage: str
            count: int
        
        workflow = StateGraph(PowerPlatformState)
        
        async def analyze_power_apps(state: PowerPlatformState) -> PowerPlatformState:
            """Analyze Power Apps components"""
            message = state.get('message', 'Analyzing Power Platform')
            count = state.get('count', 0) + 1
            
            # Simulate Power Apps analysis
            app_analysis = {
                "total_apps": 3,
                "canvas_apps": 2,
                "model_driven_apps": 1,
                "compliance_score": 85,
                "performance_issues": ["Slow loading screens", "Complex formulas"]
            }
            
            return {
                **state,
                "message": f"Power Apps Analysis: {message}",
                "platform_type": "Power Platform",
                "app_analysis": app_analysis,
                "stage": "apps_analyzed",
                "count": count
            }
        
        async def analyze_power_automate(state: PowerPlatformState) -> PowerPlatformState:
            """Analyze Power Automate flows"""
            flow_analysis = {
                "total_flows": 5,
                "active_flows": 4,
                "failed_runs": 2,
                "efficiency_score": 78,
                "bottlenecks": ["SharePoint connector delays", "Approval timeouts"]
            }
            
            return {
                **state,
                "flow_analysis": flow_analysis,
                "stage": "flows_analyzed"
            }
        
        async def security_audit(state: PowerPlatformState) -> PowerPlatformState:
            """Perform security audit"""
            security_check = {
                "dlp_policies": 2,
                "governance_score": 92,
                "admin_roles": 3,
                "vulnerabilities": ["Overprivileged connectors"],
                "compliance_status": "Compliant"
            }
            
            return {
                **state,
                "security_check": security_check,
                "stage": "security_audited"
            }
        
        async def generate_recommendations(state: PowerPlatformState) -> PowerPlatformState:
            """Generate recommendations based on analysis"""
            app_issues = len(state.get('app_analysis', {}).get('performance_issues', []))
            flow_issues = state.get('flow_analysis', {}).get('failed_runs', 0)
            security_vulns = len(state.get('security_check', {}).get('vulnerabilities', []))
            
            recommendations = []
            
            if app_issues > 0:
                recommendations.append("Optimize app performance by simplifying complex formulas")
            
            if flow_issues > 0:
                recommendations.append("Review and fix failed Power Automate flows")
            
            if security_vulns > 0:
                recommendations.append("Address security vulnerabilities in connectors")
            
            recommendations.append("Implement regular governance reviews")
            recommendations.append("Enable advanced monitoring and alerting")
            
            return {
                **state,
                "recommendations": recommendations,
                "stage": "completed"
            }
        
        # Build the workflow
        workflow.add_node("analyze_apps", analyze_power_apps)
        workflow.add_node("analyze_flows", analyze_power_automate) 
        workflow.add_node("security_audit", security_audit)
        workflow.add_node("generate_recommendations", generate_recommendations)
        
        # Set up the flow
        workflow.set_entry_point("analyze_apps")
        workflow.add_edge("analyze_apps", "analyze_flows")
        workflow.add_edge("analyze_flows", "security_audit")
        workflow.add_edge("security_audit", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        
        compiled_graph = workflow.compile()
        logger.info("Power Platform graph created successfully")
        return compiled_graph
        
    except Exception as e:
        logger.error(f"Failed to create Power Platform graph: {e}")
        return _create_fallback_graph("power_platform_graph_error")


def create_minimal_test_graph() -> CompiledStateGraph:
    """
    Minimal graph for quick testing and debugging.
    Perfect for rapid iteration during development.
    
    Returns:
        CompiledStateGraph: Minimal test graph
    """
    logger.info("Creating minimal test graph")
    
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        # Use a simple TypedDict instead of importing complex state
        class SimpleState(TypedDict):
            message: str
            count: int
        
        workflow = StateGraph(SimpleState)
        
        async def minimal_node(state: SimpleState) -> SimpleState:
            """Minimal node for quick testing"""
            return {
                "message": f"Processed: {state.get('message', 'Hello')}",
                "count": state.get("count", 0) + 1
            }
        
        workflow.add_node("minimal", minimal_node)
        workflow.set_entry_point("minimal")
        workflow.add_edge("minimal", END)
        
        compiled_graph = workflow.compile()
        logger.info("Minimal test graph created successfully")
        return compiled_graph
        
    except Exception as e:
        logger.error(f"Failed to create minimal test graph: {e}")
        return _create_fallback_graph("minimal_test_error")


def _create_fallback_graph(error_context: str) -> CompiledStateGraph:
    """
    Create a simple fallback graph when main graph creation fails.
    This ensures LangGraph Studio always has something to work with.
    
    Args:
        error_context: Context of what failed
        
    Returns:
        CompiledStateGraph: Simple fallback graph
    """
    logger.warning(f"Creating fallback graph for error context: {error_context}")
    
    from langgraph.graph import StateGraph, END
    
    # Use a simple dict state for fallback
    workflow = StateGraph(dict)
    
    def fallback_node(state: dict) -> dict:
        state['error'] = f"Fallback graph activated due to: {error_context}"
        state['status'] = 'fallback'
        return state
    
    workflow.add_node("fallback", fallback_node)
    workflow.set_entry_point("fallback")
    workflow.add_edge("fallback", END)
    
    return workflow.compile()


# Export functions that can be parameterized
def create_graph_with_domain(domain: str = "power_platform") -> CompiledStateGraph:
    """
    Wrapper function that allows domain parameterization from langgraph.json
    
    Args:
        domain: Domain to create graph for
        
    Returns:
        CompiledStateGraph: Domain-specific conversation graph
    """
    return create_conversation_graph(domain)


# Development utilities
def list_available_graphs() -> Dict[str, str]:
    """
    List all available graph factory functions.
    Useful for development and debugging.
    
    Returns:
        Dict mapping graph names to descriptions
    """
    return {
        "conversation": "Main conversation graph with full LangGraph orchestration",
        "independent": "Lightweight independent analysis graph", 
        "development": "Development graph for testing new nodes and edges",
        "power_platform": "Specialized Power Platform analysis graph",
        "minimal_test": "Minimal graph for quick testing and debugging"
    }


def get_graph_info(graph_name: str) -> Dict[str, Any]:
    """
    Get information about a specific graph.
    
    Args:
        graph_name: Name of the graph
        
    Returns:
        Dict with graph information
    """
    graphs_info = {
        "conversation": {
            "description": "Main conversation graph",
            "nodes": ["intent_classifier", "information_extractor", "requirement_analyzer", 
                     "sufficiency_checker", "implementation_generator", "direct_answer"],
            "supports_domains": ["power_platform", "servicenow", "salesforce", "generic"],
            "features": ["parallel_execution", "supervisor_mode", "conditional_routing"]
        },
        "independent": {
            "description": "Independent analysis graph",
            "nodes": ["intent_analysis", "anomaly_detection", "insight_generation", "completion"],
            "supports_domains": ["all"],
            "features": ["llm_analysis", "anomaly_detection", "insight_generation"]
        },
        "development": {
            "description": "Development testing graph",
            "nodes": ["configurable"],
            "supports_domains": ["test"],
            "features": ["conditional_routing", "testing", "experimentation"]
        }
    }
    
    return graphs_info.get(graph_name, {"error": f"Graph {graph_name} not found"})


if __name__ == "__main__":
    # Test the factory functions when run directly
    print("Testing graph factory functions...")
    
    print("\nAvailable graphs:")
    for name, desc in list_available_graphs().items():
        print(f"  {name}: {desc}")
    
    print("\nTesting minimal graph creation...")
    try:
        graph = create_minimal_test_graph()
        print(f"✓ Minimal test graph created successfully: {type(graph)}")
    except Exception as e:
        print(f"✗ Failed to create minimal test graph: {e}")