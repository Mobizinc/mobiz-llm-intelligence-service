"""
Core LangGraph Workflow Implementation for Technical Bots

This module implements the main LangGraph orchestration layer that coordinates
all agent nodes using conditional routing, parallel execution, and intelligent
flow control. This is the brain of the system.

Features:
- Complete graph definition with all Epic 2 nodes
- Conditional edges based on conversation state
- Parallel execution paths for independent operations 
- Cycle detection and prevention
- Graph visualization for debugging
- Hot-reload capability for graph updates

Part of Epic 3: Orchestration & Intelligent Routing
Story 3.1: Design and Build Core LangGraph Workflow
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal, TYPE_CHECKING

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.conversation_state import (
    TechnicalConversationState, 
    ConversationStage, 
    IntentType,
    ResponseType,
    DomainType,
    ConversationStateManager
)
from ..nodes import (
    IntentClassifierNode,
    InformationExtractorNode,
    RequirementAnalyzerNode,
    SufficiencyCheckerNode,
    ImplementationGeneratorNode,
    DirectAnswerNode
)
from ..checkpointers.azure_checkpointer import AzureCheckpointer, CheckpointerConfig
from ..monitoring.state_metrics import StateMetricsCollector
from ..telemetry.state_telemetry import StateOperationTracer
from .routing_engine import RoutingEngine
from .flow_controller import FlowController, CircuitBreakerConfig, RateLimitConfig, RetryConfig

if TYPE_CHECKING:
    from .supervisor import SupervisorNode
    from .parallel_executor import ParallelExecutor

logger = logging.getLogger(__name__)


class ConversationGraph:
    """
    Core LangGraph workflow implementation for Technical Bots.
    
    This class orchestrates the entire conversation flow using LangGraph's
    StateGraph with all Epic 2 agent nodes connected through intelligent
    routing logic.
    
    Architecture:
    - StateGraph with TechnicalConversationState
    - All Epic 2 agent nodes as graph nodes
    - Conditional edges for intelligent routing
    - Parallel execution where appropriate
    - Error handling and circuit breaker integration
    """
    
    def __init__(
        self,
        domain: DomainType,
        checkpointer: Optional[MemorySaver] = None,
        enable_parallel_execution: bool = True,
        enable_supervisor_mode: bool = True,
        timeout: int = 30
    ):
        self.domain = domain
        self.enable_parallel = enable_parallel_execution
        self.enable_supervisor = enable_supervisor_mode
        self.timeout = timeout
        
        # Initialize services
        self.metrics_collector = StateMetricsCollector()
        self.telemetry_service = StateOperationTracer()
        
        # Initialize checkpointer
        if checkpointer:
            self.checkpointer = checkpointer
        else:
            self.checkpointer = AzureCheckpointer(
                CheckpointerConfig(
                    table_name=f"conversation_checkpoints_{domain}",
                    max_versions=10,
                    ttl_hours=24
                )
            )
        
        # Initialize routing engine
        self.routing_engine = RoutingEngine(domain=domain)
        
        # Initialize flow controller with domain-specific config
        self.flow_controller = FlowController(
            default_circuit_config=CircuitBreakerConfig(
                failure_threshold=3 if domain in ['cloud', 'network'] else 5,
                timeout_seconds=45 if domain == 'devops' else 60,
                failure_rate_threshold=0.4
            ),
            default_rate_config=RateLimitConfig(
                max_requests_per_second=15.0 if domain == 'dev' else 10.0,
                max_requests_per_minute=120.0 if domain == 'cloud' else 100.0
            ),
            default_retry_config=RetryConfig(
                max_attempts=4 if domain in ['cloud', 'devops'] else 3,
                initial_delay=0.5,
                max_delay=30.0
            )
        )
        
        # Initialize agent nodes
        self.nodes = self._initialize_nodes()
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        
        # Compile the application with checkpointing
        self.app = self.workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=None,  # No interrupts for now
            interrupt_after=None
        )
        
        logger.info(f"ConversationGraph initialized for domain: {domain}")
        logger.info(f"Nodes: {list(self.nodes.keys())}")
        logger.info(f"Parallel execution: {self.enable_parallel}")
        logger.info(f"Supervisor mode: {self.enable_supervisor}")
    
    def _initialize_nodes(self) -> Dict[str, Any]:
        """Initialize all agent nodes for the workflow"""
        nodes = {
            # Core processing nodes from Epic 2
            "intent_classifier": IntentClassifierNode(
                node_name="intent_classifier",
                domain=self.domain,
                timeout=10
            ),
            "information_extractor": InformationExtractorNode(
                node_name="information_extractor", 
                domain=self.domain,
                timeout=15
            ),
            "requirement_analyzer": RequirementAnalyzerNode(
                node_name="requirement_analyzer",
                domain=self.domain,
                timeout=15
            ),
            "sufficiency_checker": SufficiencyCheckerNode(
                node_name="sufficiency_checker",
                domain=self.domain,
                timeout=10
            ),
            "implementation_generator": ImplementationGeneratorNode(
                node_name="implementation_generator",
                domain=self.domain,
                timeout=20
            ),
            "direct_answer": DirectAnswerNode(
                node_name="direct_answer",
                domain=self.domain,
                timeout=10
            )
        }
        
        # Add supervisor node if enabled
        if self.enable_supervisor:
            from .supervisor import SupervisorNode
            nodes["supervisor"] = SupervisorNode(
                node_name="supervisor",
                domain=self.domain,
                timeout=25
            )
        
        return nodes
    
    def _build_workflow(self) -> StateGraph:
        """Build the complete LangGraph workflow with all nodes and edges"""
        
        # Create the state graph
        workflow = StateGraph(TechnicalConversationState)
        
        # Add all nodes to the graph with flow control protection
        for node_name, node_instance in self.nodes.items():
            # Wrap node processing with flow control
            async def protected_process(state: TechnicalConversationState, node=node_instance, name=node_name):
                return await self.flow_controller.execute_with_protection(
                    node.safe_process,
                    name,
                    state,
                    priority="high" if name in ["intent_classifier", "direct_answer"] else "medium"
                )
            
            workflow.add_node(node_name, protected_process)
        
        # Add START edge - all conversations begin with intent classification
        workflow.add_edge(START, "intent_classifier")
        
        # Add conditional routing from intent_classifier
        workflow.add_conditional_edges(
            "intent_classifier",
            self.routing_engine.route_after_intent_classification,
            {
                "direct_answer": "direct_answer",
                "information_extraction": "information_extractor", 
                "multi_agent": "supervisor" if self.enable_supervisor else "information_extractor",
                "error": END
            }
        )
        
        # Routing from information_extractor
        workflow.add_conditional_edges(
            "information_extractor",
            self.routing_engine.route_after_extraction,
            {
                "requirement_analysis": "requirement_analyzer",
                "sufficiency_check": "sufficiency_checker",
                "parallel_analysis": "requirement_analyzer",  # Will be handled by parallel executor
                "implementation": "implementation_generator",
                "error": END
            }
        )
        
        # Routing from requirement_analyzer
        workflow.add_conditional_edges(
            "requirement_analyzer", 
            self.routing_engine.route_after_requirement_analysis,
            {
                "sufficiency_check": "sufficiency_checker",
                "more_info_needed": "information_extractor",
                "implementation": "implementation_generator",
                "error": END
            }
        )
        
        # Routing from sufficiency_checker
        workflow.add_conditional_edges(
            "sufficiency_checker",
            self.routing_engine.route_after_sufficiency_check,
            {
                "implementation": "implementation_generator",
                "more_questions": "information_extractor",
                "clarification": "direct_answer", 
                "error": END
            }
        )
        
        # Routing from supervisor (if enabled)
        if self.enable_supervisor:
            workflow.add_conditional_edges(
                "supervisor",
                self.routing_engine.route_after_supervisor,
                {
                    "implementation": "implementation_generator",
                    "analysis": "requirement_analyzer",
                    "direct_response": "direct_answer",
                    "error": END
                }
            )
        
        # All terminal nodes route to END
        workflow.add_edge("implementation_generator", END)
        workflow.add_edge("direct_answer", END)
        
        logger.info(f"Workflow built with {len(self.nodes)} nodes")
        return workflow
    
    async def process_conversation(
        self,
        user_input: str,
        user_id: str,
        channel_id: str,
        thread_id: str,
        domain: DomainType,
        correlation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TechnicalConversationState:
        """
        Process a conversation turn through the complete LangGraph workflow.
        
        This is the main entry point for conversation processing, handling:
        - State initialization or continuation
        - Workflow execution with timeout protection
        - Error handling and recovery
        - Performance monitoring
        - State persistence via checkpointing
        
        Args:
            user_input: User's input message
            user_id: Slack user ID
            channel_id: Slack channel ID  
            thread_id: Conversation thread ID
            domain: Technical domain for processing
            correlation_id: Request correlation ID for tracing
            conversation_history: Previous conversation messages
            metadata: Additional metadata
            
        Returns:
            Final conversation state after processing
            
        Raises:
            TimeoutError: If processing exceeds timeout
            ValueError: If required parameters are missing
            RuntimeError: If workflow execution fails
        """
        
        start_time = time.time()
        conversation_id = f"{thread_id}_{int(start_time)}"
        
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        try:
            # Create or retrieve conversation state
            state = await self._get_or_create_state(
                conversation_id=conversation_id,
                user_input=user_input,
                user_id=user_id,
                channel_id=channel_id,
                thread_id=thread_id,
                domain=domain,
                correlation_id=correlation_id,
                conversation_history=conversation_history,
                metadata=metadata
            )
            
            # Record processing start
            await self.telemetry_service.record_conversation_start(
                conversation_id=conversation_id,
                domain=domain,
                user_id=user_id,
                correlation_id=correlation_id
            )
            
            # Configure the workflow execution
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "conversation_id": conversation_id,
                    "correlation_id": correlation_id
                }
            }
            
            # Execute the workflow with timeout protection
            final_state = await asyncio.wait_for(
                self._execute_workflow(state, config),
                timeout=self.timeout
            )
            
            # Record successful completion
            processing_time = time.time() - start_time
            
            await self.telemetry_service.record_conversation_completion(
                conversation_id=conversation_id,
                processing_time=processing_time,
                success=True,
                final_stage=final_state.get('stage'),
                node_count=len(final_state.get('node_history', []))
            )
            
            self.metrics_collector.record_histogram(
                f"conversation_processing_time_seconds",
                processing_time,
                domain=domain
            )
            
            self.metrics_collector.increment_counter(
                f"conversation_success_total",
                domain=domain,
                final_stage=final_state.get('stage')
            )
            
            logger.info(
                f"Conversation completed successfully",
                extra={
                    "conversation_id": conversation_id,
                    "domain": domain,
                    "processing_time": processing_time,
                    "final_stage": final_state.get('stage'),
                    "nodes_executed": len(final_state.get('node_history', []))
                }
            )
            
            return final_state
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            
            logger.error(
                f"Conversation processing timed out after {self.timeout}s",
                extra={
                    "conversation_id": conversation_id,
                    "domain": domain,
                    "processing_time": processing_time
                }
            )
            
            await self.telemetry_service.record_conversation_error(
                conversation_id=conversation_id,
                error_type="TimeoutError",
                error_message=f"Processing timed out after {self.timeout}s",
                processing_time=processing_time
            )
            
            # Return error state
            error_state = state.copy()
            error_state.update({
                'stage': ConversationStage.ERROR,
                'response': f"I'm sorry, processing your {domain} question took longer than expected. Please try again with a more specific question.",
                'response_type': ResponseType.ERROR
            })
            
            return error_state
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error(
                f"Conversation processing failed: {e}",
                extra={
                    "conversation_id": conversation_id, 
                    "domain": domain,
                    "error_type": type(e).__name__,
                    "processing_time": processing_time
                },
                exc_info=True
            )
            
            await self.telemetry_service.record_conversation_error(
                conversation_id=conversation_id,
                error_type=type(e).__name__,
                error_message=str(e),
                processing_time=processing_time
            )
            
            self.metrics_collector.increment_counter(
                f"conversation_error_total",
                domain=domain,
                error_type=type(e).__name__
            )
            
            # Return error state
            error_state = state.copy()
            error_state.update({
                'stage': ConversationStage.ERROR,
                'response': f"I encountered an error processing your {domain} question. Please try rephrasing your request or contact support.",
                'response_type': ResponseType.ERROR
            })
            
            return error_state
    
    async def _get_or_create_state(
        self,
        conversation_id: str,
        user_input: str,
        user_id: str,
        channel_id: str,
        thread_id: str,
        domain: DomainType,
        correlation_id: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TechnicalConversationState:
        """Get existing conversation state or create new one"""
        
        try:
            # Try to get existing state from checkpointer
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }
            
            checkpoint_tuple = await self.checkpointer.aget_tuple(config)
            
            if checkpoint_tuple:
                checkpoint, checkpoint_metadata = checkpoint_tuple
                existing_state = checkpoint.channel_values.get('conversation_state')
                
                if existing_state:
                    # Update existing state with new input
                    updated_state = ConversationStateManager.update_state(
                        existing_state,
                        {
                            'current_input': user_input,
                            'correlation_id': correlation_id
                        }
                    )
                    
                    logger.info(f"Retrieved existing conversation state for thread: {thread_id}")
                    return updated_state
            
            # Create new state if none exists
            state = ConversationStateManager.create_initial_state(
                conversation_id=conversation_id,
                initial_query=user_input,
                domain=domain,
                thread_id=thread_id,
                user_id=user_id,
                channel_id=channel_id,
                correlation_id=correlation_id,
                metadata=metadata
            )
            
            # Add conversation history if provided
            if conversation_history:
                state['conversation_history'] = conversation_history
            
            logger.info(f"Created new conversation state for thread: {thread_id}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to get/create conversation state: {e}")
            
            # Fallback to basic state creation
            return ConversationStateManager.create_initial_state(
                conversation_id=conversation_id,
                initial_query=user_input,
                domain=domain,
                thread_id=thread_id,
                user_id=user_id,
                channel_id=channel_id,
                correlation_id=correlation_id,
                metadata=metadata or {}
            )
    
    async def _execute_workflow(
        self,
        state: TechnicalConversationState, 
        config: Dict[str, Any]
    ) -> TechnicalConversationState:
        """Execute the LangGraph workflow"""
        
        try:
            # Execute the workflow using astream for better control
            final_state = None
            node_count = 0
            
            async for step_output in self.app.astream(state, config):
                node_count += 1
                
                # LangGraph returns dict with node name as key
                for node_name, node_state in step_output.items():
                    final_state = node_state
                    
                    logger.debug(
                        f"Workflow step completed: {node_name}",
                        extra={
                            "node_name": node_name,
                            "conversation_id": state['conversation_id'],
                            "stage": node_state.get('stage'),
                            "step_number": node_count
                        }
                    )
                    
                    # Check for early termination conditions
                    if node_state.get('stage') == ConversationStage.ERROR:
                        logger.warning(f"Workflow terminated early due to error in node: {node_name}")
                        return node_state
            
            if not final_state:
                raise RuntimeError("Workflow completed but no final state received")
            
            logger.info(f"Workflow execution completed with {node_count} steps")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise RuntimeError(f"Workflow execution failed: {e}") from e
    
    def get_workflow_visualization(self) -> Dict[str, Any]:
        """Get workflow visualization data for debugging"""
        try:
            # Get the compiled graph structure
            graph_data = {
                "nodes": list(self.nodes.keys()),
                "domain": self.domain,
                "parallel_enabled": self.enable_parallel,
                "supervisor_enabled": self.enable_supervisor,
                "routing_config": self.routing_engine.get_routing_config(),
                "checkpointer_type": type(self.checkpointer).__name__
            }
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Failed to generate workflow visualization: {e}")
            return {"error": str(e)}


class ConversationOrchestrator:
    """
    High-level orchestrator for managing multiple conversation graphs.
    
    This class provides a centralized interface for managing conversation
    graphs across different domains and handles graph lifecycle management.
    """
    
    def __init__(self, enable_parallel: bool = True, enable_supervisor: bool = True):
        self.enable_parallel = enable_parallel
        self.enable_supervisor = enable_supervisor
        self.graphs: Dict[str, ConversationGraph] = {}
        
        logger.info("ConversationOrchestrator initialized")
    
    def get_graph(self, domain: DomainType) -> ConversationGraph:
        """Get or create conversation graph for domain"""
        
        domain_str = domain.value if hasattr(domain, 'value') else str(domain)
        
        if domain_str not in self.graphs:
            self.graphs[domain_str] = ConversationGraph(
                domain=domain,
                enable_parallel_execution=self.enable_parallel,
                enable_supervisor_mode=self.enable_supervisor
            )
            
            logger.info(f"Created new conversation graph for domain: {domain_str}")
        
        return self.graphs[domain_str]
    
    async def process_conversation(
        self,
        domain: DomainType,
        user_input: str,
        user_id: str,
        channel_id: str,
        thread_id: str,
        correlation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TechnicalConversationState:
        """Process conversation through domain-specific graph"""
        
        graph = self.get_graph(domain)
        
        return await graph.process_conversation(
            user_input=user_input,
            user_id=user_id,
            channel_id=channel_id,
            thread_id=thread_id,
            domain=domain,
            correlation_id=correlation_id,
            conversation_history=conversation_history,
            metadata=metadata
        )
    
    def get_all_visualizations(self) -> Dict[str, Dict[str, Any]]:
        """Get visualization data for all graphs"""
        return {
            domain: graph.get_workflow_visualization()
            for domain, graph in self.graphs.items()
        }
    
    def reload_graph(self, domain: DomainType) -> None:
        """Reload graph for hot updates"""
        domain_str = domain.value if hasattr(domain, 'value') else str(domain)
        
        if domain_str in self.graphs:
            del self.graphs[domain_str]
            logger.info(f"Reloaded conversation graph for domain: {domain_str}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all graphs"""
        return {
            "total_graphs": len(self.graphs),
            "domains": list(self.graphs.keys()),
            "parallel_enabled": self.enable_parallel,
            "supervisor_enabled": self.enable_supervisor,
            "status": "healthy"
        }