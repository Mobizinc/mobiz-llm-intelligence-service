"""
StreamingGraphExecutor - Orchestration service for managing LangGraph agentic workflow execution
and translating step-by-step reasoning into real-time streams.

This service bridges the existing LangGraph architecture with streaming requirements,
providing real-time updates during complex AI processing workflows.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from ..models.conversation import ConversationRequest, ConversationContext, ConversationMessage
from ..models.streaming import (
    StreamMessage, StreamTextChunk, StreamUIComponent, StreamStatusUpdate, 
    StreamError, StreamComplete, StreamConfig
)
from ..models.conversation_state import TechnicalConversationState, DomainType, IntentType
from ..nodes.intent_classifier import IntentClassifierNode
from ..nodes.information_extractor import InformationExtractorNode
from ..nodes.implementation_generator import ImplementationGeneratorNode
from ..nodes.direct_answer import DirectAnswerNode
from ..services.client_context_service import ClientContextService

logger = logging.getLogger(__name__)


class StreamingGraphExecutor:
    """
    Conductor service for managing LangGraph agentic workflow execution with streaming support.
    
    Orchestrates the execution of pre-computation nodes (IntentClassifier, InformationExtractor, etc.)
    and provides real-time stream updates throughout the reasoning process.
    """
    
    def __init__(self, stream_config: Optional[StreamConfig] = None):
        """Initialize the streaming executor with configuration"""
        self.stream_config = stream_config or StreamConfig()
        self.client_context_service = ClientContextService()
        
        # Initialize nodes
        self.intent_classifier = IntentClassifierNode()
        self.information_extractor = InformationExtractorNode()
        self.implementation_generator = ImplementationGeneratorNode()
        self.direct_answer = DirectAnswerNode()
        
        # Execution tracking
        self.current_execution_id: Optional[str] = None
        self.execution_start_time: Optional[datetime] = None
        self.cancellation_event: Optional[asyncio.Event] = None
    
    async def stream_execution(
        self, 
        request: ConversationRequest,
        correlation_id: Optional[str] = None
    ) -> AsyncGenerator[StreamMessage, None]:
        """
        Main streaming execution method that orchestrates the complete LangGraph workflow.
        
        Args:
            request: Conversation request containing messages and context
            correlation_id: Optional correlation ID for request tracing
            
        Yields:
            StreamMessage: Real-time updates during execution
        """
        execution_id = f"exec_{datetime.now().isoformat()}_{request.user_id}"
        self.current_execution_id = execution_id
        self.execution_start_time = datetime.now()
        self.cancellation_event = asyncio.Event()
        
        # Enhanced logging with correlation ID
        logger.info(
            f"Starting streaming execution for user {request.user_id}",
            extra={
                "correlation_id": correlation_id or "no-correlation-id",
                "execution_id": execution_id,
                "user_id": request.user_id,
                "bot_type": request.bot_type,
                "conversation_id": request.conversation_id,
                "message_count": len(request.messages) if request.messages else 0
            }
        )
        
        try:
            # Execute workflow with cancellation support
            async for message in self._execute_workflow(request, execution_id, correlation_id):
                # Check for cancellation
                if self.cancellation_event and self.cancellation_event.is_set():
                    logger.warning(
                        f"Execution cancelled by user request",
                        extra={
                            "correlation_id": correlation_id or "no-correlation-id",
                            "execution_id": execution_id,
                            "user_id": request.user_id
                        }
                    )
                    yield StreamError(
                        error_code="execution_cancelled",
                        message="Execution was cancelled by user request"
                    )
                    return
                yield message
                
        except asyncio.TimeoutError:
            logger.error(
                f"Execution timed out after {self.stream_config.max_stream_duration}s",
                extra={
                    "correlation_id": correlation_id or "no-correlation-id",
                    "execution_id": execution_id,
                    "user_id": request.user_id,
                    "timeout_duration": self.stream_config.max_stream_duration
                }
            )
            yield StreamError(
                error_code="execution_timeout",
                message=f"Execution exceeded maximum duration of {self.stream_config.max_stream_duration} seconds"
            )
            
        except asyncio.CancelledError:
            logger.info(
                f"Execution was cancelled",
                extra={
                    "correlation_id": correlation_id or "no-correlation-id",
                    "execution_id": execution_id,
                    "user_id": request.user_id
                }
            )
            yield StreamError(
                error_code="execution_cancelled", 
                message="Execution was cancelled"
            )
            
        except Exception as e:
            logger.error(
                f"Error in execution: {e}",
                extra={
                    "correlation_id": correlation_id or "no-correlation-id",
                    "execution_id": execution_id,
                    "user_id": request.user_id,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            yield StreamError(
                error_code="execution_error",
                message=f"Execution failed: {str(e)}"
            )
            
        finally:
            # Clean up execution state
            self.current_execution_id = None
            self.execution_start_time = None
            self.cancellation_event = None
    
    async def _execute_workflow(
        self, 
        request: ConversationRequest,
        execution_id: str,
        correlation_id: Optional[str] = None
    ) -> AsyncGenerator[StreamMessage, None]:
        """Internal workflow execution with timeout and cancellation support"""
        try:
            # Initialize conversation state
            initial_state = await self._initialize_conversation_state(request)
            
            # Yield initialization status
            if self.stream_config.enable_status_updates:
                yield StreamStatusUpdate(
                    status="initializing",
                    progress=0.0,
                    operation="Setting up conversation context",
                    details=f"Processing {len(request.messages)} messages"
                )
            
            # Load client context if client_id provided
            if request.client_id:
                client_context = await self.client_context_service.load_client_context(request.client_id)
                initial_state['client_context'] = client_context
                
                if self.stream_config.enable_ui_components:
                    yield StreamUIComponent(
                        component="ContextLoadedCard",
                        props={
                            "client_id": request.client_id,
                            "context_keys": list(client_context.keys()) if client_context else [],
                            "status": "loaded"
                        }
                    )
            
            # Phase 1: Pre-computation nodes
            current_state = initial_state
            processing_phases = [
                ("intent_classification", self.intent_classifier, 20.0),
                ("information_extraction", self.information_extractor, 40.0)
            ]
            
            for phase_name, node, progress in processing_phases:
                if self.stream_config.enable_status_updates:
                    yield StreamStatusUpdate(
                        status="processing",
                        progress=progress,
                        operation=f"Running {phase_name}",
                        details=f"Analyzing message intent and context"
                    )
                
                # Execute node with timeout and cancellation check
                try:
                    # Check for cancellation before node execution
                    if self.cancellation_event and self.cancellation_event.is_set():
                        return
                        
                    current_state = await asyncio.wait_for(
                        node.process(current_state),
                        timeout=30.0  # Per-node timeout
                    )
                    
                    # Yield node completion update
                    if self.stream_config.enable_ui_components:
                        yield StreamUIComponent(
                            component="NodeCompletionCard",
                            props={
                                "node_name": phase_name,
                                "status": "completed",
                                "metadata": self._extract_node_metadata(current_state, phase_name)
                            }
                        )
                        
                except Exception as e:
                    logger.error(
                        f"Error in {phase_name}: {str(e)}",
                        extra={
                            "correlation_id": correlation_id or "no-correlation-id",
                            "execution_id": execution_id,
                            "phase": phase_name,
                            "node_class": node.__class__.__name__,
                            "error_type": type(e).__name__
                        },
                        exc_info=True
                    )
                    yield StreamError(
                        error_code=f"{phase_name}_error",
                        message=f"Error during {phase_name}: {str(e)}",
                        details={"phase": phase_name, "node": node.__class__.__name__}
                    )
                    return
            
            # Phase 2: Determine response strategy
            response_strategy = self._determine_response_strategy(current_state)
            
            if self.stream_config.enable_status_updates:
                yield StreamStatusUpdate(
                    status="generating_response",
                    progress=60.0,
                    operation=f"Generating {response_strategy} response",
                    details="Processing with specialized reasoning node"
                )
            
            # Phase 3: Stream final response generation
            response_node = self._get_response_node(response_strategy)
            
            async for chunk in self._stream_final_response(current_state, response_node):
                yield chunk
                
            # Execution completed
            execution_time = (datetime.now() - self.execution_start_time).total_seconds()
            
            logger.info(
                f"Streaming execution completed successfully",
                extra={
                    "correlation_id": correlation_id or "no-correlation-id", 
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "response_strategy": response_strategy,
                    "user_id": request.user_id,
                    "bot_type": request.bot_type
                }
            )
            
            yield StreamComplete(
                final_response="Response generation completed",
                metadata={
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "total_tokens": current_state.get('token_usage', {}),
                    "response_strategy": response_strategy,
                    "correlation_id": correlation_id
                }
            )
            
        except Exception as e:
            logger.error(
                f"Execution error: {str(e)}",
                extra={
                    "correlation_id": correlation_id or "no-correlation-id",
                    "execution_id": execution_id,
                    "error_type": type(e).__name__,
                    "user_id": request.user_id,
                    "bot_type": request.bot_type
                },
                exc_info=True
            )
            yield StreamError(
                error_code="execution_error",
                message=f"Execution failed: {str(e)}",
                details={"execution_id": execution_id, "correlation_id": correlation_id}
            )
        
        finally:
            self.current_execution_id = None
            self.execution_start_time = None
    
    async def _initialize_conversation_state(
        self, 
        request: ConversationRequest
    ) -> TechnicalConversationState:
        """Initialize conversation state from request"""
        
        # Get the latest user message
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        current_input = user_messages[-1].content if user_messages else ""
        initial_query = current_input
        
        # Build conversation history
        conversation_history = []
        for i in range(0, len(request.messages) - 1, 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                bot_msg = request.messages[i + 1] if i + 1 < len(request.messages) else None
                
                if user_msg.role == "user":
                    history_entry = {"user": user_msg.content}
                    if bot_msg and bot_msg.role == "assistant":
                        history_entry["bot"] = bot_msg.content
                    conversation_history.append(history_entry)
        
        # Determine domain from bot_type
        domain_mapping = {
            "cloud": "cloud",
            "devops": "devops", 
            "network": "network",
            "dev": "dev"
        }
        domain = domain_mapping.get(request.bot_type, "general")
        
        # Create initial state
        state = TechnicalConversationState(
            conversation_id=request.conversation_id or f"conv_{datetime.now().isoformat()}",
            user_id=request.user_id,
            initial_query=initial_query,
            current_input=current_input,
            domain=domain,
            conversation_history=conversation_history,
            thread_id=request.conversation_id,
            questions_asked=[],
            extracted_info={},
            accumulated_info={},
            missing_fields=[],
            response="",
            response_type="processing",
            intent=IntentType.NEW_QUERY,
            confidence_score=0.0,
            token_usage={},
            cache_hits=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=request.metadata
        )
        
        return state
    
    def _extract_node_metadata(self, state: TechnicalConversationState, phase_name: str) -> Dict[str, Any]:
        """Extract relevant metadata from state for UI display"""
        
        metadata = {}
        
        if phase_name == "intent_classification":
            metadata = {
                "intent": state.get('intent'),
                "confidence": state.get('confidence_score'),
                "reasoning": state.get('intent_metadata', {}).get('reasoning', ''),
                "requires_clarification": state.get('intent_metadata', {}).get('requires_clarification', False)
            }
        
        elif phase_name == "information_extraction":
            extracted_info = state.get('extracted_info', {})
            extraction_summary = {}
            
            # Summarize extracted entities by category
            for category, entities in extracted_info.items():
                if isinstance(entities, list) and len(entities) > 0:
                    extraction_summary[category] = len(entities)
                elif category == 'extraction_metadata':
                    extraction_summary['confidence'] = entities.get('extraction_confidence', 0.0)
            
            metadata = {
                "extracted_categories": list(extraction_summary.keys()),
                "extraction_summary": extraction_summary,
                "total_entities": sum(v for v in extraction_summary.values() if isinstance(v, int))
            }
        
        return metadata
    
    def _determine_response_strategy(self, state: TechnicalConversationState) -> str:
        """Determine which response node to use based on state analysis"""
        
        intent = state.get('intent')
        extracted_info = state.get('extracted_info', {})
        
        # Check if clarification is needed
        intent_metadata = state.get('intent_metadata', {})
        if intent_metadata.get('requires_clarification', False):
            return "clarification"
        
        # Check if we have enough technical information for implementation
        has_technical_info = any(
            len(extracted_info.get(category, [])) > 0
            for category in ['technologies', 'frameworks', 'technical_requirements', 'platforms']
        )
        
        has_implementation_intent = intent in [IntentType.IMPLEMENTATION, IntentType.NEW_QUERY]
        
        if has_implementation_intent and has_technical_info:
            return "implementation"
        else:
            return "direct_answer"
    
    def _get_response_node(self, strategy: str):
        """Get the appropriate response node based on strategy"""
        
        node_mapping = {
            "implementation": self.implementation_generator,
            "direct_answer": self.direct_answer,
            "clarification": self.direct_answer  # Handle clarification as direct answer
        }
        
        return node_mapping.get(strategy, self.direct_answer)
    
    async def _stream_final_response(
        self, 
        state: TechnicalConversationState, 
        response_node
    ) -> AsyncGenerator[StreamMessage, None]:
        """Stream the final response generation from the selected node"""
        
        try:
            # Check if node supports streaming
            if hasattr(response_node, 'stream_process'):
                # Use streaming method
                async for text_chunk in response_node.stream_process(state):
                    yield StreamTextChunk(
                        content=text_chunk,
                        metadata={"node": response_node.__class__.__name__}
                    )
            else:
                # Fallback to non-streaming with simulated chunking
                if self.stream_config.enable_status_updates:
                    yield StreamStatusUpdate(
                        status="generating",
                        progress=80.0,
                        operation="Generating response",
                        details="Processing with legacy node (non-streaming)"
                    )
                
                # Process normally and chunk the result
                processed_state = await response_node.process(state)
                response_content = processed_state.get('response', '')
                
                # Simulate streaming by chunking the response
                chunk_size = self.stream_config.chunk_size
                for i in range(0, len(response_content), chunk_size):
                    chunk = response_content[i:i + chunk_size]
                    yield StreamTextChunk(
                        content=chunk,
                        metadata={"node": response_node.__class__.__name__, "chunk_index": i // chunk_size}
                    )
                    
                    # Small delay to simulate real streaming
                    if self.stream_config.buffer_timeout > 0:
                        await asyncio.sleep(self.stream_config.buffer_timeout)
                        
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield StreamError(
                error_code="response_generation_error",
                message=f"Error generating response: {str(e)}",
                details={"node": response_node.__class__.__name__}
            )
    
    async def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status for monitoring"""
        
        if not self.current_execution_id:
            return {"status": "idle", "execution_id": None}
        
        execution_time = (
            (datetime.now() - self.execution_start_time).total_seconds() 
            if self.execution_start_time else 0
        )
        
        return {
            "status": "running",
            "execution_id": self.current_execution_id,
            "execution_time": execution_time,
            "max_duration": self.stream_config.max_stream_duration
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        
        if self.current_execution_id == execution_id and self.cancellation_event:
            logger.info(f"Cancelling execution {execution_id}")
            # Set the cancellation event to signal cooperative cancellation
            self.cancellation_event.set()
            return True
        
        return False