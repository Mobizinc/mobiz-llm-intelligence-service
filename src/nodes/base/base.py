"""
Base Node Implementation for LangGraph Agent Nodes

This module provides the abstract base class and common functionality for all
agent nodes in the Technical Bots system. All specialized nodes inherit from
BaseNode and implement the process method.

Part of Epic 2: Agent Nodes Implementation
Story 2.x: Base Node Architecture
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, AsyncGenerator
from datetime import datetime, timezone
import logging
import uuid

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

from ...models.conversation_state import TechnicalConversationState, ConversationStage
from ...core.config import settings
from ...monitoring.state_metrics import StateMetricsCollector
from ...telemetry.state_telemetry import StateOperationTracer

logger = logging.getLogger(__name__)


class NodeError(Exception):
    """Base exception for node processing errors"""
    pass


class NodeTimeoutError(NodeError):
    """Raised when node processing exceeds timeout"""
    pass


class NodeValidationError(NodeError):
    """Raised when node input/output validation fails"""
    pass


class CircuitBreakerOpen(NodeError):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for LLM calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
        
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class BaseNode(ABC):
    """
    Abstract base class for all LangGraph agent nodes.
    
    Provides common functionality including:
    - LLM client management with circuit breaker
    - Telemetry and metrics collection
    - Error handling and retry logic
    - Performance monitoring
    - State validation
    """
    
    def __init__(
        self,
        node_name: str,
        domain: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        enable_cache: bool = True
    ):
        self.node_name = node_name
        self.domain = domain or "general"
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        
        # Initialize LLM client with circuit breaker
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000,
            timeout=timeout,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )
        
        # Circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker()
        
        # Telemetry services
        self.metrics_collector = StateMetricsCollector()
        self.telemetry_service = StateOperationTracer()
        
        # Cache for node results (simple in-memory cache)
        self._cache: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.node_name} node for domain: {self.domain}")
    
    @abstractmethod
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process the conversation state through this node.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
            
        Raises:
            NodeError: If processing fails
            NodeTimeoutError: If processing exceeds timeout
            NodeValidationError: If validation fails
        """
        pass
    
    async def safe_process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Safely process state with error handling, retries, and telemetry.
        
        This is the main entry point for node processing, providing:
        - Pre/post processing hooks
        - Error handling and retries
        - Performance monitoring
        - State validation
        """
        start_time = time.time()
        node_id = str(uuid.uuid4())
        
        try:
            # Pre-processing validation
            self._validate_input_state(state)
            
            # Add node to processing history
            updated_state = self._add_to_node_history(state, "started")
            
            # Record node start
            await self.telemetry_service.record_node_start(
                node_name=self.node_name,
                node_id=node_id,
                conversation_id=updated_state['conversation_id'],
                state_version=updated_state.get('version', 1)
            )
            
            # Process with retries
            result_state = await self._process_with_retries(updated_state)
            
            # Post-processing validation
            self._validate_output_state(result_state)
            
            # Record successful completion
            processing_time = time.time() - start_time
            result_state = self._update_performance_metrics(result_state, processing_time)
            result_state = self._add_to_node_history(result_state, "completed")
            
            await self.telemetry_service.record_node_completion(
                node_name=self.node_name,
                node_id=node_id,
                conversation_id=result_state['conversation_id'],
                processing_time=processing_time,
                success=True
            )
            
            self.metrics_collector.increment_counter(
                f"{self.node_name}_success_total",
                domain=self.domain
            )
            
            logger.info(
                f"Node {self.node_name} completed successfully",
                extra={
                    "node_name": self.node_name,
                    "processing_time": processing_time,
                    "conversation_id": result_state['conversation_id']
                }
            )
            
            return result_state
            
        except Exception as e:
            # Record failure
            processing_time = time.time() - start_time
            error_state = self._handle_node_error(state, e, processing_time)
            
            await self.telemetry_service.record_node_error(
                node_name=self.node_name,
                node_id=node_id,
                conversation_id=state['conversation_id'],
                error_type=type(e).__name__,
                error_message=str(e),
                processing_time=processing_time
            )
            
            self.metrics_collector.increment_counter(
                f"{self.node_name}_error_total",
                domain=self.domain,
                error_type=type(e).__name__
            )
            
            logger.error(
                f"Node {self.node_name} failed: {e}",
                extra={
                    "node_name": self.node_name,
                    "error_type": type(e).__name__,
                    "processing_time": processing_time,
                    "conversation_id": state['conversation_id']
                },
                exc_info=True
            )
            
            return error_state
    
    async def _process_with_retries(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """Process with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check circuit breaker
                if not self.circuit_breaker.can_execute():
                    raise CircuitBreakerOpen(f"Circuit breaker open for {self.node_name}")
                
                # Process with timeout
                result = await asyncio.wait_for(
                    self.process(state),
                    timeout=self.timeout
                )
                
                # Record success
                self.circuit_breaker.record_success()
                return result
                
            except asyncio.TimeoutError:
                last_exception = NodeTimeoutError(f"Node {self.node_name} timed out after {self.timeout}s")
                self.circuit_breaker.record_failure()
                
            except Exception as e:
                last_exception = e
                self.circuit_breaker.record_failure()
                
                # Don't retry on validation errors
                if isinstance(e, NodeValidationError):
                    raise
                    
                logger.warning(
                    f"Node {self.node_name} attempt {attempt + 1} failed: {e}",
                    extra={
                        "node_name": self.node_name,
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries
                    }
                )
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
        
        # All retries exhausted
        raise last_exception or NodeError(f"All retries exhausted for {self.node_name}")
    
    def _validate_input_state(self, state: TechnicalConversationState) -> None:
        """Validate input state"""
        required_fields = ['conversation_id', 'current_input', 'domain', 'intent']
        
        for field in required_fields:
            if field not in state or not state[field]:
                raise NodeValidationError(f"Required field '{field}' missing or empty in state")
    
    def _validate_output_state(self, state: TechnicalConversationState) -> None:
        """Validate output state - to be overridden by specific nodes"""
        # Base validation - ensure state is still valid
        self._validate_input_state(state)
    
    def _add_to_node_history(self, state: TechnicalConversationState, status: str) -> TechnicalConversationState:
        """Add node processing to history"""
        updated_state = state.copy()
        node_history = updated_state.get('node_history', []).copy()
        timestamp = datetime.now(timezone.utc).isoformat()
        node_history.append(f"{self.node_name}:{status}:{timestamp}")
        updated_state['node_history'] = node_history
        return updated_state
    
    def _update_performance_metrics(self, state: TechnicalConversationState, processing_time: float) -> TechnicalConversationState:
        """Update performance metrics in state"""
        updated_state = state.copy()
        processing_times = updated_state.get('processing_times', {}).copy()
        processing_times[self.node_name] = processing_time
        updated_state['processing_times'] = processing_times
        
        # Record histogram metric
        self.metrics_collector.record_histogram(
            f"{self.node_name}_processing_time_seconds",
            processing_time,
            domain=self.domain
        )
        
        return updated_state
    
    def _handle_node_error(self, state: TechnicalConversationState, error: Exception, processing_time: float) -> TechnicalConversationState:
        """Handle node processing error"""
        updated_state = state.copy()
        
        # Add error to history
        error_history = updated_state.get('error_history', []).copy()
        timestamp = datetime.now(timezone.utc).isoformat()
        error_history.append(f"{timestamp}: {self.node_name}: {type(error).__name__}: {str(error)}")
        updated_state['error_history'] = error_history
        
        # Update stage to error if critical failure
        if isinstance(error, (NodeTimeoutError, CircuitBreakerOpen)):
            updated_state['stage'] = ConversationStage.ERROR
            
        # Add performance metrics even for failures
        updated_state = self._update_performance_metrics(updated_state, processing_time)
        
        return updated_state
    
    def _get_cache_key(self, state: TechnicalConversationState, additional_keys: Optional[List[str]] = None) -> str:
        """Generate cache key for state"""
        key_parts = [
            self.node_name,
            state['conversation_id'],
            state.get('current_input', ''),
            state.get('intent', ''),
            str(state.get('version', 1))
        ]
        
        if additional_keys:
            key_parts.extend(additional_keys)
        
        return "|".join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache"""
        if not self.enable_cache:
            return None
        
        result = self._cache.get(cache_key)
        if result:
            self.metrics_collector.increment_counter(
                f"{self.node_name}_cache_hit_total",
                domain=self.domain
            )
        
        return result
    
    def _set_cache(self, cache_key: str, result: Any, ttl: int = 3600) -> None:
        """Set result in cache"""
        if not self.enable_cache:
            return
        
        # Simple TTL implementation - store timestamp
        self._cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'ttl': ttl
        }
        
        # Clean old entries (simple cleanup)
        current_time = time.time()
        expired_keys = [
            k for k, v in self._cache.items()
            if current_time - v['timestamp'] > v['ttl']
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    async def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        response_parser: Optional[PydanticOutputParser] = None,
        **kwargs
    ) -> Any:
        """
        Call LLM with circuit breaker and error handling.
        
        Args:
            system_prompt: System prompt for the LLM
            user_message: User message to process
            response_parser: Optional Pydantic parser for structured output
            **kwargs: Additional arguments for LLM
            
        Returns:
            LLM response (parsed if parser provided)
        """
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpen(f"Circuit breaker open for {self.node_name} LLM calls")
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            # Create chain with appropriate parser
            if response_parser:
                chain = self.llm | response_parser
            else:
                chain = self.llm | StrOutputParser()
            
            # Invoke the chain
            response = await chain.ainvoke(messages, **kwargs)
            
            self.circuit_breaker.record_success()
            
            self.metrics_collector.increment_counter(
                f"{self.node_name}_llm_call_total",
                domain=self.domain,
                status="success"
            )
            
            return response
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            
            self.metrics_collector.increment_counter(
                f"{self.node_name}_llm_call_total", 
                domain=self.domain,
                status="error",
                error_type=type(e).__name__
            )
            
            raise NodeError(f"LLM call failed in {self.node_name}: {e}") from e
    
    async def stream_llm(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response with circuit breaker and error handling.
        
        Args:
            system_prompt: System prompt for the LLM
            user_message: User message to process
            **kwargs: Additional arguments for LLM
            
        Yields:
            str: Text chunks from the streaming LLM response
        """
        if not self.circuit_breaker.can_execute():
            yield f"Circuit breaker open for {self.node_name} - please try again later."
            return
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            # Stream the response
            async for chunk in self.llm.astream(messages, **kwargs):
                # Extract content from chunk - LangChain returns AIMessageChunk
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
            
            self.circuit_breaker.record_success()
            
            self.metrics_collector.increment_counter(
                f"{self.node_name}_llm_stream_total",
                labels={"status": "success"}
            )
            
        except Exception as e:
            logger.error(f"LLM streaming failed in {self.node_name}: {e}")
            self.circuit_breaker.record_failure()
            
            self.metrics_collector.increment_counter(
                f"{self.node_name}_llm_stream_total",
                labels={"status": "error", "error_type": type(e).__name__}
            )
            
            yield f"Error generating response: {str(e)}"
    
    def get_domain_context(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific context for prompts"""
        domain_contexts = {
            "cloud": {
                "technologies": ["Azure", "AWS", "GCP", "Windows Server", "Linux"],
                "focus_areas": ["Infrastructure", "Scalability", "Security", "Cost Optimization"],
                "common_patterns": ["IaaS deployment", "PaaS services", "Hybrid cloud", "Migration"]
            },
            "network": {
                "technologies": ["Palo Alto", "Cisco", "Fortinet", "Meraki", "VPN"],
                "focus_areas": ["Security", "Performance", "Reliability", "Compliance"],
                "common_patterns": ["Site-to-site VPN", "Firewall rules", "Network segmentation", "Load balancing"]
            },
            "devops": {
                "technologies": ["Docker", "Kubernetes", "Jenkins", "GitHub Actions", "Terraform"],
                "focus_areas": ["Automation", "CI/CD", "Monitoring", "Infrastructure as Code"],
                "common_patterns": ["Pipeline deployment", "Container orchestration", "Infrastructure automation", "Monitoring setup"]
            },
            "dev": {
                "technologies": ["Python", "JavaScript", "React", "Node.js", "APIs"],
                "focus_areas": ["Code quality", "Performance", "Security", "Maintainability"],
                "common_patterns": ["API development", "Frontend/backend integration", "Database design", "Testing strategies"]
            }
        }
        
        return domain_contexts.get(domain, {
            "technologies": [],
            "focus_areas": ["General technical guidance"],
            "common_patterns": []
        })