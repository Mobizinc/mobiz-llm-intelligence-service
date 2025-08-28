"""
Simplified Telemetry for LangGraph State Operations

Lightweight telemetry and observability for conversation state operations.
"""

import logging
import time
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from functools import wraps

logger = logging.getLogger(__name__)


class StateOperationTracer:
    """
    Simple distributed tracing for state operations.
    
    Provides correlation IDs, timing, and context tracking
    across all state-related operations.
    """
    
    def __init__(self):
        # Active traces
        self.active_traces: Dict[str, Dict[str, Any]] = {}
        self.traces = []  # Legacy compatibility
        self.enabled = True  # Enable by default
    
    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        conversation_id: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        **context
    ):
        """
        Context manager for tracing state operations.
        
        Args:
            operation_name: Name of the operation being traced
            conversation_id: Conversation ID being processed
            trace_id: Optional existing trace ID
            parent_span_id: Optional parent span ID for nested operations
            **context: Additional context data
        """
        # Generate IDs
        trace_id = trace_id or str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create span context
        span_context = {
            'trace_id': trace_id,
            'span_id': span_id,
            'parent_span_id': parent_span_id,
            'operation_name': operation_name,
            'conversation_id': conversation_id,
            'start_time': start_time,
            'context': context
        }
        
        # Track active span
        self.active_traces[span_id] = span_context
        
        try:
            # Log operation start
            if self.enabled:
                logger.info(
                    f"State operation started: {operation_name}",
                    extra={
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent_span_id,
                        "operation_name": operation_name,
                        "conversation_id": conversation_id,
                        "timestamp": start_time,
                        **context
                    }
                )
            
            yield span_context
            
            # Calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Log successful completion
            if self.enabled:
                logger.info(
                    f"State operation completed: {operation_name} ({duration_ms:.2f}ms)",
                    extra={
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "operation_name": operation_name,
                        "conversation_id": conversation_id,
                        "duration_ms": duration_ms,
                        "timestamp": end_time,
                        "success": True,
                        **context
                    }
                )
            
        except Exception as e:
            # Calculate duration for failed operation
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Log operation failure
            if self.enabled:
                logger.error(
                    f"State operation failed: {operation_name} ({duration_ms:.2f}ms) - {str(e)}",
                    extra={
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "operation_name": operation_name,
                        "conversation_id": conversation_id,
                        "duration_ms": duration_ms,
                        "timestamp": end_time,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        **context
                    }
                )
            
            raise
        
        finally:
            # Clean up active trace
            self.active_traces.pop(span_id, None)
    
    def add_span_annotation(self, span_id: str, key: str, value: Any):
        """Add annotation to active span"""
        if span_id in self.active_traces:
            if 'annotations' not in self.active_traces[span_id]:
                self.active_traces[span_id]['annotations'] = {}
            self.active_traces[span_id]['annotations'][key] = value
            
            # Log annotation
            if self.enabled:
                logger.debug(
                    f"Span annotation added: {key}={value}",
                    extra={
                        "span_id": span_id,
                        "trace_id": self.active_traces[span_id]['trace_id'],
                        "annotation_key": key,
                        "annotation_value": str(value),
                        "timestamp": time.time()
                    }
                )
    
    def get_active_traces(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently active traces"""
        return self.active_traces.copy()
    
    # Legacy compatibility methods
    async def record_conversation_start(
        self,
        conversation_id: str,
        domain: str,
        user_id: str,
        correlation_id: str
    ):
        """Record the start of a conversation"""
        if self.enabled:
            trace = {
                "event": "conversation_start",
                "conversation_id": conversation_id,
                "domain": domain,
                "user_id": user_id,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.traces.append(trace)
            logger.debug(f"Conversation started: {conversation_id}")
    
    async def record_conversation_completion(
        self,
        conversation_id: str,
        processing_time: float,
        success: bool,
        final_stage: Optional[str] = None,
        node_count: int = 0
    ):
        """Record the completion of a conversation"""
        if self.enabled:
            trace = {
                "event": "conversation_completion",
                "conversation_id": conversation_id,
                "processing_time": processing_time,
                "success": success,
                "final_stage": final_stage,
                "node_count": node_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.traces.append(trace)
            logger.debug(f"Conversation completed: {conversation_id} in {processing_time}s")
    
    async def record_conversation_error(
        self,
        conversation_id: str,
        error_type: str,
        error_message: str,
        processing_time: float
    ):
        """Record an error in conversation processing"""
        if self.enabled:
            trace = {
                "event": "conversation_error",
                "conversation_id": conversation_id,
                "error_type": error_type,
                "error_message": error_message,
                "processing_time": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.traces.append(trace)
            logger.debug(f"Conversation error: {conversation_id} - {error_type}")
    
    def record_routing_decision(
        self,
        conversation_id: str,
        routing_point: str,
        decision: str,
        confidence: float,
        reasoning: str,
        metadata: Dict[str, Any]
    ):
        """Record a routing decision"""
        if self.enabled:
            trace = {
                "event": "routing_decision",
                "conversation_id": conversation_id,
                "routing_point": routing_point,
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "metadata": metadata,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.traces.append(trace)
            logger.debug(f"Routing decision at {routing_point}: {decision}")
    
    def get_traces(self) -> list:
        """Get all collected traces"""
        return self.traces.copy()
    
    def reset(self):
        """Reset all traces"""
        self.traces = []
        self.active_traces = {}
        logger.debug("Telemetry traces reset")


class StateMetricsReporter:
    """
    Simple metrics reporting for state operations.
    
    Provides basic metrics tracking and reporting functionality.
    """
    
    def __init__(self):
        # Simple in-memory metrics storage
        self.metrics: Dict[str, Any] = {
            'operations': {},
            'errors': {},
            'conversations': {},
            'last_updated': time.time()
        }
    
    def record_operation(
        self,
        operation_type: str,
        conversation_id: str,
        duration_ms: float,
        success: bool = True,
        **metadata
    ):
        """Record an operation metric"""
        current_time = time.time()
        
        # Initialize operation metrics if not exists
        if operation_type not in self.metrics['operations']:
            self.metrics['operations'][operation_type] = {
                'count': 0,
                'total_duration_ms': 0,
                'success_count': 0,
                'error_count': 0,
                'last_operation': None
            }
        
        # Update metrics
        op_metrics = self.metrics['operations'][operation_type]
        op_metrics['count'] += 1
        op_metrics['total_duration_ms'] += duration_ms
        op_metrics['last_operation'] = current_time
        
        if success:
            op_metrics['success_count'] += 1
        else:
            op_metrics['error_count'] += 1
        
        # Update conversation metrics
        if conversation_id not in self.metrics['conversations']:
            self.metrics['conversations'][conversation_id] = {
                'operations': 0,
                'last_activity': current_time,
                'created': current_time
            }
        
        conv_metrics = self.metrics['conversations'][conversation_id]
        conv_metrics['operations'] += 1
        conv_metrics['last_activity'] = current_time
        
        self.metrics['last_updated'] = current_time
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        current_time = time.time()
        
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_operations': sum(op['count'] for op in self.metrics['operations'].values()),
            'active_conversations': len(self.metrics['conversations']),
            'operations_by_type': {},
            'overall_success_rate': 0.0,
            'average_duration_ms': 0.0
        }
        
        # Calculate per-operation metrics
        total_operations = 0
        total_successes = 0
        total_duration = 0
        
        for op_type, op_data in self.metrics['operations'].items():
            count = op_data['count']
            if count > 0:
                avg_duration = op_data['total_duration_ms'] / count
                success_rate = op_data['success_count'] / count
                
                summary['operations_by_type'][op_type] = {
                    'count': count,
                    'success_rate': success_rate,
                    'average_duration_ms': avg_duration,
                    'success_count': op_data['success_count'],
                    'error_count': op_data['error_count']
                }
                
                total_operations += count
                total_successes += op_data['success_count']
                total_duration += op_data['total_duration_ms']
        
        # Calculate overall metrics
        if total_operations > 0:
            summary['overall_success_rate'] = total_successes / total_operations
            summary['average_duration_ms'] = total_duration / total_operations
        
        return summary


# Global instances
_state_tracer = None
_state_metrics_reporter = None


def get_state_tracer() -> StateOperationTracer:
    """Get singleton instance of StateOperationTracer"""
    global _state_tracer
    
    if _state_tracer is None:
        _state_tracer = StateOperationTracer()
    
    return _state_tracer


def get_state_metrics_reporter() -> StateMetricsReporter:
    """Get singleton instance of StateMetricsReporter"""
    global _state_metrics_reporter
    
    if _state_metrics_reporter is None:
        _state_metrics_reporter = StateMetricsReporter()
    
    return _state_metrics_reporter


# Convenience decorators
def trace_state_operation(operation_name: str, include_state_context: bool = True):
    """Decorator to automatically trace state operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract conversation_id and state context
            conversation_id = 'unknown'
            context = {}
            
            # Try to find conversation_id in arguments
            for arg in args:
                if isinstance(arg, dict) and 'conversation_id' in arg:
                    conversation_id = arg['conversation_id']
                    if include_state_context:
                        context.update({
                            'stage': arg.get('stage'),
                            'domain': arg.get('domain'),
                            'intent': arg.get('intent')
                        })
                    break
            
            # Check kwargs
            if 'conversation_id' in kwargs:
                conversation_id = kwargs['conversation_id']
            
            if 'state' in kwargs and include_state_context:
                state = kwargs['state']
                context.update({
                    'stage': state.get('stage'),
                    'domain': state.get('domain'),
                    'intent': state.get('intent')
                })
            
            tracer = get_state_tracer()
            
            async with tracer.trace_operation(
                operation_name=operation_name,
                conversation_id=conversation_id,
                **context
            ) as span:
                # Add function name to span
                tracer.add_span_annotation(span['span_id'], 'function_name', func.__name__)
                
                result = await func(*args, **kwargs)
                
                # Record metrics
                metrics_reporter = get_state_metrics_reporter()
                metrics_reporter.record_operation(
                    operation_type=operation_name,
                    conversation_id=conversation_id,
                    duration_ms=(time.time() - span['start_time']) * 1000,
                    success=True
                )
                
                return result
        
        return wrapper
    return decorator


# Monitoring initialization
def initialize_state_monitoring():
    """Initialize state monitoring components"""
    logger.info("Initializing simplified state monitoring and telemetry")
    
    # Initialize collectors
    get_state_tracer()
    get_state_metrics_reporter()
    
    logger.info("State monitoring initialized successfully")