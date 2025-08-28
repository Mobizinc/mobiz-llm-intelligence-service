"""
State-specific Metrics for LangGraph Conversation State

Comprehensive metrics collection and monitoring for conversation state operations,
performance tracking, and system health indicators.

Part of Epic 1: Core State Management & Infrastructure
Story 1.5: State Monitoring & Observability
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Counter
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from threading import Lock
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class StateOperationMetric:
    """Metric for a single state operation"""
    operation_type: str
    conversation_id: str
    timestamp: float
    duration_ms: float
    success: bool
    stage: Optional[str] = None
    domain: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StateTransitionMetric:
    """Metric for state transitions"""
    conversation_id: str
    from_stage: str
    to_stage: str
    timestamp: float
    duration_ms: float
    success: bool
    trigger: str  # What triggered the transition
    confidence_change: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StateHealthMetric:
    """Health metric for conversation state"""
    timestamp: float
    total_conversations: int
    active_conversations: int
    conversations_by_stage: Dict[str, int]
    conversations_by_domain: Dict[str, int]
    average_conversation_age_minutes: float
    corruption_rate: float
    recovery_rate: float


class StateMetricsCollector:
    """
    Collects and aggregates metrics for conversation state operations.
    
    Provides real-time metrics collection, aggregation, and reporting
    for all state-related operations and system health indicators.
    """
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self._lock = Lock()
        
        # Operation metrics
        self.operation_history: deque = deque(maxlen=max_history_size)
        self.transition_history: deque = deque(maxlen=max_history_size)
        
        # Performance counters
        self.operation_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        self.transition_counters = defaultdict(int)
        
        # Performance tracking
        self.operation_timings = defaultdict(list)
        self.stage_durations = defaultdict(list)
        
        # Health tracking
        self.health_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Real-time state
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_start_times: Dict[str, float] = {}
        
        logger.info("StateMetricsCollector initialized")
    
    def record_operation(
        self,
        operation_type: str,
        conversation_id: str,
        duration_ms: float,
        success: bool,
        stage: Optional[str] = None,
        domain: Optional[str] = None,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a state operation metric"""
        with self._lock:
            metric = StateOperationMetric(
                operation_type=operation_type,
                conversation_id=conversation_id,
                timestamp=time.time(),
                duration_ms=duration_ms,
                success=success,
                stage=stage,
                domain=domain,
                error_type=error_type,
                metadata=metadata or {}
            )
            
            self.operation_history.append(metric)
            
            # Update counters
            self.operation_counters[operation_type] += 1
            if not success:
                self.error_counters[f"{operation_type}_error"] += 1
                if error_type:
                    self.error_counters[error_type] += 1
            
            # Update timing tracking
            self.operation_timings[operation_type].append(duration_ms)
            # Keep only recent timings
            if len(self.operation_timings[operation_type]) > 1000:
                self.operation_timings[operation_type] = self.operation_timings[operation_type][-500:]
            
            # Track conversation start
            if operation_type == "state_create" and success:
                self.conversation_start_times[conversation_id] = time.time()
                self.active_conversations[conversation_id] = {
                    'stage': stage,
                    'domain': domain,
                    'started_at': time.time(),
                    'last_activity': time.time()
                }
            
            # Update conversation activity
            if conversation_id in self.active_conversations:
                self.active_conversations[conversation_id]['last_activity'] = time.time()
                if stage:
                    self.active_conversations[conversation_id]['stage'] = stage
    
    def record_transition(
        self,
        conversation_id: str,
        from_stage: str,
        to_stage: str,
        duration_ms: float,
        success: bool,
        trigger: str,
        confidence_change: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a state transition metric"""
        with self._lock:
            metric = StateTransitionMetric(
                conversation_id=conversation_id,
                from_stage=from_stage,
                to_stage=to_stage,
                timestamp=time.time(),
                duration_ms=duration_ms,
                success=success,
                trigger=trigger,
                confidence_change=confidence_change,
                metadata=metadata or {}
            )
            
            self.transition_history.append(metric)
            
            # Update counters
            transition_key = f"{from_stage}_to_{to_stage}"
            self.transition_counters[transition_key] += 1
            
            if not success:
                self.error_counters[f"transition_{transition_key}_error"] += 1
            
            # Track stage durations
            if conversation_id in self.conversation_start_times:
                if from_stage != to_stage:  # Only for actual transitions
                    stage_duration = time.time() - self.conversation_start_times[conversation_id]
                    self.stage_durations[from_stage].append(stage_duration * 1000)  # Convert to ms
                    
                    # Keep only recent durations
                    if len(self.stage_durations[from_stage]) > 1000:
                        self.stage_durations[from_stage] = self.stage_durations[from_stage][-500:]
            
            # Update active conversation
            if conversation_id in self.active_conversations:
                self.active_conversations[conversation_id]['stage'] = to_stage
                self.active_conversations[conversation_id]['last_activity'] = time.time()
    
    def record_conversation_end(self, conversation_id: str, final_stage: str):
        """Record conversation completion"""
        with self._lock:
            if conversation_id in self.active_conversations:
                conversation_data = self.active_conversations.pop(conversation_id)
                
                # Calculate total conversation duration
                if conversation_id in self.conversation_start_times:
                    total_duration = time.time() - self.conversation_start_times[conversation_id]
                    self.conversation_start_times.pop(conversation_id)
                    
                    # Record completion metric
                    self.record_operation(
                        operation_type="conversation_completed",
                        conversation_id=conversation_id,
                        duration_ms=total_duration * 1000,
                        success=True,
                        stage=final_stage,
                        domain=conversation_data.get('domain'),
                        metadata={
                            'total_duration_ms': total_duration * 1000,
                            'final_stage': final_stage
                        }
                    )
    
    def cleanup_stale_conversations(self, max_age_hours: int = 24):
        """Clean up stale conversation tracking"""
        with self._lock:
            current_time = time.time()
            stale_threshold = current_time - (max_age_hours * 3600)
            
            stale_conversations = [
                conv_id for conv_id, data in self.active_conversations.items()
                if data['last_activity'] < stale_threshold
            ]
            
            for conv_id in stale_conversations:
                self.active_conversations.pop(conv_id, None)
                self.conversation_start_times.pop(conv_id, None)
            
            if stale_conversations:
                logger.info(f"Cleaned up {len(stale_conversations)} stale conversations")
    
    def get_performance_metrics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics for specified time window"""
        with self._lock:
            current_time = time.time()
            window_start = current_time - (time_window_minutes * 60)
            
            # Filter recent operations
            recent_operations = [
                op for op in self.operation_history
                if op.timestamp >= window_start
            ]
            
            recent_transitions = [
                tr for tr in self.transition_history
                if tr.timestamp >= window_start
            ]
            
            # Calculate performance metrics
            total_operations = len(recent_operations)
            successful_operations = sum(1 for op in recent_operations if op.success)
            
            # Average durations by operation type
            avg_durations = {}
            for op_type in set(op.operation_type for op in recent_operations):
                durations = [op.duration_ms for op in recent_operations 
                           if op.operation_type == op_type and op.success]
                if durations:
                    avg_durations[op_type] = {
                        'avg_ms': sum(durations) / len(durations),
                        'min_ms': min(durations),
                        'max_ms': max(durations),
                        'count': len(durations)
                    }
            
            # Transition performance
            transition_success_rate = 0.0
            if recent_transitions:
                successful_transitions = sum(1 for tr in recent_transitions if tr.success)
                transition_success_rate = successful_transitions / len(recent_transitions)
            
            return {
                'time_window_minutes': time_window_minutes,
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
                'operation_durations': avg_durations,
                'total_transitions': len(recent_transitions),
                'transition_success_rate': transition_success_rate,
                'active_conversations': len(self.active_conversations),
                'operations_per_minute': total_operations / time_window_minutes if time_window_minutes > 0 else 0
            }
    
    def get_conversation_distribution(self) -> Dict[str, Any]:
        """Get current conversation distribution metrics"""
        with self._lock:
            stage_distribution = defaultdict(int)
            domain_distribution = defaultdict(int)
            age_distribution = defaultdict(int)
            
            current_time = time.time()
            
            for conv_id, data in self.active_conversations.items():
                stage = data.get('stage', 'unknown')
                domain = data.get('domain', 'unknown')
                age_minutes = (current_time - data.get('started_at', current_time)) / 60
                
                stage_distribution[stage] += 1
                domain_distribution[domain] += 1
                
                # Age buckets
                if age_minutes < 5:
                    age_distribution['0-5min'] += 1
                elif age_minutes < 15:
                    age_distribution['5-15min'] += 1
                elif age_minutes < 60:
                    age_distribution['15-60min'] += 1
                else:
                    age_distribution['60min+'] += 1
            
            return {
                'total_conversations': len(self.active_conversations),
                'by_stage': dict(stage_distribution),
                'by_domain': dict(domain_distribution),
                'by_age': dict(age_distribution)
            }
    
    def get_error_metrics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get error metrics for specified time window"""
        with self._lock:
            current_time = time.time()
            window_start = current_time - (time_window_minutes * 60)
            
            # Filter recent operations with errors
            recent_errors = [
                op for op in self.operation_history
                if op.timestamp >= window_start and not op.success
            ]
            
            # Group errors by type
            error_by_type = defaultdict(int)
            error_by_operation = defaultdict(int)
            
            for error_op in recent_errors:
                if error_op.error_type:
                    error_by_type[error_op.error_type] += 1
                error_by_operation[error_op.operation_type] += 1
            
            total_operations = len([
                op for op in self.operation_history
                if op.timestamp >= window_start
            ])
            
            return {
                'time_window_minutes': time_window_minutes,
                'total_errors': len(recent_errors),
                'error_rate': len(recent_errors) / total_operations if total_operations > 0 else 0,
                'errors_by_type': dict(error_by_type),
                'errors_by_operation': dict(error_by_operation),
                'errors_per_minute': len(recent_errors) / time_window_minutes if time_window_minutes > 0 else 0
            }
    
    def get_transition_metrics(self) -> Dict[str, Any]:
        """Get state transition metrics"""
        with self._lock:
            # Most common transitions
            transition_counts = defaultdict(int)
            transition_durations = defaultdict(list)
            
            for transition in self.transition_history:
                key = f"{transition.from_stage}_to_{transition.to_stage}"
                transition_counts[key] += 1
                transition_durations[key].append(transition.duration_ms)
            
            # Calculate averages
            avg_transition_durations = {}
            for key, durations in transition_durations.items():
                avg_transition_durations[key] = {
                    'avg_ms': sum(durations) / len(durations),
                    'count': len(durations)
                }
            
            # Common transition paths
            top_transitions = sorted(
                transition_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                'total_transitions': len(self.transition_history),
                'unique_transition_types': len(transition_counts),
                'top_transitions': top_transitions,
                'avg_durations': avg_transition_durations
            }
    
    def record_health_snapshot(
        self,
        total_conversations: int,
        corruption_rate: float,
        recovery_rate: float
    ):
        """Record a health snapshot"""
        with self._lock:
            conversation_dist = self.get_conversation_distribution()
            
            # Calculate average conversation age
            current_time = time.time()
            ages = [
                (current_time - data.get('started_at', current_time)) / 60
                for data in self.active_conversations.values()
            ]
            avg_age = sum(ages) / len(ages) if ages else 0
            
            health_metric = StateHealthMetric(
                timestamp=current_time,
                total_conversations=total_conversations,
                active_conversations=len(self.active_conversations),
                conversations_by_stage=conversation_dist['by_stage'],
                conversations_by_domain=conversation_dist['by_domain'],
                average_conversation_age_minutes=avg_age,
                corruption_rate=corruption_rate,
                recovery_rate=recovery_rate
            )
            
            self.health_history.append(health_metric)
    
    def get_health_trend(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trend over specified hours"""
        with self._lock:
            current_time = time.time()
            window_start = current_time - (hours * 3600)
            
            recent_health = [
                h for h in self.health_history
                if h.timestamp >= window_start
            ]
            
            if not recent_health:
                return {'status': 'no_data', 'hours': hours}
            
            # Calculate trends
            latest = recent_health[-1]
            oldest = recent_health[0]
            
            conversation_trend = latest.active_conversations - oldest.active_conversations
            corruption_trend = latest.corruption_rate - oldest.corruption_rate
            recovery_trend = latest.recovery_rate - oldest.recovery_rate
            
            return {
                'hours': hours,
                'current_health': asdict(latest),
                'trends': {
                    'conversation_count': conversation_trend,
                    'corruption_rate': corruption_trend,
                    'recovery_rate': recovery_trend
                },
                'snapshots_count': len(recent_health)
            }
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring systems"""
        with self._lock:
            return {
                'timestamp': time.time(),
                'performance': self.get_performance_metrics(),
                'conversations': self.get_conversation_distribution(),
                'errors': self.get_error_metrics(),
                'transitions': self.get_transition_metrics(),
                'health_trend': self.get_health_trend(),
                'counters': {
                    'operations': dict(self.operation_counters),
                    'errors': dict(self.error_counters),
                    'transitions': dict(self.transition_counters)
                }
            }
    
    def reset_metrics(self):
        """Reset all metrics (for testing or periodic reset)"""
        with self._lock:
            self.operation_history.clear()
            self.transition_history.clear()
            self.health_history.clear()
            
            self.operation_counters.clear()
            self.error_counters.clear()
            self.transition_counters.clear()
            
            self.operation_timings.clear()
            self.stage_durations.clear()
            
            self.active_conversations.clear()
            self.conversation_start_times.clear()
            
            logger.info("All state metrics reset")

    # Legacy compatibility methods
    def increment_counter(self, metric_name: str, **labels):
        """Increment a counter metric"""
        logger.debug(f"Incremented {metric_name} with labels {labels}")
    
    def record_histogram(self, metric_name: str, value: float, **labels):
        """Record a histogram metric"""
        logger.debug(f"Recorded {metric_name}={value} with labels {labels}")
    
    def record_gauge(self, metric_name: str, value: float, **labels):
        """Record a gauge metric"""
        logger.debug(f"Set gauge {metric_name}={value} with labels {labels}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return self.export_metrics()
    
    def reset(self):
        """Reset all metrics"""
        self.reset_metrics()


# Global metrics collector instance
_state_metrics_collector = None

def get_state_metrics_collector() -> StateMetricsCollector:
    """Get singleton instance of StateMetricsCollector"""
    global _state_metrics_collector
    
    if _state_metrics_collector is None:
        _state_metrics_collector = StateMetricsCollector()
    
    return _state_metrics_collector


# Convenience functions for common metrics
def record_state_operation(operation_type: str, conversation_id: str, duration_ms: float, 
                          success: bool, **kwargs):
    """Convenience function to record state operation"""
    collector = get_state_metrics_collector()
    collector.record_operation(operation_type, conversation_id, duration_ms, success, **kwargs)


def record_state_transition(conversation_id: str, from_stage: str, to_stage: str,
                           duration_ms: float, success: bool, trigger: str, **kwargs):
    """Convenience function to record state transition"""
    collector = get_state_metrics_collector()
    collector.record_transition(conversation_id, from_stage, to_stage, duration_ms, 
                               success, trigger, **kwargs)


def record_conversation_completion(conversation_id: str, final_stage: str):
    """Convenience function to record conversation completion"""
    collector = get_state_metrics_collector()
    collector.record_conversation_end(conversation_id, final_stage)