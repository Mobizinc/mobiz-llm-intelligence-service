"""
Flow Control and Circuit Breakers for LangGraph Workflows

This module implements comprehensive flow control mechanisms including
circuit breakers, rate limiting, backpressure handling, automatic retries
with exponential backoff, and graceful degradation strategies to ensure
system stability under all conditions.

Features:
- Circuit breakers for each node with configurable thresholds
- Rate limiting for LLM calls and resource usage
- Backpressure handling for queue buildup
- Automatic retries with exponential backoff
- Graceful degradation strategies
- Flow control metrics and alerting
- Manual circuit breaker controls

Part of Epic 3: Orchestration & Intelligent Routing
Story 3.5: Create Flow Control and Circuit Breakers
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..nodes.base import BaseNode, NodeError, CircuitBreakerOpen
from ..models.conversation_state import (
    TechnicalConversationState,
    ConversationStage,
    ResponseType
)
from ..monitoring.state_metrics import StateMetricsCollector
from ..telemetry.state_telemetry import StateOperationTracer

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open" # Testing recovery


class RetryStrategy(str, Enum):
    """Retry strategies"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI = "fibonacci"


class DegradationLevel(str, Enum):
    """Graceful degradation levels"""
    NONE = "none"
    MINIMAL = "minimal"     # Reduce features slightly
    MODERATE = "moderate"   # Reduce features significantly
    SEVERE = "severe"       # Minimal functionality only
    EMERGENCY = "emergency" # Emergency mode


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 3        # Successes to close from half-open
    timeout_seconds: int = 60         # Time before half-open retry
    half_open_max_calls: int = 3      # Max calls in half-open state
    
    # Advanced configuration
    failure_rate_threshold: float = 0.5  # Failure rate to open (0.0-1.0)
    minimum_requests: int = 10            # Min requests before rate calculation
    sliding_window_size: int = 100        # Size of sliding window for rate calculation
    
    # Recovery configuration
    recovery_timeout_multiplier: float = 2.0  # Multiplier for progressive recovery
    max_recovery_timeout: int = 300           # Maximum recovery timeout


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    max_requests_per_second: float = 10.0
    max_requests_per_minute: float = 100.0
    max_requests_per_hour: float = 1000.0
    
    # Burst handling
    burst_size: int = 20
    burst_timeout: int = 60
    
    # Priority-based limits
    priority_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "high": 2.0,
        "medium": 1.0,
        "low": 0.5
    })


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Jitter configuration
    jitter: bool = True
    jitter_factor: float = 0.1
    
    # Retry conditions
    retryable_exceptions: List[str] = field(default_factory=lambda: [
        "NodeTimeoutError", "ConnectionError", "HTTPError", "TemporaryError"
    ])
    non_retryable_exceptions: List[str] = field(default_factory=lambda: [
        "NodeValidationError", "AuthenticationError", "PermissionError"
    ])


class CircuitBreaker:
    """
    Circuit breaker implementation with advanced features.
    
    Provides protection against cascading failures with:
    - Configurable failure thresholds
    - Sliding window failure rate calculation
    - Progressive recovery timeouts
    - Half-open state testing
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._recovery_timeout = config.timeout_seconds
        
        # Sliding window for failure rate calculation
        self._request_history: List[Tuple[float, bool]] = []  # (timestamp, success)
        self._half_open_calls = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"CircuitBreaker '{name}' initialized with config: {config}")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"CircuitBreaker '{self.name}' transitioning to HALF_OPEN")
                    return True
                return False
            else:  # HALF_OPEN
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
    
    def record_success(self) -> None:
        """Record successful execution"""
        
        with self._lock:
            current_time = time.time()
            self._request_history.append((current_time, True))
            self._cleanup_old_requests(current_time)
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._close_circuit()
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)
    
    def record_failure(self) -> None:
        """Record failed execution"""
        
        with self._lock:
            current_time = time.time()
            self._request_history.append((current_time, False))
            self._cleanup_old_requests(current_time)
            self._last_failure_time = current_time
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._open_circuit()
            else:
                self._failure_count += 1
                if self._should_open_circuit():
                    self._open_circuit()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open"""
        
        # Check simple failure count
        if self._failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate if we have enough requests
        if len(self._request_history) >= self.config.minimum_requests:
            failure_rate = self._calculate_failure_rate()
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN to HALF_OPEN"""
        
        if not self._last_failure_time:
            return True
        
        return (time.time() - self._last_failure_time) >= self._recovery_timeout
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate from sliding window"""
        
        if not self._request_history:
            return 0.0
        
        failures = sum(1 for _, success in self._request_history if not success)
        return failures / len(self._request_history)
    
    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove old requests from sliding window"""
        
        cutoff_time = current_time - (self.config.sliding_window_size * 60)  # Convert to seconds
        self._request_history = [
            (timestamp, success) for timestamp, success in self._request_history
            if timestamp > cutoff_time
        ]
    
    def _open_circuit(self) -> None:
        """Open the circuit"""
        
        self._state = CircuitBreakerState.OPEN
        self._half_open_calls = 0
        self._success_count = 0
        
        # Progressive recovery timeout
        self._recovery_timeout = min(
            self._recovery_timeout * self.config.recovery_timeout_multiplier,
            self.config.max_recovery_timeout
        )
        
        logger.warning(f"CircuitBreaker '{self.name}' OPENED - recovery timeout: {self._recovery_timeout}s")
    
    def _close_circuit(self) -> None:
        """Close the circuit"""
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._recovery_timeout = self.config.timeout_seconds  # Reset to initial timeout
        
        logger.info(f"CircuitBreaker '{self.name}' CLOSED")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "failure_rate": self._calculate_failure_rate(),
                "requests_in_window": len(self._request_history),
                "recovery_timeout": self._recovery_timeout,
                "last_failure_time": self._last_failure_time
            }
    
    def force_open(self) -> None:
        """Manually force circuit breaker open"""
        
        with self._lock:
            self._open_circuit()
            logger.warning(f"CircuitBreaker '{self.name}' manually forced OPEN")
    
    def force_close(self) -> None:
        """Manually force circuit breaker closed"""
        
        with self._lock:
            self._close_circuit()
            logger.info(f"CircuitBreaker '{self.name}' manually forced CLOSED")


class RateLimiter:
    """
    Rate limiter with token bucket algorithm and priority support.
    
    Provides:
    - Multiple time window rate limiting
    - Burst handling with token bucket
    - Priority-based rate adjustments
    - Backpressure detection
    """
    
    def __init__(self, name: str, config: RateLimitConfig):
        self.name = name
        self.config = config
        
        # Token buckets for different time windows
        self._buckets = {
            "second": {"tokens": config.max_requests_per_second, "last_refill": time.time()},
            "minute": {"tokens": config.max_requests_per_minute, "last_refill": time.time()},
            "hour": {"tokens": config.max_requests_per_hour, "last_refill": time.time()}
        }
        
        # Burst handling
        self._burst_bucket = {"tokens": config.burst_size, "last_refill": time.time()}
        
        # Request tracking
        self._request_history: List[Tuple[float, str]] = []  # (timestamp, priority)
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"RateLimiter '{name}' initialized with config: {config}")
    
    async def acquire(self, priority: str = "medium") -> bool:
        """Acquire rate limit token"""
        
        with self._lock:
            current_time = time.time()
            
            # Refill buckets
            self._refill_buckets(current_time)
            
            # Check if request can be accommodated
            priority_multiplier = self.config.priority_multipliers.get(priority, 1.0)
            required_tokens = 1.0 / priority_multiplier
            
            # Check all time windows
            for window, bucket in self._buckets.items():
                if bucket["tokens"] < required_tokens:
                    logger.debug(f"RateLimiter '{self.name}' - {window} bucket depleted")
                    return False
            
            # Check burst bucket
            if self._burst_bucket["tokens"] < required_tokens:
                logger.debug(f"RateLimiter '{self.name}' - burst bucket depleted")
                return False
            
            # Consume tokens
            for bucket in self._buckets.values():
                bucket["tokens"] -= required_tokens
            self._burst_bucket["tokens"] -= required_tokens
            
            # Record request
            self._request_history.append((current_time, priority))
            self._cleanup_old_requests(current_time)
            
            return True
    
    def _refill_buckets(self, current_time: float) -> None:
        """Refill token buckets based on time elapsed"""
        
        # Refill second bucket
        time_elapsed = current_time - self._buckets["second"]["last_refill"]
        if time_elapsed >= 1.0:
            self._buckets["second"]["tokens"] = min(
                self.config.max_requests_per_second,
                self._buckets["second"]["tokens"] + (time_elapsed * self.config.max_requests_per_second)
            )
            self._buckets["second"]["last_refill"] = current_time
        
        # Refill minute bucket
        time_elapsed = current_time - self._buckets["minute"]["last_refill"]
        if time_elapsed >= 60.0:
            self._buckets["minute"]["tokens"] = min(
                self.config.max_requests_per_minute,
                self._buckets["minute"]["tokens"] + (time_elapsed / 60.0 * self.config.max_requests_per_minute)
            )
            self._buckets["minute"]["last_refill"] = current_time
        
        # Refill hour bucket
        time_elapsed = current_time - self._buckets["hour"]["last_refill"]
        if time_elapsed >= 3600.0:
            self._buckets["hour"]["tokens"] = min(
                self.config.max_requests_per_hour,
                self._buckets["hour"]["tokens"] + (time_elapsed / 3600.0 * self.config.max_requests_per_hour)
            )
            self._buckets["hour"]["last_refill"] = current_time
        
        # Refill burst bucket
        time_elapsed = current_time - self._burst_bucket["last_refill"]
        if time_elapsed >= self.config.burst_timeout:
            self._burst_bucket["tokens"] = self.config.burst_size
            self._burst_bucket["last_refill"] = current_time
    
    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove old requests from tracking"""
        
        cutoff_time = current_time - 3600  # Keep 1 hour of history
        self._request_history = [
            (timestamp, priority) for timestamp, priority in self._request_history
            if timestamp > cutoff_time
        ]
    
    def get_state(self) -> Dict[str, Any]:
        """Get current rate limiter state"""
        
        with self._lock:
            current_time = time.time()
            self._refill_buckets(current_time)
            
            return {
                "name": self.name,
                "buckets": {
                    window: {
                        "available_tokens": bucket["tokens"],
                        "max_tokens": getattr(self.config, f"max_requests_per_{window}")
                    }
                    for window, bucket in self._buckets.items()
                },
                "burst_tokens": self._burst_bucket["tokens"],
                "recent_requests": len([
                    req for req in self._request_history
                    if current_time - req[0] < 60  # Last minute
                ])
            }
    
    def is_backpressure_detected(self) -> bool:
        """Check if backpressure is detected"""
        
        with self._lock:
            current_time = time.time()
            
            # Check if any bucket is severely depleted
            self._refill_buckets(current_time)
            
            for window, bucket in self._buckets.items():
                max_tokens = getattr(self.config, f"max_requests_per_{window}")
                if bucket["tokens"] < (max_tokens * 0.1):  # Less than 10% available
                    return True
            
            return False


class RetryHandler:
    """
    Advanced retry handler with multiple strategies and jitter.
    
    Provides:
    - Multiple retry strategies (exponential, linear, fibonacci)
    - Configurable jitter to prevent thundering herd
    - Exception-based retry decisions
    - Progressive backoff with max limits
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        
    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry and backoff"""
        
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                
                # Success - log if this was a retry
                if attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e):
                    logger.info(f"Non-retryable exception: {type(e).__name__}")
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}, retrying in {delay:.2f}s",
                    extra={
                        "attempt": attempt + 1,
                        "max_attempts": self.config.max_attempts,
                        "delay": delay,
                        "exception_type": type(e).__name__
                    }
                )
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        logger.error(f"All {self.config.max_attempts} retry attempts exhausted")
        raise last_exception
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Determine if exception is retryable"""
        
        exception_name = type(exception).__name__
        
        # Check non-retryable exceptions first
        if exception_name in self.config.non_retryable_exceptions:
            return False
        
        # Check retryable exceptions
        if exception_name in self.config.retryable_exceptions:
            return True
        
        # Default behavior based on exception type
        return not isinstance(exception, (ValueError, TypeError))
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay * (self.config.multiplier ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * (attempt + 1)
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.initial_delay * self._fibonacci(attempt + 1)
        else:  # FIXED_DELAY
            delay = self.config.initial_delay
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.1, delay)  # Minimum delay of 100ms
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)


class FlowController:
    """
    Comprehensive flow controller with circuit breakers, rate limiting,
    retries, and graceful degradation.
    
    This is the main entry point for flow control in the orchestration layer,
    providing protection against cascading failures and system overload.
    """
    
    def __init__(
        self,
        default_circuit_config: Optional[CircuitBreakerConfig] = None,
        default_rate_config: Optional[RateLimitConfig] = None,
        default_retry_config: Optional[RetryConfig] = None
    ):
        self.default_circuit_config = default_circuit_config or CircuitBreakerConfig()
        self.default_rate_config = default_rate_config or RateLimitConfig()
        self.default_retry_config = default_retry_config or RetryConfig()
        
        # Component registries
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        
        # Global degradation level
        self.degradation_level = DegradationLevel.NONE
        self._degradation_history: List[Tuple[datetime, DegradationLevel]] = []
        
        # Monitoring
        self.metrics_collector = StateMetricsCollector()
        self.telemetry_service = StateOperationTracer()
        
        logger.info("FlowController initialized with default configurations")
    
    def register_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Register a new circuit breaker"""
        
        breaker_config = config or self.default_circuit_config
        circuit_breaker = CircuitBreaker(name, breaker_config)
        self.circuit_breakers[name] = circuit_breaker
        
        logger.info(f"Registered circuit breaker: {name}")
        return circuit_breaker
    
    def register_rate_limiter(
        self,
        name: str,
        config: Optional[RateLimitConfig] = None
    ) -> RateLimiter:
        """Register a new rate limiter"""
        
        limiter_config = config or self.default_rate_config
        rate_limiter = RateLimiter(name, limiter_config)
        self.rate_limiters[name] = rate_limiter
        
        logger.info(f"Registered rate limiter: {name}")
        return rate_limiter
    
    def register_retry_handler(
        self,
        name: str,
        config: Optional[RetryConfig] = None
    ) -> RetryHandler:
        """Register a new retry handler"""
        
        retry_config = config or self.default_retry_config
        retry_handler = RetryHandler(retry_config)
        self.retry_handlers[name] = retry_handler
        
        logger.info(f"Registered retry handler: {name}")
        return retry_handler
    
    async def execute_with_protection(
        self,
        func: Callable,
        node_name: str,
        *args,
        priority: str = "medium",
        **kwargs
    ) -> Any:
        """Execute function with full flow control protection"""
        
        start_time = time.time()
        
        try:
            # Get or create circuit breaker
            circuit_breaker = self.circuit_breakers.get(node_name)
            if not circuit_breaker:
                circuit_breaker = self.register_circuit_breaker(node_name)
            
            # Check circuit breaker
            if not circuit_breaker.can_execute():
                error_msg = f"Circuit breaker '{node_name}' is OPEN"
                logger.warning(error_msg)
                raise CircuitBreakerOpen(error_msg)
            
            # Get or create rate limiter
            rate_limiter = self.rate_limiters.get(node_name)
            if not rate_limiter:
                rate_limiter = self.register_rate_limiter(node_name)
            
            # Check rate limits
            if not await rate_limiter.acquire(priority):
                error_msg = f"Rate limit exceeded for '{node_name}'"
                logger.warning(error_msg)
                raise NodeError(error_msg)
            
            # Get or create retry handler
            retry_handler = self.retry_handlers.get(node_name)
            if not retry_handler:
                retry_handler = self.register_retry_handler(node_name)
            
            # Check degradation level and adjust behavior
            adjusted_func = self._apply_degradation(func, node_name)
            
            # Execute with retry protection
            result = await retry_handler.retry_with_backoff(
                adjusted_func, *args, **kwargs
            )
            
            # Record success
            circuit_breaker.record_success()
            
            execution_time = time.time() - start_time
            await self._record_success_metrics(node_name, execution_time, priority)
            
            return result
            
        except Exception as e:
            # Record failure
            if node_name in self.circuit_breakers:
                self.circuit_breakers[node_name].record_failure()
            
            execution_time = time.time() - start_time
            await self._record_failure_metrics(node_name, execution_time, type(e).__name__)
            
            # Check if we should increase degradation level
            await self._check_and_adjust_degradation()
            
            raise
    
    def _apply_degradation(self, func: Callable, node_name: str) -> Callable:
        """Apply degradation adjustments to function"""
        
        if self.degradation_level == DegradationLevel.NONE:
            return func
        
        # Create degraded wrapper
        async def degraded_func(*args, **kwargs):
            if self.degradation_level == DegradationLevel.MINIMAL:
                # Reduce timeout slightly
                if 'timeout' in kwargs:
                    kwargs['timeout'] = kwargs['timeout'] * 0.8
            
            elif self.degradation_level == DegradationLevel.MODERATE:
                # Reduce timeout more and simplify processing
                if 'timeout' in kwargs:
                    kwargs['timeout'] = kwargs['timeout'] * 0.6
                # Could add logic to skip non-essential processing
            
            elif self.degradation_level == DegradationLevel.SEVERE:
                # Minimal functionality only
                if 'timeout' in kwargs:
                    kwargs['timeout'] = min(kwargs['timeout'] * 0.4, 10)
                # Return simplified response for complex nodes
                if 'complex_processing' in kwargs:
                    kwargs['complex_processing'] = False
            
            elif self.degradation_level == DegradationLevel.EMERGENCY:
                # Emergency mode - return fallback response
                logger.warning(f"Emergency mode activated for {node_name}")
                return self._emergency_response(node_name, *args, **kwargs)
            
            return await func(*args, **kwargs)
        
        return degraded_func
    
    def _emergency_response(self, node_name: str, *args, **kwargs) -> TechnicalConversationState:
        """Generate emergency fallback response"""
        
        # Create minimal emergency state
        emergency_state = {
            'stage': ConversationStage.ERROR,
            'response': f"System is currently operating in emergency mode. Please try again later or contact support.",
            'response_type': ResponseType.ERROR,
            'emergency_mode': True,
            'degradation_level': self.degradation_level.value
        }
        
        # If input state provided, merge with it
        if args and isinstance(args[0], dict):
            input_state = args[0]
            emergency_state.update({
                'conversation_id': input_state.get('conversation_id'),
                'user_id': input_state.get('user_id'),
                'channel_id': input_state.get('channel_id'),
                'correlation_id': input_state.get('correlation_id')
            })
        
        return emergency_state
    
    async def _check_and_adjust_degradation(self) -> None:
        """Check system health and adjust degradation level"""
        
        try:
            # Count open circuit breakers
            open_breakers = sum(
                1 for cb in self.circuit_breakers.values()
                if cb._state == CircuitBreakerState.OPEN
            )
            
            # Count rate limiters under pressure
            pressured_limiters = sum(
                1 for rl in self.rate_limiters.values()
                if rl.is_backpressure_detected()
            )
            
            total_components = len(self.circuit_breakers) + len(self.rate_limiters)
            if total_components == 0:
                return
            
            # Calculate health scores
            health_score = 1.0 - ((open_breakers + pressured_limiters) / total_components)
            
            # Determine appropriate degradation level
            new_level = DegradationLevel.NONE
            
            if health_score < 0.2:
                new_level = DegradationLevel.EMERGENCY
            elif health_score < 0.4:
                new_level = DegradationLevel.SEVERE
            elif health_score < 0.6:
                new_level = DegradationLevel.MODERATE
            elif health_score < 0.8:
                new_level = DegradationLevel.MINIMAL
            
            # Update degradation level if changed
            if new_level != self.degradation_level:
                old_level = self.degradation_level
                self.degradation_level = new_level
                
                # Record change
                self._degradation_history.append((datetime.now(), new_level))
                
                logger.warning(
                    f"Degradation level changed: {old_level.value} -> {new_level.value}",
                    extra={
                        "health_score": health_score,
                        "open_breakers": open_breakers,
                        "pressured_limiters": pressured_limiters
                    }
                )
                
                # Record metric
                self.metrics_collector.set_gauge(
                    "system_degradation_level",
                    list(DegradationLevel).index(new_level)
                )
                
        except Exception as e:
            logger.error(f"Failed to check degradation level: {e}")
    
    async def _record_success_metrics(
        self,
        node_name: str,
        execution_time: float,
        priority: str
    ) -> None:
        """Record success metrics"""
        
        try:
            self.metrics_collector.record_histogram(
                f"flow_control_execution_time_seconds",
                execution_time,
                node=node_name,
                priority=priority,
                status="success"
            )
            
            self.metrics_collector.increment_counter(
                f"flow_control_executions_total",
                node=node_name,
                priority=priority,
                status="success"
            )
            
        except Exception as e:
            logger.warning(f"Failed to record success metrics: {e}")
    
    async def _record_failure_metrics(
        self,
        node_name: str,
        execution_time: float,
        exception_type: str
    ) -> None:
        """Record failure metrics"""
        
        try:
            self.metrics_collector.record_histogram(
                f"flow_control_execution_time_seconds",
                execution_time,
                node=node_name,
                status="failure",
                exception_type=exception_type
            )
            
            self.metrics_collector.increment_counter(
                f"flow_control_executions_total",
                node=node_name,
                status="failure",
                exception_type=exception_type
            )
            
        except Exception as e:
            logger.warning(f"Failed to record failure metrics: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        circuit_breaker_states = {
            name: cb.get_state() for name, cb in self.circuit_breakers.items()
        }
        
        rate_limiter_states = {
            name: rl.get_state() for name, rl in self.rate_limiters.items()
        }
        
        # Calculate overall health
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(
            1 for cb in self.circuit_breakers.values()
            if cb._state == CircuitBreakerState.OPEN
        )
        
        health_score = 1.0 - (open_breakers / max(total_breakers, 1))
        
        return {
            "health_score": health_score,
            "degradation_level": self.degradation_level.value,
            "circuit_breakers": circuit_breaker_states,
            "rate_limiters": rate_limiter_states,
            "total_components": len(self.circuit_breakers) + len(self.rate_limiters),
            "degradation_history": [
                {"timestamp": dt.isoformat(), "level": level.value}
                for dt, level in self._degradation_history[-10:]  # Last 10 changes
            ]
        }
    
    def force_degradation_level(self, level: DegradationLevel) -> None:
        """Manually force degradation level"""
        
        old_level = self.degradation_level
        self.degradation_level = level
        
        self._degradation_history.append((datetime.now(), level))
        
        logger.warning(f"Degradation level manually forced: {old_level.value} -> {level.value}")
    
    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state"""
        
        for cb in self.circuit_breakers.values():
            cb.force_close()
        
        logger.info("All circuit breakers reset to CLOSED state")
    
    def shutdown(self) -> None:
        """Shutdown flow controller"""
        
        logger.info("FlowController shutting down")
        
        # Clear all registries
        self.circuit_breakers.clear()
        self.rate_limiters.clear()
        self.retry_handlers.clear()
        
        # Reset degradation
        self.degradation_level = DegradationLevel.NONE