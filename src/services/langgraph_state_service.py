"""
LangGraph State Service

Dedicated service for managing LangGraph conversation state with high-performance
operations, caching, and comprehensive error handling.

Part of Epic 1: Core State Management & Infrastructure
Story 1.2: Azure Table Storage Integration
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor

from ..shared.conversation_manager import ConversationManager
from ..models.conversation_state import (
    TechnicalConversationState,
    ConversationStateManager,
    ConversationStage
)
from ..core.singletons import get_security_manager
from ..shared.telemetry import EnhancedTelemetryManager

logger = logging.getLogger(__name__)


class LangGraphStateService:
    """
    High-performance service for LangGraph conversation state management.
    
    Features:
    - Async/await operations for non-blocking performance
    - In-memory caching for frequently accessed states
    - Automatic state versioning and migration
    - Performance monitoring and metrics
    - Batch operations for efficiency
    - Circuit breaker pattern for resilience
    """
    
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.telemetry_manager = EnhancedTelemetryManager()
        
        # Performance optimization: in-memory cache
        self._state_cache: Dict[str, TechnicalConversationState] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl_seconds = 300  # 5 minutes cache TTL
        self._max_cache_size = 1000
        
        # Performance monitoring
        self._performance_metrics = {
            'save_operations': 0,
            'load_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_save_time': 0.0,
            'total_load_time': 0.0
        }
        
        # Circuit breaker for resilience
        self._failure_count = 0
        self._last_failure_time = None
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 30  # seconds
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="langgraph-state")
        
        logger.info("LangGraphStateService initialized with caching and performance monitoring")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open (too many recent failures)"""
        if self._failure_count < self._circuit_breaker_threshold:
            return False
        
        if self._last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self._last_failure_time
        return time_since_failure < self._circuit_breaker_timeout
    
    def _record_success(self):
        """Record successful operation for circuit breaker"""
        self._failure_count = 0
        self._last_failure_time = None
    
    def _record_failure(self):
        """Record failed operation for circuit breaker"""
        self._failure_count += 1
        self._last_failure_time = time.time()
    
    def _get_cache_key(self, conversation_id: str, checkpoint_name: Optional[str] = None) -> str:
        """Generate cache key for state"""
        if checkpoint_name:
            return f"{conversation_id}:{checkpoint_name}"
        return f"{conversation_id}:latest"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached state is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[cache_key]
        return (time.time() - cache_time) < self._cache_ttl_seconds
    
    def _cache_state(self, cache_key: str, state: TechnicalConversationState):
        """Cache state with automatic eviction"""
        # Evict oldest entries if cache is full
        if len(self._state_cache) >= self._max_cache_size:
            # Remove 10% of oldest entries
            sorted_keys = sorted(self._cache_timestamps.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:int(self._max_cache_size * 0.1)]]
            
            for key in keys_to_remove:
                self._state_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        
        # Cache the state
        self._state_cache[cache_key] = state
        self._cache_timestamps[cache_key] = time.time()
    
    def _get_cached_state(self, cache_key: str) -> Optional[TechnicalConversationState]:
        """Get state from cache if valid"""
        if not self._is_cache_valid(cache_key):
            # Remove invalid cache entry
            self._state_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            return None
        
        return self._state_cache.get(cache_key)
    
    async def save_state(
        self,
        conversation_id: str,
        state: TechnicalConversationState,
        checkpoint_name: Optional[str] = None,
        force_write: bool = False
    ) -> bool:
        """
        Save conversation state with caching and performance monitoring.
        
        Args:
            conversation_id: Unique conversation identifier
            state: Complete conversation state
            checkpoint_name: Optional checkpoint name
            force_write: Skip cache and force write to storage
            
        Returns:
            True if saved successfully, False otherwise
        """
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker open, skipping save operation")
            return False
        
        start_time = time.time()
        
        try:
            # Validate state before saving
            ConversationStateManager.validate_state(state)
            
            # Update cache regardless of storage success
            cache_key = self._get_cache_key(conversation_id, checkpoint_name)
            self._cache_state(cache_key, state)
            
            # Save to storage
            success = await self.conversation_manager.save_langgraph_state(
                conversation_id, state, checkpoint_name
            )
            
            if success:
                self._record_success()
                self._performance_metrics['save_operations'] += 1
                
                # Track performance
                duration = time.time() - start_time
                self._performance_metrics['total_save_time'] += duration
                
                self.telemetry_manager.log_custom_event(
                    "langgraph.service.save_success",
                    {
                        "conversation_id": conversation_id,
                        "checkpoint_name": checkpoint_name,
                        "duration_ms": duration * 1000,
                        "state_version": state['version'],
                        "cache_size": len(self._state_cache)
                    }
                )
                
                logger.debug(f"State saved successfully: {conversation_id}, duration: {duration:.3f}s")
            else:
                self._record_failure()
                
            return success
            
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to save state: {e}")
            
            self.telemetry_manager.log_custom_event(
                "langgraph.service.save_failed",
                {
                    "conversation_id": conversation_id,
                    "checkpoint_name": checkpoint_name,
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                }
            )
            
            return False
    
    async def load_state(
        self,
        conversation_id: str,
        checkpoint_name: Optional[str] = None
    ) -> Optional[TechnicalConversationState]:
        """
        Load conversation state with caching and performance monitoring.
        
        Args:
            conversation_id: Unique conversation identifier
            checkpoint_name: Optional checkpoint name
            
        Returns:
            TechnicalConversationState if found, None otherwise
        """
        if self._is_circuit_breaker_open():
            logger.warning("Circuit breaker open, skipping load operation")
            return None
        
        start_time = time.time()
        cache_key = self._get_cache_key(conversation_id, checkpoint_name)
        
        try:
            # Check cache first
            cached_state = self._get_cached_state(cache_key)
            if cached_state:
                self._performance_metrics['cache_hits'] += 1
                
                self.telemetry_manager.log_custom_event(
                    "langgraph.service.cache_hit",
                    {
                        "conversation_id": conversation_id,
                        "checkpoint_name": checkpoint_name,
                        "cache_age_seconds": time.time() - self._cache_timestamps[cache_key]
                    }
                )
                
                logger.debug(f"Cache hit for state: {conversation_id}")
                return cached_state
            
            # Cache miss - load from storage
            self._performance_metrics['cache_misses'] += 1
            self._performance_metrics['load_operations'] += 1
            
            state = await self.conversation_manager.load_langgraph_state(
                conversation_id, checkpoint_name
            )
            
            if state:
                # Cache the loaded state
                self._cache_state(cache_key, state)
                self._record_success()
                
                # Track performance
                duration = time.time() - start_time
                self._performance_metrics['total_load_time'] += duration
                
                self.telemetry_manager.log_custom_event(
                    "langgraph.service.load_success",
                    {
                        "conversation_id": conversation_id,
                        "checkpoint_name": checkpoint_name,
                        "duration_ms": duration * 1000,
                        "state_version": state['version'],
                        "cache_miss": True
                    }
                )
                
                logger.debug(f"State loaded from storage: {conversation_id}, duration: {duration:.3f}s")
            else:
                self.telemetry_manager.log_custom_event(
                    "langgraph.service.state_not_found",
                    {
                        "conversation_id": conversation_id,
                        "checkpoint_name": checkpoint_name
                    }
                )
            
            return state
            
        except Exception as e:
            self._record_failure()
            logger.error(f"Failed to load state: {e}")
            
            self.telemetry_manager.log_custom_event(
                "langgraph.service.load_failed",
                {
                    "conversation_id": conversation_id,
                    "checkpoint_name": checkpoint_name,
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                }
            )
            
            return None
    
    async def update_state(
        self,
        conversation_id: str,
        updates: Dict[str, Any],
        checkpoint_name: Optional[str] = None
    ) -> Optional[TechnicalConversationState]:
        """
        Update existing conversation state efficiently.
        
        Args:
            conversation_id: Unique conversation identifier
            updates: Updates to apply to state
            checkpoint_name: Optional checkpoint name
            
        Returns:
            Updated state if successful, None otherwise
        """
        try:
            # Load current state
            current_state = await self.load_state(conversation_id, checkpoint_name)
            
            if not current_state:
                logger.warning(f"Cannot update non-existent state: {conversation_id}")
                return None
            
            # Apply updates
            updated_state = ConversationStateManager.update_state(current_state, updates)
            
            # Save updated state
            success = await self.save_state(conversation_id, updated_state, checkpoint_name)
            
            if success:
                return updated_state
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to update state: {e}")
            return None
    
    async def create_checkpoint(
        self,
        conversation_id: str,
        checkpoint_name: str
    ) -> bool:
        """
        Create a checkpoint of the current conversation state.
        
        Args:
            conversation_id: Unique conversation identifier
            checkpoint_name: Name for the checkpoint
            
        Returns:
            True if checkpoint created successfully, False otherwise
        """
        try:
            # Load current state
            current_state = await self.load_state(conversation_id)
            
            if not current_state:
                logger.warning(f"Cannot checkpoint non-existent state: {conversation_id}")
                return False
            
            # Create checkpoint
            return await self.conversation_manager.create_langgraph_checkpoint(
                conversation_id, current_state, checkpoint_name
            )
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return False
    
    async def rollback_to_checkpoint(
        self,
        conversation_id: str,
        checkpoint_name: str
    ) -> Optional[TechnicalConversationState]:
        """
        Rollback conversation to a specific checkpoint.
        
        Args:
            conversation_id: Unique conversation identifier
            checkpoint_name: Name of checkpoint to rollback to
            
        Returns:
            Rolled back state if successful, None otherwise
        """
        try:
            # Invalidate cache for this conversation
            cache_patterns = [
                self._get_cache_key(conversation_id),
                self._get_cache_key(conversation_id, checkpoint_name)
            ]
            
            for pattern in cache_patterns:
                if pattern in self._state_cache:
                    del self._state_cache[pattern]
                    del self._cache_timestamps[pattern]
            
            # Perform rollback
            rolled_back_state = await self.conversation_manager.rollback_langgraph_state(
                conversation_id, checkpoint_name
            )
            
            if rolled_back_state:
                # Cache the rolled back state
                cache_key = self._get_cache_key(conversation_id)
                self._cache_state(cache_key, rolled_back_state)
            
            return rolled_back_state
            
        except Exception as e:
            logger.error(f"Failed to rollback to checkpoint: {e}")
            return None
    
    async def list_checkpoints(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            List of checkpoint metadata dictionaries
        """
        try:
            return await self.conversation_manager.list_langgraph_checkpoints(conversation_id)
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    async def batch_save_states(
        self,
        states: List[tuple[str, TechnicalConversationState, Optional[str]]]
    ) -> Dict[str, bool]:
        """
        Save multiple conversation states in batch for efficiency.
        
        Args:
            states: List of (conversation_id, state, checkpoint_name) tuples
            
        Returns:
            Dictionary mapping conversation_id to save success status
        """
        results = {}
        
        # Process in parallel with limited concurrency
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent saves
        
        async def save_single(conversation_id: str, state: TechnicalConversationState, checkpoint_name: Optional[str]):
            async with semaphore:
                success = await self.save_state(conversation_id, state, checkpoint_name)
                return conversation_id, success
        
        # Execute all saves concurrently
        tasks = [
            save_single(conv_id, state, checkpoint)
            for conv_id, state, checkpoint in states
        ]
        
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, tuple):
                    conv_id, success = result
                    results[conv_id] = success
                else:
                    # Handle exceptions
                    logger.error(f"Batch save exception: {result}")
            
            self.telemetry_manager.log_custom_event(
                "langgraph.service.batch_save",
                {
                    "total_states": len(states),
                    "successful_saves": sum(1 for success in results.values() if success),
                    "failed_saves": sum(1 for success in results.values() if not success)
                }
            )
            
        except Exception as e:
            logger.error(f"Batch save operation failed: {e}")
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        cache_hit_rate = 0.0
        total_cache_operations = self._performance_metrics['cache_hits'] + self._performance_metrics['cache_misses']
        
        if total_cache_operations > 0:
            cache_hit_rate = self._performance_metrics['cache_hits'] / total_cache_operations
        
        avg_save_time = 0.0
        if self._performance_metrics['save_operations'] > 0:
            avg_save_time = self._performance_metrics['total_save_time'] / self._performance_metrics['save_operations']
        
        avg_load_time = 0.0
        if self._performance_metrics['load_operations'] > 0:
            avg_load_time = self._performance_metrics['total_load_time'] / self._performance_metrics['load_operations']
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._state_cache),
            'avg_save_time_ms': avg_save_time * 1000,
            'avg_load_time_ms': avg_load_time * 1000,
            'total_operations': (
                self._performance_metrics['save_operations'] + 
                self._performance_metrics['load_operations']
            ),
            'circuit_breaker_open': self._is_circuit_breaker_open(),
            'failure_count': self._failure_count,
            **self._performance_metrics
        }
    
    def clear_cache(self, conversation_id_pattern: Optional[str] = None):
        """
        Clear cache entries, optionally matching a pattern.
        
        Args:
            conversation_id_pattern: If provided, only clear entries for this conversation ID
        """
        if conversation_id_pattern:
            # Clear specific conversation from cache
            keys_to_remove = [
                key for key in self._state_cache.keys()
                if key.startswith(f"{conversation_id_pattern}:")
            ]
            
            for key in keys_to_remove:
                self._state_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
            
            logger.info(f"Cleared {len(keys_to_remove)} cache entries for conversation {conversation_id_pattern}")
        else:
            # Clear entire cache
            cache_size = len(self._state_cache)
            self._state_cache.clear()
            self._cache_timestamps.clear()
            
            logger.info(f"Cleared entire state cache ({cache_size} entries)")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the state service.
        
        Returns:
            Dictionary with health status and metrics
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'circuit_breaker_open': self._is_circuit_breaker_open(),
            'cache_health': {
                'size': len(self._state_cache),
                'hit_rate': 0.0
            }
        }
        
        try:
            # Test basic operations
            test_state = ConversationStateManager.create_initial_state(
                conversation_id="health_check_test",
                initial_query="Health check query",
                domain="cloud",
                thread_id="health_check_thread",
                user_id="health_check_user",
                channel_id="health_check_channel",
                correlation_id="health_check_corr"
            )
            
            # Test serialization/deserialization
            serialized = ConversationStateManager.serialize_state(test_state)
            deserialized = ConversationStateManager.deserialize_state(serialized)
            
            if deserialized['conversation_id'] != test_state['conversation_id']:
                health_status['status'] = 'unhealthy'
                health_status['error'] = 'Serialization test failed'
            
            # Calculate cache hit rate
            total_cache_ops = self._performance_metrics['cache_hits'] + self._performance_metrics['cache_misses']
            if total_cache_ops > 0:
                health_status['cache_health']['hit_rate'] = self._performance_metrics['cache_hits'] / total_cache_ops
            
            # Add performance metrics
            health_status['performance'] = self.get_performance_metrics()
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)


# Global instance for dependency injection
_langgraph_state_service = None

def get_langgraph_state_service() -> LangGraphStateService:
    """Get singleton instance of LangGraphStateService"""
    global _langgraph_state_service
    
    if _langgraph_state_service is None:
        _langgraph_state_service = LangGraphStateService()
    
    return _langgraph_state_service