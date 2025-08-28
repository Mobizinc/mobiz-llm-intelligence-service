"""
Parallel Execution and Synchronization for LangGraph Workflows

This module implements parallel node execution with proper synchronization
at merge points, resource pooling, deadlock prevention, and timeout handling
for parallel branches. It provides performance optimization through concurrent
execution while maintaining data consistency.

Features:
- Parallel execution for independent nodes
- Proper synchronization at merge points
- Resource pooling for parallel operations
- Deadlock prevention mechanisms
- Timeout handling for parallel branches
- Performance monitoring for parallel paths
- Graceful degradation to sequential on resource constraints

Part of Epic 3: Orchestration & Intelligent Routing
Story 3.4: Implement Parallel Execution and Synchronization
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..nodes.base import BaseNode, NodeError, NodeTimeoutError
from ..models.conversation_state import (
    TechnicalConversationState,
    ConversationStage,
    ConversationStateManager
)
from ..monitoring.state_metrics import StateMetricsCollector
from ..telemetry.state_telemetry import StateOperationTracer

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of parallel tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SynchronizationStrategy(str, Enum):
    """Strategies for synchronizing parallel results"""
    WAIT_ALL = "wait_all"           # Wait for all tasks to complete
    FIRST_SUCCESS = "first_success" # Return on first successful completion
    MAJORITY = "majority"           # Wait for majority to complete
    TIMEOUT_PARTIAL = "timeout_partial"  # Return partial results on timeout


@dataclass
class ParallelTask:
    """Represents a task for parallel execution"""
    task_id: str
    node_name: str
    node_instance: BaseNode
    input_state: TechnicalConversationState
    priority: int = 1  # Higher number = higher priority
    timeout: int = 30
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result_state: Optional[TechnicalConversationState] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring"""
        return {
            'task_id': self.task_id,
            'node_name': self.node_name,
            'priority': self.priority,
            'timeout': self.timeout,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'execution_time': self.execution_time,
            'error': self.error,
            'resource_usage': self.resource_usage
        }


@dataclass
class SynchronizationPoint:
    """Represents a synchronization point for merging parallel results"""
    sync_id: str
    required_tasks: Set[str]  # Task IDs that must complete
    optional_tasks: Set[str]  # Task IDs that are optional
    strategy: SynchronizationStrategy
    timeout: int = 60
    merge_function: Optional[Callable] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # State tracking
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    task_results: Dict[str, TechnicalConversationState] = field(default_factory=dict)
    is_completed: bool = False
    merged_result: Optional[TechnicalConversationState] = None


@dataclass
class ParallelExecutionResult:
    """Results from parallel execution"""
    successful_tasks: List[ParallelTask]
    failed_tasks: List[ParallelTask]
    merged_state: TechnicalConversationState
    execution_summary: Dict[str, Any]
    synchronization_points: List[SynchronizationPoint]
    total_execution_time: float
    resource_usage_summary: Dict[str, Any]


class ResourcePool:
    """Resource pool for managing parallel execution resources"""
    
    def __init__(
        self,
        max_workers: int = 10,
        max_memory_mb: int = 1024,
        max_cpu_percent: float = 80.0
    ):
        self.max_workers = max_workers
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        
        # Resource tracking
        self._active_workers = 0
        self._active_memory_mb = 0.0
        self._worker_lock = threading.Lock()
        
        # Thread pool for parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Semaphore for controlling concurrent access
        self.execution_semaphore = asyncio.Semaphore(max_workers)
        
        logger.info(f"ResourcePool initialized: {max_workers} workers, {max_memory_mb}MB memory limit")
    
    async def acquire_resources(
        self, 
        estimated_memory_mb: float = 50.0,
        priority: int = 1
    ) -> bool:
        """Acquire resources for task execution"""
        
        try:
            # Check if resources are available
            async with self.execution_semaphore:
                with self._worker_lock:
                    if (self._active_workers >= self.max_workers or 
                        self._active_memory_mb + estimated_memory_mb > self.max_memory_mb):
                        return False
                    
                    # Reserve resources
                    self._active_workers += 1
                    self._active_memory_mb += estimated_memory_mb
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to acquire resources: {e}")
            return False
    
    def release_resources(self, used_memory_mb: float = 50.0) -> None:
        """Release resources after task completion"""
        
        try:
            with self._worker_lock:
                self._active_workers = max(0, self._active_workers - 1)
                self._active_memory_mb = max(0, self._active_memory_mb - used_memory_mb)
                
        except Exception as e:
            logger.error(f"Failed to release resources: {e}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource usage status"""
        
        with self._worker_lock:
            return {
                "active_workers": self._active_workers,
                "max_workers": self.max_workers,
                "active_memory_mb": self._active_memory_mb,
                "max_memory_mb": self.max_memory_mb,
                "worker_utilization": self._active_workers / self.max_workers,
                "memory_utilization": self._active_memory_mb / self.max_memory_mb
            }
    
    def shutdown(self) -> None:
        """Shutdown the resource pool"""
        self.thread_pool.shutdown(wait=True)


class DeadlockDetector:
    """Detects and prevents deadlocks in parallel execution"""
    
    def __init__(self):
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._task_status: Dict[str, TaskStatus] = {}
        self._lock = threading.Lock()
    
    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Add a dependency between tasks"""
        
        with self._lock:
            if task_id not in self._dependency_graph:
                self._dependency_graph[task_id] = set()
            self._dependency_graph[task_id].add(depends_on)
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status for deadlock detection"""
        
        with self._lock:
            self._task_status[task_id] = status
    
    def detect_deadlock(self) -> Optional[List[str]]:
        """Detect if there's a deadlock in the dependency graph"""
        
        with self._lock:
            # Simple cycle detection using DFS
            visited = set()
            rec_stack = set()
            
            def has_cycle(node: str) -> bool:
                visited.add(node)
                rec_stack.add(node)
                
                # Check all dependencies
                for neighbor in self._dependency_graph.get(node, set()):
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
                
                rec_stack.remove(node)
                return False
            
            # Check for cycles starting from any unvisited node
            for task_id in self._dependency_graph:
                if task_id not in visited:
                    if has_cycle(task_id):
                        # Return the cycle (simplified)
                        return list(rec_stack)
            
            return None
    
    def clear_task(self, task_id: str) -> None:
        """Clear task from deadlock tracking"""
        
        with self._lock:
            self._dependency_graph.pop(task_id, None)
            self._task_status.pop(task_id, None)


class ParallelExecutor:
    """
    Parallel executor for LangGraph nodes with synchronization support.
    
    Provides:
    - Concurrent execution of independent nodes
    - Resource management and pooling
    - Synchronization at merge points
    - Deadlock detection and prevention
    - Timeout handling with graceful degradation
    - Performance monitoring and metrics
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        default_timeout: int = 30,
        enable_deadlock_detection: bool = True,
        enable_resource_monitoring: bool = True
    ):
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.enable_deadlock_detection = enable_deadlock_detection
        self.enable_resource_monitoring = enable_resource_monitoring
        
        # Initialize components
        self.resource_pool = ResourcePool(max_workers=max_workers)
        self.deadlock_detector = DeadlockDetector() if enable_deadlock_detection else None
        self.metrics_collector = StateMetricsCollector()
        self.telemetry_service = StateOperationTracer()
        
        # Execution tracking
        self.active_tasks: Dict[str, ParallelTask] = {}
        self.synchronization_points: Dict[str, SynchronizationPoint] = {}
        
        # Performance tracking
        self._execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"ParallelExecutor initialized: {max_workers} max workers")
    
    async def execute_parallel(
        self,
        tasks: List[ParallelTask],
        synchronization_strategy: SynchronizationStrategy = SynchronizationStrategy.WAIT_ALL,
        timeout: Optional[int] = None
    ) -> ParallelExecutionResult:
        """
        Execute multiple tasks in parallel with synchronization.
        
        Args:
            tasks: List of tasks to execute in parallel
            synchronization_strategy: How to synchronize results
            timeout: Overall timeout for parallel execution
            
        Returns:
            ParallelExecutionResult with aggregated results
        """
        
        execution_start = time.time()
        total_timeout = timeout or (self.default_timeout * 2)
        
        try:
            logger.info(
                f"Starting parallel execution of {len(tasks)} tasks",
                extra={
                    "task_count": len(tasks),
                    "strategy": synchronization_strategy.value,
                    "timeout": total_timeout
                }
            )
            
            # Pre-execution checks
            if not tasks:
                raise ValueError("No tasks provided for parallel execution")
            
            if len(tasks) > self.max_workers:
                logger.warning(f"Task count ({len(tasks)}) exceeds max workers ({self.max_workers}), will queue excess tasks")
            
            # Create synchronization point
            sync_point = self._create_synchronization_point(
                tasks, synchronization_strategy, total_timeout
            )
            
            # Execute tasks with resource management
            task_coroutines = [
                self._execute_task_with_resources(task, sync_point.sync_id)
                for task in tasks
            ]
            
            # Wait for completion based on strategy
            successful_tasks, failed_tasks, merged_state = await self._wait_for_synchronization(
                sync_point, task_coroutines, total_timeout
            )
            
            execution_time = time.time() - execution_start
            
            # Create execution summary
            execution_summary = self._create_execution_summary(
                tasks, successful_tasks, failed_tasks, execution_time
            )
            
            # Record metrics
            await self._record_execution_metrics(execution_summary)
            
            logger.info(
                f"Parallel execution completed",
                extra={
                    "successful_tasks": len(successful_tasks),
                    "failed_tasks": len(failed_tasks),
                    "execution_time": execution_time,
                    "strategy": synchronization_strategy.value
                }
            )
            
            return ParallelExecutionResult(
                successful_tasks=successful_tasks,
                failed_tasks=failed_tasks,
                merged_state=merged_state,
                execution_summary=execution_summary,
                synchronization_points=[sync_point],
                total_execution_time=execution_time,
                resource_usage_summary=self.resource_pool.get_resource_status()
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            logger.error(f"Parallel execution failed: {e}", exc_info=True)
            
            # Return error result
            error_state = tasks[0].input_state.copy() if tasks else {}
            error_state.update({
                'stage': ConversationStage.ERROR,
                'parallel_execution_error': str(e)
            })
            
            return ParallelExecutionResult(
                successful_tasks=[],
                failed_tasks=tasks,
                merged_state=error_state,
                execution_summary={
                    "error": str(e),
                    "execution_time": execution_time
                },
                synchronization_points=[],
                total_execution_time=execution_time,
                resource_usage_summary=self.resource_pool.get_resource_status()
            )
    
    def _create_synchronization_point(
        self,
        tasks: List[ParallelTask],
        strategy: SynchronizationStrategy,
        timeout: int
    ) -> SynchronizationPoint:
        """Create synchronization point for tasks"""
        
        sync_id = f"sync_{uuid.uuid4().hex[:8]}"
        required_tasks = {task.task_id for task in tasks}
        
        sync_point = SynchronizationPoint(
            sync_id=sync_id,
            required_tasks=required_tasks,
            optional_tasks=set(),
            strategy=strategy,
            timeout=timeout,
            merge_function=self._default_merge_function
        )
        
        self.synchronization_points[sync_id] = sync_point
        return sync_point
    
    async def _execute_task_with_resources(
        self,
        task: ParallelTask,
        sync_id: str
    ) -> ParallelTask:
        """Execute a single task with resource management"""
        
        task.started_at = datetime.now(timezone.utc).isoformat()
        task.status = TaskStatus.RUNNING
        
        if self.deadlock_detector:
            self.deadlock_detector.update_task_status(task.task_id, task.status)
        
        try:
            # Acquire resources
            resources_acquired = await self.resource_pool.acquire_resources(
                estimated_memory_mb=50.0,  # Estimate based on node type
                priority=task.priority
            )
            
            if not resources_acquired:
                # Fallback to sequential execution
                logger.warning(f"Resources not available for task {task.task_id}, executing sequentially")
                return await self._execute_task_sequential(task)
            
            try:
                # Execute with timeout
                result_state = await asyncio.wait_for(
                    task.node_instance.safe_process(task.input_state),
                    timeout=task.timeout
                )
                
                task.result_state = result_state
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc).isoformat()
                
                # Update synchronization point
                sync_point = self.synchronization_points.get(sync_id)
                if sync_point:
                    sync_point.completed_tasks.add(task.task_id)
                    sync_point.task_results[task.task_id] = result_state
                
                logger.debug(f"Task {task.task_id} completed successfully")
                
            finally:
                # Always release resources
                self.resource_pool.release_resources(used_memory_mb=50.0)
                
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = f"Task timed out after {task.timeout}s"
            
            # Update synchronization point
            sync_point = self.synchronization_points.get(sync_id)
            if sync_point:
                sync_point.failed_tasks.add(task.task_id)
            
            logger.warning(f"Task {task.task_id} timed out")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            # Update synchronization point
            sync_point = self.synchronization_points.get(sync_id)
            if sync_point:
                sync_point.failed_tasks.add(task.task_id)
            
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            task.execution_time = time.time() - time.mktime(
                datetime.fromisoformat(task.started_at.replace('Z', '+00:00')).timetuple()
            )
            
            if self.deadlock_detector:
                self.deadlock_detector.update_task_status(task.task_id, task.status)
        
        return task
    
    async def _execute_task_sequential(self, task: ParallelTask) -> ParallelTask:
        """Execute task sequentially as fallback"""
        
        try:
            result_state = await task.node_instance.safe_process(task.input_state)
            task.result_state = result_state
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
        
        return task
    
    async def _wait_for_synchronization(
        self,
        sync_point: SynchronizationPoint,
        task_coroutines: List,
        timeout: int
    ) -> Tuple[List[ParallelTask], List[ParallelTask], TechnicalConversationState]:
        """Wait for synchronization based on strategy"""
        
        successful_tasks = []
        failed_tasks = []
        
        try:
            if sync_point.strategy == SynchronizationStrategy.WAIT_ALL:
                # Wait for all tasks
                completed_tasks = await asyncio.gather(*task_coroutines, return_exceptions=True)
                
            elif sync_point.strategy == SynchronizationStrategy.FIRST_SUCCESS:
                # Wait for first successful completion
                completed_tasks = []
                for coro in asyncio.as_completed(task_coroutines, timeout=timeout):
                    task = await coro
                    completed_tasks.append(task)
                    if task.status == TaskStatus.COMPLETED:
                        # Cancel remaining tasks
                        for remaining_coro in task_coroutines:
                            if not remaining_coro.done():
                                remaining_coro.cancel()
                        break
                        
            elif sync_point.strategy == SynchronizationStrategy.MAJORITY:
                # Wait for majority to complete
                majority_count = len(task_coroutines) // 2 + 1
                completed_tasks = []
                
                for coro in asyncio.as_completed(task_coroutines, timeout=timeout):
                    task = await coro
                    completed_tasks.append(task)
                    
                    success_count = sum(1 for t in completed_tasks if t.status == TaskStatus.COMPLETED)
                    if success_count >= majority_count:
                        break
                        
            else:  # TIMEOUT_PARTIAL
                # Return whatever completes within timeout
                try:
                    completed_tasks = await asyncio.wait_for(
                        asyncio.gather(*task_coroutines, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Get partial results
                    completed_tasks = []
                    for coro in task_coroutines:
                        if coro.done():
                            try:
                                completed_tasks.append(coro.result())
                            except Exception:
                                pass
            
            # Categorize results
            for task in completed_tasks:
                if isinstance(task, Exception):
                    continue
                    
                if task.status == TaskStatus.COMPLETED:
                    successful_tasks.append(task)
                else:
                    failed_tasks.append(task)
            
            # Merge results
            merged_state = sync_point.merge_function(
                [task.result_state for task in successful_tasks if task.result_state],
                sync_point.strategy
            )
            
            return successful_tasks, failed_tasks, merged_state
            
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            
            # Return error state
            base_state = task_coroutines[0].input_state if task_coroutines else {}
            error_state = base_state.copy()
            error_state.update({
                'stage': ConversationStage.ERROR,
                'synchronization_error': str(e)
            })
            
            return [], list(completed_tasks) if 'completed_tasks' in locals() else [], error_state
    
    def _default_merge_function(
        self,
        states: List[TechnicalConversationState],
        strategy: SynchronizationStrategy
    ) -> TechnicalConversationState:
        """Default function for merging parallel results"""
        
        if not states:
            return {
                'stage': ConversationStage.ERROR,
                'response': 'No successful parallel results to merge',
                'parallel_execution_summary': {
                    'merged_states': 0,
                    'strategy': strategy.value
                }
            }
        
        if len(states) == 1:
            return states[0]
        
        # Merge strategy based on synchronization approach
        base_state = states[0].copy()
        
        # Merge extracted information from all states
        merged_extracted_info = {}
        merged_accumulated_info = {}
        all_missing_fields = set()
        all_required_fields = set()
        
        for state in states:
            merged_extracted_info.update(state.get('extracted_info', {}))
            merged_accumulated_info.update(state.get('accumulated_info', {}))
            all_missing_fields.update(state.get('missing_fields', []))
            all_required_fields.update(state.get('required_fields', []))
        
        # Update base state with merged information
        base_state.update({
            'extracted_info': merged_extracted_info,
            'accumulated_info': merged_accumulated_info,
            'missing_fields': list(all_missing_fields),
            'required_fields': list(all_required_fields),
            'parallel_execution_summary': {
                'merged_states': len(states),
                'strategy': strategy.value,
                'merge_timestamp': datetime.now(timezone.utc).isoformat()
            }
        })
        
        # Determine final stage based on merged results
        stages = [state.get('stage') for state in states]
        if ConversationStage.SUFFICIENT in stages:
            base_state['stage'] = ConversationStage.SUFFICIENT
        elif ConversationStage.ANALYZING in stages:
            base_state['stage'] = ConversationStage.ANALYZING
        else:
            base_state['stage'] = ConversationStage.GATHERING
        
        return base_state
    
    def _create_execution_summary(
        self,
        all_tasks: List[ParallelTask],
        successful_tasks: List[ParallelTask],
        failed_tasks: List[ParallelTask],
        execution_time: float
    ) -> Dict[str, Any]:
        """Create execution summary for monitoring"""
        
        return {
            "total_tasks": len(all_tasks),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(successful_tasks) / len(all_tasks) if all_tasks else 0.0,
            "total_execution_time": execution_time,
            "average_task_time": sum(t.execution_time for t in successful_tasks) / len(successful_tasks) if successful_tasks else 0.0,
            "max_task_time": max(t.execution_time for t in successful_tasks) if successful_tasks else 0.0,
            "min_task_time": min(t.execution_time for t in successful_tasks) if successful_tasks else 0.0,
            "task_details": [task.to_dict() for task in all_tasks],
            "resource_usage": self.resource_pool.get_resource_status()
        }
    
    async def _record_execution_metrics(self, execution_summary: Dict[str, Any]) -> None:
        """Record execution metrics for monitoring"""
        
        try:
            self.metrics_collector.record_histogram(
                "parallel_execution_time_seconds",
                execution_summary["total_execution_time"]
            )
            
            self.metrics_collector.record_histogram(
                "parallel_success_rate",
                execution_summary["success_rate"]
            )
            
            self.metrics_collector.increment_counter(
                "parallel_executions_total",
                success_rate=f"{execution_summary['success_rate']:.1f}"
            )
            
            await self.telemetry_service.record_parallel_execution(
                total_tasks=execution_summary["total_tasks"],
                successful_tasks=execution_summary["successful_tasks"],
                execution_time=execution_summary["total_execution_time"],
                resource_usage=execution_summary["resource_usage"]
            )
            
        except Exception as e:
            logger.warning(f"Failed to record execution metrics: {e}")
    
    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Add dependency between tasks for deadlock detection"""
        
        if self.deadlock_detector:
            self.deadlock_detector.add_dependency(task_id, depends_on)
    
    def check_deadlock(self) -> Optional[List[str]]:
        """Check for deadlocks in current execution"""
        
        if self.deadlock_detector:
            return self.deadlock_detector.detect_deadlock()
        return None
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        
        return {
            "active_tasks": len(self.active_tasks),
            "active_sync_points": len(self.synchronization_points),
            "resource_status": self.resource_pool.get_resource_status(),
            "deadlock_detection_enabled": self.enable_deadlock_detection,
            "resource_monitoring_enabled": self.enable_resource_monitoring
        }
    
    def shutdown(self) -> None:
        """Shutdown parallel executor"""
        
        # Cancel all active tasks
        for task in self.active_tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
        
        # Clear synchronization points
        self.synchronization_points.clear()
        
        # Shutdown resource pool
        self.resource_pool.shutdown()
        
        logger.info("ParallelExecutor shutdown completed")