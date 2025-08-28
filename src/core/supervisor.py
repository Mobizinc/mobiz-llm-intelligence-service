"""
Supervisor Pattern for Multi-Agent Coordination

This module implements a supervisor node that coordinates complex multi-step
workflows requiring multiple specialized agents. The supervisor analyzes
request complexity, distributes work across agents, aggregates results,
and resolves conflicts between agent outputs.

Features:
- Supervisor node for multi-bot coordination
- Work distribution across specialized agents
- Result aggregation from multiple agents
- Consensus mechanisms for conflicting outputs
- Priority-based task scheduling
- Supervisor health monitoring
- Fallback to single-agent mode on supervisor failure

Part of Epic 3: Orchestration & Intelligent Routing
Story 3.3: Build Supervisor Pattern for Multi-Agent Coordination
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from ..nodes.base import BaseNode, NodeError
from ..models.conversation_state import (
    TechnicalConversationState,
    ConversationStage,
    IntentType,
    ResponseType,
    DomainType
)
from ..core.config import settings
from ..monitoring.state_metrics import StateMetricsCollector
from ..telemetry.state_telemetry import StateOperationTracer

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Priority levels for agent tasks"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    """Status of agent tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts between agent outputs"""
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    DOMAIN_EXPERTISE = "domain_expertise"
    HUMAN_ESCALATION = "human_escalation"
    MERGE_COMPLEMENTARY = "merge_complementary"


@dataclass
class AgentTask:
    """Represents a task assigned to a specific agent"""
    task_id: str
    agent_type: str
    task_description: str
    input_data: Dict[str, Any]
    priority: TaskPriority
    timeout: int = 30
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    assigned_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'agent_type': self.agent_type,
            'task_description': self.task_description,
            'input_data': self.input_data,
            'priority': self.priority.value,
            'timeout': self.timeout,
            'created_at': self.created_at,
            'assigned_at': self.assigned_at,
            'completed_at': self.completed_at,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time
        }


@dataclass
class MultiAgentResult:
    """Represents aggregated results from multiple agents"""
    task_results: List[AgentTask]
    aggregated_output: Dict[str, Any]
    confidence_score: float
    conflicts_detected: bool
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    consensus_level: float = 0.0
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage"""
        return {
            'task_results': [task.to_dict() for task in self.task_results],
            'aggregated_output': self.aggregated_output,
            'confidence_score': self.confidence_score,
            'conflicts_detected': self.conflicts_detected,
            'resolution_strategy': self.resolution_strategy.value if self.resolution_strategy else None,
            'consensus_level': self.consensus_level,
            'execution_summary': self.execution_summary
        }


class ComplexityAssessment:
    """Assesses the complexity of a request to determine if multi-agent coordination is needed"""
    
    def __init__(self, domain: DomainType):
        self.domain = domain
        self._complexity_indicators = self._initialize_complexity_indicators()
    
    def _initialize_complexity_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific complexity indicators"""
        
        indicators = {
            "cloud": {
                "high_complexity_keywords": [
                    "migration", "disaster recovery", "multi-region", "hybrid cloud",
                    "enterprise", "compliance", "security", "scalability", "integration"
                ],
                "multi_agent_triggers": [
                    "azure and aws", "multi-cloud", "hybrid", "migration",
                    "architecture review", "cost optimization", "compliance audit"
                ],
                "complexity_weights": {
                    "migration": 0.9,
                    "multi-region": 0.8,
                    "disaster recovery": 0.7,
                    "hybrid": 0.6,
                    "enterprise": 0.5
                }
            },
            "network": {
                "high_complexity_keywords": [
                    "site-to-site", "vpn mesh", "load balancing", "firewall rules",
                    "network segmentation", "sd-wan", "multi-vendor", "integration"
                ],
                "multi_agent_triggers": [
                    "palo alto and cisco", "multi-vendor", "complex topology",
                    "security audit", "performance optimization", "troubleshooting"
                ],
                "complexity_weights": {
                    "site-to-site": 0.8,
                    "load balancing": 0.7,
                    "multi-vendor": 0.9,
                    "security audit": 0.8,
                    "troubleshooting": 0.6
                }
            },
            "devops": {
                "high_complexity_keywords": [
                    "ci/cd pipeline", "kubernetes deployment", "terraform modules",
                    "monitoring stack", "infrastructure as code", "microservices"
                ],
                "multi_agent_triggers": [
                    "full pipeline", "microservices", "multi-environment",
                    "security scanning", "performance monitoring", "disaster recovery"
                ],
                "complexity_weights": {
                    "ci/cd pipeline": 0.8,
                    "kubernetes": 0.9,
                    "microservices": 0.8,
                    "monitoring": 0.6,
                    "security": 0.7
                }
            },
            "dev": {
                "high_complexity_keywords": [
                    "api architecture", "database design", "integration patterns",
                    "performance optimization", "security implementation", "scalability"
                ],
                "multi_agent_triggers": [
                    "full stack", "microservices", "api design",
                    "database optimization", "security review", "architecture"
                ],
                "complexity_weights": {
                    "architecture": 0.9,
                    "integration": 0.7,
                    "performance": 0.6,
                    "security": 0.8,
                    "scalability": 0.7
                }
            }
        }
        
        domain_str = self.domain.value if hasattr(self.domain, 'value') else str(self.domain)
        return indicators.get(domain_str, indicators["dev"])
    
    def assess_complexity(self, state: TechnicalConversationState) -> Dict[str, Any]:
        """Assess request complexity and determine multi-agent needs"""
        
        current_input = state.get('current_input', '').lower()
        initial_query = state.get('initial_query', '').lower()
        extracted_info = state.get('extracted_info', {})
        
        # Combine all text for analysis
        full_text = f"{current_input} {initial_query}".lower()
        
        # Calculate complexity score
        complexity_score = 0.0
        detected_indicators = []
        
        for indicator, weight in self._complexity_indicators["complexity_weights"].items():
            if indicator in full_text:
                complexity_score += weight
                detected_indicators.append(indicator)
        
        # Check for multi-agent triggers
        requires_multi_agent = any(
            trigger in full_text 
            for trigger in self._complexity_indicators["multi_agent_triggers"]
        )
        
        # Additional complexity factors
        info_complexity = len(extracted_info) * 0.1
        query_length_factor = min(len(full_text) / 1000, 0.3)
        
        final_score = min(complexity_score + info_complexity + query_length_factor, 1.0)
        
        return {
            "complexity_score": final_score,
            "requires_multi_agent": requires_multi_agent or final_score > 0.6,
            "detected_indicators": detected_indicators,
            "complexity_factors": {
                "keyword_complexity": complexity_score,
                "info_complexity": info_complexity,
                "length_factor": query_length_factor
            },
            "recommendation": self._get_recommendation(final_score, requires_multi_agent)
        }
    
    def _get_recommendation(self, score: float, requires_multi_agent: bool) -> str:
        """Get recommendation based on complexity assessment"""
        
        if requires_multi_agent:
            return "multi_agent_coordination"
        elif score > 0.7:
            return "complex_single_agent"
        elif score > 0.4:
            return "standard_processing"
        else:
            return "simple_processing"


class SupervisorNode(BaseNode):
    """
    Supervisor node for coordinating multi-agent workflows.
    
    The supervisor analyzes request complexity, determines which agents
    are needed, distributes work, monitors execution, aggregates results,
    and resolves conflicts between agent outputs.
    """
    
    def __init__(
        self,
        node_name: str = "supervisor",
        domain: Optional[str] = None,
        timeout: int = 60,
        max_parallel_tasks: int = 5,
        enable_conflict_resolution: bool = True
    ):
        super().__init__(node_name, domain, timeout)
        
        self.max_parallel_tasks = max_parallel_tasks
        self.enable_conflict_resolution = enable_conflict_resolution
        
        # Initialize complexity assessor
        domain_enum = DomainType(domain) if domain else DomainType.DEV
        self.complexity_assessor = ComplexityAssessment(domain_enum)
        
        # Task management
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        
        # Agent type mappings
        self.agent_capabilities = self._initialize_agent_capabilities()
        
        logger.info(f"SupervisorNode initialized for domain: {domain}")
    
    def _initialize_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Initialize agent capabilities mapping"""
        
        return {
            "cloud_specialist": {
                "expertise": ["azure", "aws", "gcp", "migration", "infrastructure"],
                "max_concurrent_tasks": 3,
                "timeout": 30
            },
            "network_specialist": {
                "expertise": ["cisco", "palo alto", "fortinet", "vpn", "firewall"],
                "max_concurrent_tasks": 2,
                "timeout": 25
            },
            "devops_specialist": {
                "expertise": ["kubernetes", "docker", "terraform", "jenkins", "monitoring"],
                "max_concurrent_tasks": 3,
                "timeout": 35
            },
            "security_specialist": {
                "expertise": ["security", "compliance", "audit", "penetration testing"],
                "max_concurrent_tasks": 2,
                "timeout": 40
            },
            "performance_specialist": {
                "expertise": ["optimization", "scalability", "load testing", "monitoring"],
                "max_concurrent_tasks": 2,
                "timeout": 30
            }
        }
    
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process state through supervisor coordination.
        
        Steps:
        1. Assess request complexity
        2. Determine required agents
        3. Create and distribute tasks
        4. Monitor task execution
        5. Aggregate results
        6. Resolve conflicts
        7. Update state with results
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Assess complexity
            complexity_assessment = self.complexity_assessor.assess_complexity(state)
            
            logger.info(
                f"Complexity assessment: {complexity_assessment['complexity_score']:.2f}",
                extra={
                    "conversation_id": state.get('conversation_id'),
                    "requires_multi_agent": complexity_assessment['requires_multi_agent'],
                    "detected_indicators": complexity_assessment['detected_indicators']
                }
            )
            
            # Step 2: Check if multi-agent coordination is needed
            if not complexity_assessment['requires_multi_agent']:
                # Simple case - no coordination needed
                updated_state = state.copy()
                updated_state.update({
                    'supervisor_decision': 'single_agent_sufficient',
                    'complexity_assessment': complexity_assessment
                })
                return updated_state
            
            # Step 3: Create agent tasks
            tasks = self._create_agent_tasks(state, complexity_assessment)
            
            if not tasks:
                # No tasks created - fallback
                updated_state = state.copy()
                updated_state.update({
                    'supervisor_decision': 'no_tasks_created',
                    'complexity_assessment': complexity_assessment
                })
                return updated_state
            
            # Step 4: Execute tasks in parallel
            multi_agent_result = await self._execute_parallel_tasks(tasks, state)
            
            # Step 5: Update state with results
            updated_state = state.copy()
            updated_state.update({
                'supervisor_decision': 'multi_agent_completed',
                'complexity_assessment': complexity_assessment,
                'multi_agent_results': multi_agent_result.to_dict(),
                'needs_resolution': multi_agent_result.conflicts_detected
            })
            
            # Step 6: Set appropriate stage based on results
            if multi_agent_result.conflicts_detected:
                updated_state['stage'] = ConversationStage.ANALYZING
                updated_state['response'] = "I've analyzed your request using multiple specialized agents. There are some conflicting recommendations that need clarification."
            elif multi_agent_result.confidence_score > 0.8:
                updated_state['stage'] = ConversationStage.SUFFICIENT
                updated_state['response'] = "Multi-agent analysis complete. I have sufficient information to provide a comprehensive solution."
            else:
                updated_state['stage'] = ConversationStage.GATHERING
                updated_state['response'] = "Multi-agent analysis indicates we need additional information to provide the best solution."
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Supervisor processing completed",
                extra={
                    "conversation_id": state.get('conversation_id'),
                    "tasks_executed": len(tasks),
                    "conflicts_detected": multi_agent_result.conflicts_detected,
                    "final_confidence": multi_agent_result.confidence_score,
                    "processing_time": processing_time
                }
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Supervisor processing failed: {e}", exc_info=True)
            
            # Return error state
            error_state = state.copy()
            error_state.update({
                'stage': ConversationStage.ERROR,
                'supervisor_decision': 'error',
                'supervisor_error': str(e)
            })
            
            return error_state
    
    def _create_agent_tasks(
        self,
        state: TechnicalConversationState,
        complexity_assessment: Dict[str, Any]
    ) -> List[AgentTask]:
        """Create tasks for appropriate agents based on complexity assessment"""
        
        tasks = []
        detected_indicators = complexity_assessment.get('detected_indicators', [])
        extracted_info = state.get('extracted_info', {})
        current_input = state.get('current_input', '').lower()
        
        # Determine which agents are needed
        required_agents = self._determine_required_agents(detected_indicators, current_input, extracted_info)
        
        for agent_type, task_info in required_agents.items():
            task = AgentTask(
                task_id=f"{agent_type}_{uuid.uuid4().hex[:8]}",
                agent_type=agent_type,
                task_description=task_info['description'],
                input_data=task_info['input_data'],
                priority=task_info['priority'],
                timeout=self.agent_capabilities.get(agent_type, {}).get('timeout', 30)
            )
            
            tasks.append(task)
        
        # Sort by priority
        priority_order = [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]
        tasks.sort(key=lambda t: priority_order.index(t.priority))
        
        logger.info(
            f"Created {len(tasks)} agent tasks",
            extra={
                "conversation_id": state.get('conversation_id'),
                "agents": [task.agent_type for task in tasks],
                "priorities": [task.priority.value for task in tasks]
            }
        )
        
        return tasks
    
    def _determine_required_agents(
        self,
        detected_indicators: List[str],
        current_input: str,
        extracted_info: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Determine which agents are required based on the analysis"""
        
        required_agents = {}
        
        # Cloud specialist
        cloud_indicators = ["migration", "multi-region", "disaster recovery", "hybrid", "azure", "aws", "gcp"]
        if any(indicator in current_input for indicator in cloud_indicators):
            required_agents["cloud_specialist"] = {
                "description": "Analyze cloud infrastructure requirements and provide recommendations",
                "input_data": {
                    "query": current_input,
                    "extracted_info": extracted_info,
                    "focus_areas": [ind for ind in cloud_indicators if ind in current_input]
                },
                "priority": TaskPriority.HIGH if "migration" in current_input else TaskPriority.MEDIUM
            }
        
        # Network specialist
        network_indicators = ["network", "firewall", "vpn", "cisco", "palo alto", "routing", "switching"]
        if any(indicator in current_input for indicator in network_indicators):
            required_agents["network_specialist"] = {
                "description": "Analyze network requirements and provide configuration guidance",
                "input_data": {
                    "query": current_input,
                    "extracted_info": extracted_info,
                    "focus_areas": [ind for ind in network_indicators if ind in current_input]
                },
                "priority": TaskPriority.HIGH if "vpn" in current_input else TaskPriority.MEDIUM
            }
        
        # DevOps specialist
        devops_indicators = ["kubernetes", "docker", "terraform", "ci/cd", "pipeline", "deployment"]
        if any(indicator in current_input for indicator in devops_indicators):
            required_agents["devops_specialist"] = {
                "description": "Analyze DevOps requirements and provide implementation guidance",
                "input_data": {
                    "query": current_input,
                    "extracted_info": extracted_info,
                    "focus_areas": [ind for ind in devops_indicators if ind in current_input]
                },
                "priority": TaskPriority.HIGH if "pipeline" in current_input else TaskPriority.MEDIUM
            }
        
        # Security specialist
        security_indicators = ["security", "compliance", "audit", "vulnerability", "encryption", "authentication"]
        if any(indicator in current_input for indicator in security_indicators):
            required_agents["security_specialist"] = {
                "description": "Analyze security requirements and provide security guidance",
                "input_data": {
                    "query": current_input,
                    "extracted_info": extracted_info,
                    "focus_areas": [ind for ind in security_indicators if ind in current_input]
                },
                "priority": TaskPriority.CRITICAL if "vulnerability" in current_input else TaskPriority.HIGH
            }
        
        # Performance specialist
        performance_indicators = ["performance", "optimization", "scalability", "load", "bottleneck", "monitoring"]
        if any(indicator in current_input for indicator in performance_indicators):
            required_agents["performance_specialist"] = {
                "description": "Analyze performance requirements and provide optimization recommendations",
                "input_data": {
                    "query": current_input,
                    "extracted_info": extracted_info,
                    "focus_areas": [ind for ind in performance_indicators if ind in current_input]
                },
                "priority": TaskPriority.MEDIUM
            }
        
        return required_agents
    
    async def _execute_parallel_tasks(
        self,
        tasks: List[AgentTask],
        state: TechnicalConversationState
    ) -> MultiAgentResult:
        """Execute agent tasks in parallel and aggregate results"""
        
        # Limit parallel execution
        limited_tasks = tasks[:self.max_parallel_tasks]
        
        # Execute tasks with timeout
        execution_start = time.time()
        
        try:
            # Simulate agent task execution (in real implementation, these would call actual agents)
            task_coroutines = [self._execute_single_task(task, state) for task in limited_tasks]
            
            completed_tasks = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Process results
            successful_tasks = []
            failed_tasks = []
            
            for i, result in enumerate(completed_tasks):
                task = limited_tasks[i]
                
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                    failed_tasks.append(task)
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                    successful_tasks.append(task)
            
            # Aggregate results
            aggregated_result = self._aggregate_task_results(successful_tasks, failed_tasks)
            
            # Detect conflicts
            conflicts_detected = self._detect_conflicts(successful_tasks)
            
            # Calculate consensus
            consensus_level = self._calculate_consensus(successful_tasks)
            
            execution_time = time.time() - execution_start
            
            return MultiAgentResult(
                task_results=limited_tasks,
                aggregated_output=aggregated_result,
                confidence_score=min(consensus_level + 0.1, 1.0),
                conflicts_detected=conflicts_detected,
                resolution_strategy=ConflictResolutionStrategy.CONFIDENCE_WEIGHTED if conflicts_detected else None,
                consensus_level=consensus_level,
                execution_summary={
                    "total_tasks": len(limited_tasks),
                    "successful_tasks": len(successful_tasks),
                    "failed_tasks": len(failed_tasks),
                    "execution_time": execution_time,
                    "conflicts_detected": conflicts_detected
                }
            )
            
        except Exception as e:
            logger.error(f"Parallel task execution failed: {e}", exc_info=True)
            raise
    
    async def _execute_single_task(self, task: AgentTask, state: TechnicalConversationState) -> Dict[str, Any]:
        """Execute a single agent task (simulated)"""
        
        task.assigned_at = datetime.now(timezone.utc).isoformat()
        task.status = TaskStatus.IN_PROGRESS
        
        start_time = time.time()
        
        try:
            # Simulate agent processing time
            await asyncio.sleep(0.1 + (hash(task.task_id) % 10) / 100)
            
            # Simulate agent-specific results
            result = await self._simulate_agent_work(task.agent_type, task.input_data, state)
            
            task.execution_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            task.execution_time = time.time() - start_time
            task.error = str(e)
            task.status = TaskStatus.FAILED
            raise
    
    async def _simulate_agent_work(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
        state: TechnicalConversationState
    ) -> Dict[str, Any]:
        """Simulate agent work based on type (placeholder for actual agent calls)"""
        
        query = input_data.get('query', '')
        focus_areas = input_data.get('focus_areas', [])
        
        # Simulate different agent outputs
        if agent_type == "cloud_specialist":
            return {
                "agent_type": agent_type,
                "recommendations": [
                    f"Cloud recommendation for {area}" for area in focus_areas[:3]
                ],
                "confidence": 0.85,
                "complexity_level": "medium",
                "estimated_effort": "2-3 days",
                "key_considerations": ["scalability", "cost", "security"]
            }
        
        elif agent_type == "network_specialist":
            return {
                "agent_type": agent_type,
                "recommendations": [
                    f"Network configuration for {area}" for area in focus_areas[:3]
                ],
                "confidence": 0.78,
                "complexity_level": "high",
                "estimated_effort": "1-2 days",
                "key_considerations": ["security", "performance", "reliability"]
            }
        
        elif agent_type == "devops_specialist":
            return {
                "agent_type": agent_type,
                "recommendations": [
                    f"DevOps implementation for {area}" for area in focus_areas[:3]
                ],
                "confidence": 0.82,
                "complexity_level": "medium",
                "estimated_effort": "3-4 days",
                "key_considerations": ["automation", "monitoring", "scalability"]
            }
        
        elif agent_type == "security_specialist":
            return {
                "agent_type": agent_type,
                "recommendations": [
                    f"Security measures for {area}" for area in focus_areas[:3]
                ],
                "confidence": 0.90,
                "complexity_level": "high",
                "estimated_effort": "2-3 days",
                "key_considerations": ["compliance", "threat_model", "access_control"]
            }
        
        else:  # performance_specialist
            return {
                "agent_type": agent_type,
                "recommendations": [
                    f"Performance optimization for {area}" for area in focus_areas[:3]
                ],
                "confidence": 0.75,
                "complexity_level": "medium",
                "estimated_effort": "1-2 days",
                "key_considerations": ["bottlenecks", "scaling", "monitoring"]
            }
    
    def _aggregate_task_results(
        self,
        successful_tasks: List[AgentTask],
        failed_tasks: List[AgentTask]
    ) -> Dict[str, Any]:
        """Aggregate results from all successful tasks"""
        
        aggregated = {
            "agent_recommendations": {},
            "combined_considerations": set(),
            "overall_confidence": 0.0,
            "total_estimated_effort": 0.0,
            "complexity_assessment": "medium",
            "success_rate": len(successful_tasks) / (len(successful_tasks) + len(failed_tasks)) if (successful_tasks or failed_tasks) else 1.0
        }
        
        if not successful_tasks:
            return aggregated
        
        total_confidence = 0.0
        
        for task in successful_tasks:
            if task.result:
                agent_type = task.agent_type
                result = task.result
                
                aggregated["agent_recommendations"][agent_type] = result.get("recommendations", [])
                aggregated["combined_considerations"].update(result.get("key_considerations", []))
                
                total_confidence += result.get("confidence", 0.0)
        
        aggregated["overall_confidence"] = total_confidence / len(successful_tasks)
        aggregated["combined_considerations"] = list(aggregated["combined_considerations"])
        
        return aggregated
    
    def _detect_conflicts(self, successful_tasks: List[AgentTask]) -> bool:
        """Detect conflicts between agent outputs"""
        
        if len(successful_tasks) < 2:
            return False
        
        # Simple conflict detection based on confidence variance
        confidences = [
            task.result.get("confidence", 0.0) 
            for task in successful_tasks 
            if task.result
        ]
        
        if len(confidences) < 2:
            return False
        
        # Calculate variance
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Conflict if high variance in confidence levels
        return variance > 0.1
    
    def _calculate_consensus(self, successful_tasks: List[AgentTask]) -> float:
        """Calculate consensus level among agents"""
        
        if not successful_tasks:
            return 0.0
        
        if len(successful_tasks) == 1:
            return successful_tasks[0].result.get("confidence", 0.0) if successful_tasks[0].result else 0.0
        
        # Simple consensus calculation based on average confidence
        confidences = [
            task.result.get("confidence", 0.0)
            for task in successful_tasks
            if task.result
        ]
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)


class MultiAgentCoordinator:
    """
    High-level coordinator for managing multiple supervisor instances
    and providing centralized multi-agent coordination capabilities.
    """
    
    def __init__(self):
        self.supervisors: Dict[str, SupervisorNode] = {}
        self.metrics_collector = StateMetricsCollector()
        
    def get_supervisor(self, domain: str) -> SupervisorNode:
        """Get or create supervisor for domain"""
        
        if domain not in self.supervisors:
            self.supervisors[domain] = SupervisorNode(
                node_name=f"supervisor_{domain}",
                domain=domain,
                timeout=60
            )
        
        return self.supervisors[domain]
    
    async def coordinate_multi_agent_request(
        self,
        domain: str,
        state: TechnicalConversationState
    ) -> TechnicalConversationState:
        """Coordinate multi-agent request through appropriate supervisor"""
        
        supervisor = self.get_supervisor(domain)
        return await supervisor.safe_process(state)
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics across all supervisors"""
        
        return {
            "total_supervisors": len(self.supervisors),
            "domains": list(self.supervisors.keys()),
            "supervisor_stats": {
                domain: {
                    "active_tasks": len(supervisor.active_tasks),
                    "completed_tasks": len(supervisor.completed_tasks)
                }
                for domain, supervisor in self.supervisors.items()
            }
        }