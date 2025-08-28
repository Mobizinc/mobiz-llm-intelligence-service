"""
Conditional Routing Logic for LangGraph Workflow

This module implements intelligent routing decisions based on conversation state,
intent classification confidence, context history, and extracted information
completeness. The routing engine determines the optimal path through the
workflow for each conversation turn.

Features:
- Intent-based routing with confidence thresholds
- Context-aware routing considering conversation history  
- Dynamic routing based on information completeness
- A/B testing support for routing strategies
- Routing metrics and decision logging
- Override capabilities for testing

Part of Epic 3: Orchestration & Intelligent Routing
Story 3.2: Implement Conditional Routing Logic
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.conversation_state import (
    TechnicalConversationState,
    ConversationStage,
    IntentType,
    ResponseType,
    DomainType
)
from ..monitoring.state_metrics import StateMetricsCollector
from ..telemetry.state_telemetry import StateOperationTracer

logger = logging.getLogger(__name__)


class RoutingDecisionType(str, Enum):
    """Types of routing decisions"""
    DIRECT_ANSWER = "direct_answer"
    INFORMATION_EXTRACTION = "information_extraction"
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    SUFFICIENCY_CHECK = "sufficiency_check"
    PARALLEL_ANALYSIS = "parallel_analysis"
    IMPLEMENTATION = "implementation"
    MULTI_AGENT = "multi_agent"
    CLARIFICATION = "clarification"
    MORE_QUESTIONS = "more_questions"
    MORE_INFO_NEEDED = "more_info_needed"
    ERROR = "error"
    HUMAN_ESCALATION = "human_escalation"
    DIRECT_RESPONSE = "direct_response"
    ANALYSIS = "analysis"


@dataclass
class RoutingDecision:
    """Represents a routing decision with metadata"""
    destination: RoutingDecisionType
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: str
    decision_time_ms: float


@dataclass
class RoutingConfig:
    """Configuration for routing thresholds and parameters"""
    
    # Intent classification thresholds
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.3
    human_escalation_threshold: float = 0.2
    
    # Information completeness thresholds
    sufficient_info_threshold: float = 0.8
    partial_info_threshold: float = 0.5
    minimal_info_threshold: float = 0.2
    
    # Context-based routing parameters
    max_clarification_rounds: int = 3
    max_conversation_turns: int = 10
    parallel_execution_threshold: float = 0.6
    
    # A/B testing parameters
    enable_ab_testing: bool = False
    ab_test_percentage: float = 0.1
    routing_strategy_variant: str = "default"
    
    # Performance thresholds
    max_routing_time_ms: float = 100.0
    enable_routing_cache: bool = True
    cache_ttl_seconds: int = 300


class RoutingEngine:
    """
    Intelligent routing engine for LangGraph workflow decisions.
    
    This engine makes routing decisions based on:
    - Intent classification confidence
    - Conversation history and context
    - Information extraction completeness
    - Domain-specific routing rules
    - Performance considerations
    """
    
    def __init__(
        self,
        domain: DomainType,
        config: Optional[RoutingConfig] = None
    ):
        self.domain = domain
        self.config = config or RoutingConfig()
        
        # Initialize services
        self.metrics_collector = StateMetricsCollector()
        self.telemetry_service = StateOperationTracer()
        
        # Routing cache for performance
        self._routing_cache: Dict[str, Tuple[RoutingDecision, float]] = {}
        
        # Domain-specific routing rules
        self._domain_rules = self._initialize_domain_rules()
        
        logger.info(f"RoutingEngine initialized for domain: {domain}")
        logger.info(f"Configuration: {self.config}")
    
    def _initialize_domain_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific routing rules"""
        
        rules = {
            "cloud": {
                "high_complexity_keywords": ["migration", "disaster recovery", "multi-region", "hybrid"],
                "requires_multi_agent": ["azure", "aws", "migration", "deployment"],
                "direct_answer_patterns": ["pricing", "what is", "difference between"],
                "information_required": ["environment", "scale", "requirements", "constraints"]
            },
            "network": {
                "high_complexity_keywords": ["site-to-site", "vpn", "firewall rules", "load balancing"],
                "requires_multi_agent": ["palo alto", "cisco", "complex topology"],
                "direct_answer_patterns": ["port", "protocol", "default", "command"],
                "information_required": ["topology", "devices", "requirements", "current_config"]
            },
            "devops": {
                "high_complexity_keywords": ["ci/cd", "kubernetes", "terraform", "monitoring"],
                "requires_multi_agent": ["pipeline", "deployment", "infrastructure as code"],
                "direct_answer_patterns": ["command", "syntax", "example", "best practice"],
                "information_required": ["platform", "tools", "requirements", "current_setup"]
            },
            "dev": {
                "high_complexity_keywords": ["architecture", "integration", "performance", "security"],
                "requires_multi_agent": ["api design", "database", "scalability"],
                "direct_answer_patterns": ["syntax", "example", "library", "framework"],
                "information_required": ["language", "framework", "requirements", "constraints"]
            }
        }
        
        domain_str = self.domain.value if hasattr(self.domain, 'value') else str(self.domain)
        return rules.get(domain_str, rules["dev"])  # Default to dev rules
    
    def route_after_intent_classification(self, state: TechnicalConversationState) -> str:
        """Route after intent classification node"""
        
        start_time = time.time()
        
        try:
            # Check for cached routing decision
            cache_key = self._generate_cache_key(state, "intent_classification")
            cached_decision = self._get_cached_decision(cache_key)
            
            if cached_decision:
                self._record_routing_metrics("intent_classification", cached_decision.destination, True)
                return cached_decision.destination.value
            
            # Analyze the current state
            intent = state.get('intent', IntentType.NEW_QUERY)
            confidence = state.get('confidence_score', 0.0)
            current_input = state.get('current_input', '')
            stage = state.get('stage', ConversationStage.INITIAL)
            
            # Make routing decision based on confidence and content
            decision = self._make_intent_routing_decision(
                intent=intent,
                confidence=confidence,
                input_text=current_input,
                stage=stage,
                state=state
            )
            
            # Cache the decision
            self._cache_decision(cache_key, decision)
            
            # Record metrics
            decision_time = (time.time() - start_time) * 1000
            self._record_routing_decision(state, "intent_classification", decision)
            self._record_routing_metrics("intent_classification", decision.destination, False)
            
            logger.info(
                f"Intent routing decision: {decision.destination.value}",
                extra={
                    "conversation_id": state.get('conversation_id'),
                    "intent": intent.value if hasattr(intent, 'value') else str(intent),
                    "confidence": confidence,
                    "reasoning": decision.reasoning,
                    "decision_time_ms": decision_time
                }
            )
            
            return decision.destination.value
            
        except Exception as e:
            logger.error(f"Intent routing failed: {e}", exc_info=True)
            self._record_routing_error(state, "intent_classification", e)
            return RoutingDecisionType.ERROR.value
    
    def route_after_extraction(self, state: TechnicalConversationState) -> str:
        """Route after information extraction node"""
        
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(state, "extraction")
            cached_decision = self._get_cached_decision(cache_key)
            
            if cached_decision:
                return cached_decision.destination.value
            
            # Analyze extracted information
            extracted_info = state.get('extracted_info', {})
            missing_fields = state.get('missing_fields', [])
            accumulated_info = state.get('accumulated_info', {})
            
            # Calculate information completeness
            completeness_score = self._calculate_completeness_score(
                extracted_info, missing_fields, accumulated_info
            )
            
            # Make routing decision
            decision = self._make_extraction_routing_decision(
                completeness_score=completeness_score,
                extracted_info=extracted_info,
                missing_fields=missing_fields,
                state=state
            )
            
            # Cache and record
            self._cache_decision(cache_key, decision)
            self._record_routing_decision(state, "extraction", decision)
            
            logger.info(
                f"Extraction routing decision: {decision.destination.value}",
                extra={
                    "conversation_id": state.get('conversation_id'),
                    "completeness_score": completeness_score,
                    "reasoning": decision.reasoning
                }
            )
            
            return decision.destination.value
            
        except Exception as e:
            logger.error(f"Extraction routing failed: {e}", exc_info=True)
            return RoutingDecisionType.ERROR.value
    
    def route_after_requirement_analysis(self, state: TechnicalConversationState) -> str:
        """Route after requirement analysis node"""
        
        try:
            # Analyze requirement analysis results
            required_fields = state.get('required_fields', [])
            missing_fields = state.get('missing_fields', [])
            accumulated_info = state.get('accumulated_info', {})
            
            # Calculate how much information we still need
            missing_ratio = len(missing_fields) / max(len(required_fields), 1)
            
            if missing_ratio == 0:
                # All requirements satisfied
                decision = RoutingDecision(
                    destination=RoutingDecisionType.IMPLEMENTATION,
                    confidence=0.95,
                    reasoning="All requirements satisfied, ready for implementation",
                    metadata={"missing_ratio": missing_ratio},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            elif missing_ratio < 0.3:
                # Most requirements satisfied, check sufficiency
                decision = RoutingDecision(
                    destination=RoutingDecisionType.SUFFICIENCY_CHECK,
                    confidence=0.8,
                    reasoning="Most requirements satisfied, checking if sufficient",
                    metadata={"missing_ratio": missing_ratio},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            else:
                # Need more information
                decision = RoutingDecision(
                    destination=RoutingDecisionType.MORE_INFO_NEEDED,
                    confidence=0.9,
                    reasoning="Significant information gaps, need more details",
                    metadata={"missing_ratio": missing_ratio, "missing_fields": missing_fields},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            
            self._record_routing_decision(state, "requirement_analysis", decision)
            
            logger.info(
                f"Requirement analysis routing: {decision.destination.value}",
                extra={
                    "conversation_id": state.get('conversation_id'),
                    "missing_ratio": missing_ratio,
                    "reasoning": decision.reasoning
                }
            )
            
            return decision.destination.value
            
        except Exception as e:
            logger.error(f"Requirement analysis routing failed: {e}", exc_info=True)
            return RoutingDecisionType.ERROR.value
    
    def route_after_sufficiency_check(self, state: TechnicalConversationState) -> str:
        """Route after sufficiency checker node"""
        
        try:
            stage = state.get('stage', ConversationStage.INITIAL)
            missing_fields = state.get('missing_fields', [])
            questions_asked = state.get('questions_asked', [])
            
            if stage == ConversationStage.SUFFICIENT:
                # Information is sufficient for implementation
                decision = RoutingDecision(
                    destination=RoutingDecisionType.IMPLEMENTATION,
                    confidence=0.95,
                    reasoning="Information deemed sufficient for implementation",
                    metadata={"stage": stage.value if hasattr(stage, 'value') else str(stage)},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            elif len(questions_asked) >= self.config.max_clarification_rounds:
                # Too many clarification rounds, provide best effort answer
                decision = RoutingDecision(
                    destination=RoutingDecisionType.CLARIFICATION,
                    confidence=0.7,
                    reasoning="Maximum clarification rounds reached, providing best effort answer",
                    metadata={"questions_asked_count": len(questions_asked)},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            else:
                # Need more questions
                decision = RoutingDecision(
                    destination=RoutingDecisionType.MORE_QUESTIONS,
                    confidence=0.8,
                    reasoning="Information insufficient, need clarifying questions",
                    metadata={"missing_fields": missing_fields},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            
            self._record_routing_decision(state, "sufficiency_check", decision)
            
            return decision.destination.value
            
        except Exception as e:
            logger.error(f"Sufficiency check routing failed: {e}", exc_info=True)
            return RoutingDecisionType.ERROR.value
    
    def route_after_supervisor(self, state: TechnicalConversationState) -> str:
        """Route after supervisor node (if enabled)"""
        
        try:
            # Check if supervisor has completed multi-agent coordination
            multi_agent_results = state.get('multi_agent_results', {})
            needs_resolution = state.get('needs_resolution', False)
            
            if needs_resolution:
                # Conflicts need resolution, route to direct answer for clarification
                decision = RoutingDecision(
                    destination=RoutingDecisionType.DIRECT_RESPONSE,
                    confidence=0.8,
                    reasoning="Multi-agent conflicts detected, need clarification",
                    metadata={"conflicts": True},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            elif multi_agent_results:
                # Results available, proceed to implementation
                decision = RoutingDecision(
                    destination=RoutingDecisionType.IMPLEMENTATION,
                    confidence=0.9,
                    reasoning="Multi-agent results available, proceeding to implementation",
                    metadata={"multi_agent_results": len(multi_agent_results)},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            else:
                # Need more analysis
                decision = RoutingDecision(
                    destination=RoutingDecisionType.ANALYSIS,
                    confidence=0.7,
                    reasoning="Supervisor requires more analysis",
                    metadata={},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            
            self._record_routing_decision(state, "supervisor", decision)
            
            return decision.destination.value
            
        except Exception as e:
            logger.error(f"Supervisor routing failed: {e}", exc_info=True)
            return RoutingDecisionType.ERROR.value
    
    def _make_intent_routing_decision(
        self,
        intent: IntentType,
        confidence: float,
        input_text: str,
        stage: ConversationStage,
        state: TechnicalConversationState
    ) -> RoutingDecision:
        """Make routing decision based on intent analysis"""
        
        input_lower = input_text.lower()
        
        # Check for human escalation conditions
        if confidence < self.config.human_escalation_threshold:
            return RoutingDecision(
                destination=RoutingDecisionType.HUMAN_ESCALATION,
                confidence=0.95,
                reasoning=f"Confidence too low ({confidence:.2f}), escalating to human",
                metadata={"original_confidence": confidence},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
        
        # Check for direct answer patterns
        if any(pattern in input_lower for pattern in self._domain_rules["direct_answer_patterns"]):
            if confidence > self.config.high_confidence_threshold:
                return RoutingDecision(
                    destination=RoutingDecisionType.DIRECT_ANSWER,
                    confidence=confidence,
                    reasoning="High confidence direct answer pattern detected",
                    metadata={"pattern_match": True},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
        
        # Check for multi-agent requirements
        if any(keyword in input_lower for keyword in self._domain_rules["requires_multi_agent"]):
            return RoutingDecision(
                destination=RoutingDecisionType.MULTI_AGENT,
                confidence=0.8,
                reasoning="Multi-agent coordination required for complex query",
                metadata={"complexity_indicators": self._domain_rules["requires_multi_agent"]},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
        
        # Check for high complexity indicators
        if any(keyword in input_lower for keyword in self._domain_rules["high_complexity_keywords"]):
            return RoutingDecision(
                destination=RoutingDecisionType.INFORMATION_EXTRACTION,
                confidence=0.85,
                reasoning="High complexity query requires information extraction",
                metadata={"complexity_level": "high"},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
        
        # Default routing based on intent and confidence
        if intent == IntentType.NEW_QUERY:
            if confidence > self.config.high_confidence_threshold:
                return RoutingDecision(
                    destination=RoutingDecisionType.INFORMATION_EXTRACTION,
                    confidence=confidence,
                    reasoning="High confidence new query, extracting information",
                    metadata={"intent": intent.value if hasattr(intent, 'value') else str(intent)},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
            else:
                return RoutingDecision(
                    destination=RoutingDecisionType.DIRECT_ANSWER,
                    confidence=confidence,
                    reasoning="Lower confidence new query, providing direct answer",
                    metadata={"intent": intent.value if hasattr(intent, 'value') else str(intent)},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    decision_time_ms=0.0
                )
        elif intent == IntentType.PROVIDING_INFO:
            return RoutingDecision(
                destination=RoutingDecisionType.INFORMATION_EXTRACTION,
                confidence=confidence,
                reasoning="User providing information, extracting data",
                metadata={"intent": intent.value if hasattr(intent, 'value') else str(intent)},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
        else:
            return RoutingDecision(
                destination=RoutingDecisionType.DIRECT_ANSWER,
                confidence=confidence,
                reasoning="Default routing to direct answer",
                metadata={"intent": intent.value if hasattr(intent, 'value') else str(intent)},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
    
    def _make_extraction_routing_decision(
        self,
        completeness_score: float,
        extracted_info: Dict[str, Any],
        missing_fields: List[str],
        state: TechnicalConversationState
    ) -> RoutingDecision:
        """Make routing decision based on information extraction results"""
        
        if completeness_score >= self.config.sufficient_info_threshold:
            return RoutingDecision(
                destination=RoutingDecisionType.IMPLEMENTATION,
                confidence=0.9,
                reasoning="Information extraction complete, ready for implementation",
                metadata={"completeness_score": completeness_score},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
        elif completeness_score >= self.config.parallel_execution_threshold:
            # Consider parallel analysis for partial information
            return RoutingDecision(
                destination=RoutingDecisionType.PARALLEL_ANALYSIS,
                confidence=0.8,
                reasoning="Partial information available, starting parallel analysis",
                metadata={"completeness_score": completeness_score},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
        elif completeness_score >= self.config.partial_info_threshold:
            return RoutingDecision(
                destination=RoutingDecisionType.REQUIREMENT_ANALYSIS,
                confidence=0.75,
                reasoning="Partial information available, analyzing requirements",
                metadata={"completeness_score": completeness_score},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
        else:
            return RoutingDecision(
                destination=RoutingDecisionType.SUFFICIENCY_CHECK,
                confidence=0.7,
                reasoning="Limited information extracted, checking sufficiency",
                metadata={"completeness_score": completeness_score, "missing_fields": missing_fields},
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_time_ms=0.0
            )
    
    def _calculate_completeness_score(
        self,
        extracted_info: Dict[str, Any],
        missing_fields: List[str],
        accumulated_info: Dict[str, Any]
    ) -> float:
        """Calculate information completeness score"""
        
        # Combine extracted and accumulated information
        total_info = {**accumulated_info, **extracted_info}
        
        # Count non-empty values
        filled_fields = sum(1 for v in total_info.values() if v and str(v).strip())
        
        # Calculate based on domain requirements
        required_fields = self._domain_rules["information_required"]
        required_count = len(required_fields)
        
        if required_count == 0:
            return 1.0
        
        # Calculate score based on filled vs required fields
        base_score = min(filled_fields / required_count, 1.0)
        
        # Penalty for missing critical fields
        missing_penalty = len(missing_fields) * 0.1
        
        # Bonus for extra relevant information
        extra_bonus = max(0, (filled_fields - required_count)) * 0.05
        
        score = max(0.0, min(1.0, base_score - missing_penalty + extra_bonus))
        
        return score
    
    def _generate_cache_key(self, state: TechnicalConversationState, routing_point: str) -> str:
        """Generate cache key for routing decision"""
        
        key_components = [
            routing_point,
            str(state.get('conversation_id', '')),
            str(state.get('intent', '')),
            f"{state.get('confidence_score', 0.0):.2f}",
            str(len(state.get('extracted_info', {}))),
            str(len(state.get('missing_fields', []))),
            str(state.get('stage', ''))
        ]
        
        return "|".join(key_components)
    
    def _get_cached_decision(self, cache_key: str) -> Optional[RoutingDecision]:
        """Get cached routing decision if still valid"""
        
        if not self.config.enable_routing_cache:
            return None
        
        cached_data = self._routing_cache.get(cache_key)
        if cached_data:
            decision, timestamp = cached_data
            if time.time() - timestamp < self.config.cache_ttl_seconds:
                return decision
            else:
                # Remove expired entry
                del self._routing_cache[cache_key]
        
        return None
    
    def _cache_decision(self, cache_key: str, decision: RoutingDecision) -> None:
        """Cache routing decision"""
        
        if self.config.enable_routing_cache:
            self._routing_cache[cache_key] = (decision, time.time())
    
    def _record_routing_decision(
        self,
        state: TechnicalConversationState,
        routing_point: str,
        decision: RoutingDecision
    ) -> None:
        """Record routing decision for telemetry"""
        
        try:
            self.telemetry_service.record_routing_decision(
                conversation_id=state.get('conversation_id'),
                routing_point=routing_point,
                decision=decision.destination.value,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                metadata=decision.metadata
            )
        except Exception as e:
            logger.warning(f"Failed to record routing decision: {e}")
    
    def _record_routing_metrics(
        self,
        routing_point: str,
        destination: RoutingDecisionType,
        cache_hit: bool
    ) -> None:
        """Record routing metrics"""
        
        try:
            self.metrics_collector.increment_counter(
                f"routing_decision_total",
                domain=str(self.domain),
                routing_point=routing_point,
                destination=destination.value,
                cache_hit=str(cache_hit)
            )
        except Exception as e:
            logger.warning(f"Failed to record routing metrics: {e}")
    
    def _record_routing_error(
        self,
        state: TechnicalConversationState,
        routing_point: str,
        error: Exception
    ) -> None:
        """Record routing error"""
        
        try:
            self.metrics_collector.increment_counter(
                f"routing_error_total",
                domain=str(self.domain),
                routing_point=routing_point,
                error_type=type(error).__name__
            )
        except Exception as e:
            logger.warning(f"Failed to record routing error: {e}")
    
    def get_routing_config(self) -> Dict[str, Any]:
        """Get current routing configuration"""
        
        return {
            "domain": str(self.domain),
            "thresholds": {
                "high_confidence": self.config.high_confidence_threshold,
                "low_confidence": self.config.low_confidence_threshold,
                "human_escalation": self.config.human_escalation_threshold,
                "sufficient_info": self.config.sufficient_info_threshold,
                "partial_info": self.config.partial_info_threshold,
                "parallel_execution": self.config.parallel_execution_threshold
            },
            "limits": {
                "max_clarification_rounds": self.config.max_clarification_rounds,
                "max_conversation_turns": self.config.max_conversation_turns
            },
            "domain_rules": self._domain_rules,
            "cache_enabled": self.config.enable_routing_cache,
            "ab_testing_enabled": self.config.enable_ab_testing
        }
    
    def update_config(self, new_config: RoutingConfig) -> None:
        """Update routing configuration for hot reloading"""
        
        self.config = new_config
        # Clear cache when config changes
        self._routing_cache.clear()
        
        logger.info(f"Routing configuration updated for domain: {self.domain}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        
        return {
            "cache_size": len(self._routing_cache),
            "domain": str(self.domain),
            "config": self.get_routing_config()
        }