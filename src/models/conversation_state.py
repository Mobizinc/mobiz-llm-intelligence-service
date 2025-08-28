"""
LangGraph Conversation State Schema for Technical Bots

This module defines the comprehensive state schema for LangGraph-based 
technical conversations, including validation, serialization, and type safety.

Part of Epic 1: Core State Management & Infrastructure
Story 1.1: Conversation State Schema Design
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal, Union
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


class ConversationStage(str, Enum):
    """Enumeration of conversation stages"""
    INITIAL = "initial"           # First query in conversation
    GATHERING = "gathering"       # Collecting more information
    ANALYZING = "analyzing"       # Processing collected information
    SUFFICIENT = "sufficient"     # Enough info to provide implementation
    ERROR = "error"              # Error state requiring intervention


class IntentType(str, Enum):
    """Enumeration of conversation intent types"""
    NEW_QUERY = "new_query"              # Brand new technical query
    PROVIDING_INFO = "providing_info"    # User providing requested information
    ASKING_FOLLOWUP = "asking_followup"  # Follow-up question to previous response
    CLARIFICATION = "clarification"      # Asking for clarification
    IMPLEMENTATION = "implementation"     # Requesting implementation details


class ResponseType(str, Enum):
    """Enumeration of response types"""
    QUESTION = "question"            # Bot asking for more information
    IMPLEMENTATION = "implementation" # Bot providing technical solution
    ERROR = "error"                  # Error response
    CLARIFICATION = "clarification"  # Clarification response
    ACKNOWLEDGMENT = "acknowledgment" # Simple acknowledgment


class DomainType(str, Enum):
    """Technical domains supported by the system"""
    CLOUD = "cloud"
    NETWORK = "network"
    DEVOPS = "devops"
    DEV = "dev"


class TechnicalConversationState(TypedDict):
    """
    Comprehensive state schema for LangGraph-based technical conversations.
    
    This TypedDict defines all necessary fields for tracking conversation state
    throughout the LangGraph workflow, ensuring type safety and validation.
    
    Required fields are marked as such, optional fields use NotRequired.
    """
    
    # === CORE CONVERSATION DATA ===
    conversation_id: str                    # Unique conversation identifier
    initial_query: str                     # Original user query
    current_input: str                     # Current user input being processed
    domain: DomainType                     # Technical domain (cloud, network, etc.)
    thread_id: str                        # Slack thread identifier
    
    # === INTENT & CLASSIFICATION ===
    intent: IntentType                     # Classified intent of current input
    confidence_score: float                # Confidence in intent classification (0.0-1.0)
    intent_metadata: NotRequired[Dict[str, Any]]  # Additional intent analysis data
    
    # === INFORMATION MANAGEMENT ===
    extracted_info: Dict[str, Any]         # Information extracted from current input
    accumulated_info: Dict[str, Any]       # All information gathered so far
    required_fields: List[str]             # Fields still needed for solution
    missing_fields: List[str]              # Specifically missing required fields
    validation_errors: NotRequired[List[str]]  # Field validation errors
    
    # === CONVERSATION FLOW CONTROL ===
    stage: ConversationStage               # Current conversation stage
    questions_asked: List[str]             # Questions asked to user so far
    response: str                          # Generated response to user
    response_type: ResponseType            # Type of response being provided
    
    # === CONTEXTUAL DATA ===
    user_id: str                          # Slack user ID
    channel_id: str                       # Slack channel ID
    correlation_id: str                   # Request correlation ID for tracing
    conversation_history: NotRequired[List[Dict[str, Any]]]  # Previous conversation messages
    
    # === TIMING & VERSIONING ===
    created_at: str                       # ISO timestamp when conversation started
    updated_at: str                       # ISO timestamp of last update
    version: int                          # State version for schema evolution
    expires_at: NotRequired[str]          # Optional expiration timestamp
    
    # === METADATA & DEBUGGING ===
    metadata: NotRequired[Dict[str, Any]] # Additional metadata for debugging
    node_history: NotRequired[List[str]]  # LangGraph nodes visited
    error_history: NotRequired[List[str]] # Errors encountered during processing
    
    # === PERFORMANCE & MONITORING ===
    processing_times: NotRequired[Dict[str, float]]  # Node processing times
    token_usage: NotRequired[Dict[str, int]]         # Token consumption tracking
    cache_hits: NotRequired[List[str]]               # Cache hit tracking


class TechnicalConversationStateValidator(BaseModel):
    """
    Pydantic model for validating TechnicalConversationState.
    
    Provides comprehensive validation, serialization, and type checking
    for conversation state data.
    """
    
    # Core fields with validation
    conversation_id: str = Field(..., min_length=1, max_length=100)
    initial_query: str = Field(..., min_length=1, max_length=10000)
    current_input: str = Field(..., min_length=1, max_length=10000)
    domain: DomainType
    thread_id: str = Field(..., min_length=1, max_length=100)
    
    # Intent and classification
    intent: IntentType
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    intent_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Information management
    extracted_info: Dict[str, Any] = Field(default_factory=dict)
    accumulated_info: Dict[str, Any] = Field(default_factory=dict)
    required_fields: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    validation_errors: Optional[List[str]] = Field(default_factory=list)
    
    # Flow control
    stage: ConversationStage
    questions_asked: List[str] = Field(default_factory=list)
    response: str = Field(default="")
    response_type: ResponseType
    
    # Context
    user_id: str = Field(..., min_length=1, max_length=50)
    channel_id: str = Field(..., min_length=1, max_length=50)
    correlation_id: str = Field(..., min_length=1, max_length=100)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    
    # Timing and versioning
    created_at: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*')
    updated_at: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*')
    version: int = Field(default=1, ge=1)
    expires_at: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*')
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    node_history: Optional[List[str]] = Field(default_factory=list)
    error_history: Optional[List[str]] = Field(default_factory=list)
    
    # Performance
    processing_times: Optional[Dict[str, float]] = Field(default_factory=dict)
    token_usage: Optional[Dict[str, int]] = Field(default_factory=dict)
    cache_hits: Optional[List[str]] = Field(default_factory=list)
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"  # Prevent additional fields
    )
    
    @field_validator('questions_asked')
    @classmethod
    def validate_questions_asked(cls, v):
        """Validate that questions_asked contains valid questions"""
        if len(v) > 20:  # Reasonable limit
            raise ValueError("Too many questions asked (max 20)")
        return v
    
    @field_validator('response')
    @classmethod
    def validate_response_length(cls, v):
        """Validate response length"""
        if len(v) > 50000:  # Reasonable limit for Slack
            raise ValueError("Response too long (max 50000 characters)")
        return v
    
    @field_validator('missing_fields')
    @classmethod
    def validate_missing_fields_subset(cls, v, info):
        """Ensure missing_fields is subset of required_fields"""
        if info.data:
            required_fields = info.data.get('required_fields', [])
            if not all(field in required_fields for field in v):
                raise ValueError("missing_fields must be subset of required_fields")
        return v
    
    @model_validator(mode='after')
    def validate_stage_consistency(self):
        """Validate that stage is consistent with other fields"""
        if self.stage == ConversationStage.SUFFICIENT and self.missing_fields:
            raise ValueError("Stage 'sufficient' cannot have missing_fields")
        
        if self.stage == ConversationStage.GATHERING and not self.missing_fields:
            raise ValueError("Stage 'gathering' should have missing_fields")
        
        return self
    
    def to_typeddict(self) -> TechnicalConversationState:
        """Convert to TypedDict format for LangGraph compatibility"""
        data = self.model_dump()
        
        # Remove None values for NotRequired fields
        cleaned_data = {k: v for k, v in data.items() if v is not None}
        
        # Ensure required fields are present
        required_keys = {
            'conversation_id', 'initial_query', 'current_input', 'domain', 
            'thread_id', 'intent', 'confidence_score', 'extracted_info',
            'accumulated_info', 'required_fields', 'missing_fields', 'stage',
            'questions_asked', 'response', 'response_type', 'user_id',
            'channel_id', 'correlation_id', 'created_at', 'updated_at', 'version'
        }
        
        for key in required_keys:
            if key not in cleaned_data:
                raise ValueError(f"Required field '{key}' missing from state")
        
        return cleaned_data  # type: ignore


class ConversationStateManager:
    """
    Manager class for conversation state operations.
    
    Provides high-level methods for creating, validating, and managing
    conversation state throughout the LangGraph workflow.
    """
    
    @staticmethod
    def create_initial_state(
        conversation_id: str,
        initial_query: str,
        domain: DomainType,
        thread_id: str,
        user_id: str,
        channel_id: str,
        correlation_id: str,
        intent: IntentType = IntentType.NEW_QUERY,
        confidence_score: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TechnicalConversationState:
        """
        Create initial conversation state for new conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            initial_query: User's initial query
            domain: Technical domain for conversation
            thread_id: Slack thread identifier
            user_id: Slack user ID
            channel_id: Slack channel ID
            correlation_id: Request correlation ID
            intent: Initial intent classification
            confidence_score: Confidence in intent classification
            metadata: Additional metadata
            
        Returns:
            Validated TechnicalConversationState ready for processing
        """
        now = datetime.now(timezone.utc).isoformat()
        
        state_data = {
            'conversation_id': conversation_id,
            'initial_query': initial_query,
            'current_input': initial_query,
            'domain': domain,
            'thread_id': thread_id,
            'intent': intent,
            'confidence_score': confidence_score,
            'extracted_info': {},
            'accumulated_info': {},
            'required_fields': [],
            'missing_fields': [],
            'stage': ConversationStage.INITIAL,
            'questions_asked': [],
            'response': "",
            'response_type': ResponseType.QUESTION,
            'user_id': user_id,
            'channel_id': channel_id,
            'correlation_id': correlation_id,
            'created_at': now,
            'updated_at': now,
            'version': 1,
            'metadata': metadata or {},
            'node_history': [],
            'error_history': []
        }
        
        # Validate using Pydantic model
        validator = TechnicalConversationStateValidator(**state_data)
        return validator.to_typeddict()
    
    @staticmethod
    def validate_state(state: TechnicalConversationState) -> bool:
        """
        Validate conversation state using Pydantic model.
        
        Args:
            state: State to validate
            
        Returns:
            True if valid, raises ValidationError if invalid
        """
        validator = TechnicalConversationStateValidator(**state)
        return True
    
    @staticmethod
    def update_state(
        state: TechnicalConversationState,
        updates: Dict[str, Any],
        increment_version: bool = True
    ) -> TechnicalConversationState:
        """
        Update conversation state with new data.
        
        Args:
            state: Current state
            updates: Updates to apply
            increment_version: Whether to increment version number
            
        Returns:
            Updated and validated state
        """
        # Create updated state
        updated_state = state.copy()
        updated_state.update(updates)
        
        # Update timestamp and version
        updated_state['updated_at'] = datetime.now(timezone.utc).isoformat()
        if increment_version:
            updated_state['version'] = updated_state.get('version', 1) + 1
        
        # Validate updated state
        validator = TechnicalConversationStateValidator(**updated_state)
        return validator.to_typeddict()
    
    @staticmethod
    def serialize_state(state: TechnicalConversationState) -> str:
        """
        Serialize state to JSON string for storage.
        
        Args:
            state: State to serialize
            
        Returns:
            JSON string representation
        """
        return json.dumps(state, default=str, sort_keys=True)
    
    @staticmethod
    def deserialize_state(json_str: str) -> TechnicalConversationState:
        """
        Deserialize state from JSON string.
        
        Args:
            json_str: JSON string to deserialize
            
        Returns:
            Validated TechnicalConversationState
        """
        data = json.loads(json_str)
        validator = TechnicalConversationStateValidator(**data)
        return validator.to_typeddict()
    
    @staticmethod
    def get_state_hash(state: TechnicalConversationState) -> str:
        """
        Generate hash of state for change detection.
        
        Args:
            state: State to hash
            
        Returns:
            SHA256 hash of serialized state
        """
        serialized = ConversationStateManager.serialize_state(state)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    @staticmethod
    def add_node_to_history(
        state: TechnicalConversationState,
        node_name: str
    ) -> TechnicalConversationState:
        """
        Add node to processing history.
        
        Args:
            state: Current state
            node_name: Name of node being processed
            
        Returns:
            Updated state with node added to history
        """
        updated_state = state.copy()
        node_history = updated_state.get('node_history', []).copy()
        node_history.append(f"{node_name}:{datetime.now(timezone.utc).isoformat()}")
        updated_state['node_history'] = node_history
        return updated_state
    
    @staticmethod
    def add_error_to_history(
        state: TechnicalConversationState,
        error_message: str
    ) -> TechnicalConversationState:
        """
        Add error to error history.
        
        Args:
            state: Current state
            error_message: Error message to add
            
        Returns:
            Updated state with error added to history
        """
        updated_state = state.copy()
        error_history = updated_state.get('error_history', []).copy()
        error_history.append(f"{datetime.now(timezone.utc).isoformat()}: {error_message}")
        updated_state['error_history'] = error_history
        return updated_state


# Utility functions for common state operations
def is_conversation_complete(state: TechnicalConversationState) -> bool:
    """Check if conversation has enough information to provide solution"""
    return (
        state['stage'] == ConversationStage.SUFFICIENT and
        len(state['missing_fields']) == 0 and
        len(state['response']) > 0
    )


def get_conversation_age_minutes(state: TechnicalConversationState) -> float:
    """Get conversation age in minutes"""
    created = datetime.fromisoformat(state['created_at'].replace('Z', '+00:00'))
    now = datetime.now(timezone.utc)
    return (now - created).total_seconds() / 60


def should_expire_conversation(state: TechnicalConversationState, max_age_hours: int = 24) -> bool:
    """Check if conversation should be expired based on age"""
    age_minutes = get_conversation_age_minutes(state)
    return age_minutes > (max_age_hours * 60)


# Export main types and classes
__all__ = [
    'TechnicalConversationState',
    'TechnicalConversationStateValidator', 
    'ConversationStateManager',
    'ConversationStage',
    'IntentType',
    'ResponseType',
    'DomainType',
    'is_conversation_complete',
    'get_conversation_age_minutes',
    'should_expire_conversation'
]