"""
Pydantic models for interface-agnostic conversation data contracts.
These models serve as generic, internal representations that work across both Slack and web interfaces.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class ConversationMessage(BaseModel):
    """Individual message within a conversation"""
    role: Literal['user', 'assistant', 'system'] = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationRequest(BaseModel):
    """Request model for conversation processing"""
    messages: List[ConversationMessage] = Field(..., description="List of conversation messages")
    client_id: Optional[str] = Field(default=None, description="Client identifier for context loading")
    user_id: str = Field(..., description="User identifier")
    conversation_id: Optional[str] = Field(default=None, description="Conversation identifier for threading")
    channel_id: Optional[str] = Field(default=None, description="Channel identifier for Slack integration")
    bot_type: Optional[str] = Field(default="general", description="Type of bot (cloud, devops, network, dev)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")


class ConversationResponse(BaseModel):
    """Response model for conversation processing"""
    content: str = Field(..., description="Response content")
    conversation_id: str = Field(..., description="Conversation identifier")
    status: str = Field(default="success", description="Response status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")


class ConversationContext(BaseModel):
    """Context information for conversation processing"""
    client_context: Optional[Dict[str, Any]] = Field(default=None, description="Client-specific context")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="User-specific context") 
    conversation_history: List[ConversationMessage] = Field(default_factory=list, description="Previous messages")
    channel_context: Optional[Dict[str, Any]] = Field(default=None, description="Channel-specific context")