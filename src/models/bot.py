from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from enum import Enum


class BotDomain(str, Enum):
    """Enumeration of available bot domains"""
    CLOUD = "cloud"
    DEVOPS = "devops"
    DEV = "dev"
    NETWORK = "network"


class BotHelpResponse(BaseModel):
    """Model for bot help responses"""
    domain: str
    emoji: str
    title: str
    description: str
    capabilities: List[str]
    examples: List[str]


class ConversationContext(BaseModel):
    """Model for conversation context analysis"""
    is_follow_up: bool = False
    confidence: float = 0.0
    context_messages_count: int = 0
    formatted_context: str = "No previous conversation context available."
    conversation_history: List[Dict[str, Any]] = []


class ChannelContext(BaseModel):
    """Model for Slack channel context"""
    id: str
    name: str = "unknown"
    is_technical: bool = False
    is_general: bool = False
    user_id: Optional[str] = None


class AIProcessingMetrics(BaseModel):
    """Model for AI processing metrics"""
    domain: str
    correlation_id: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    token_usage: Optional[int] = None
    success: bool = False
    error_message: Optional[str] = None


class BackgroundTaskStatus(BaseModel):
    """Model for background task status tracking"""
    task_id: str
    domain: str
    correlation_id: str
    status: str  # "started", "processing", "completed", "failed"
    start_time: float
    end_time: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None