from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class SlackCommandRequest(BaseModel):
    """Pydantic model for Slack slash command requests"""
    token: Optional[str] = None
    team_id: Optional[str] = None
    team_domain: Optional[str] = None
    channel_id: str
    channel_name: Optional[str] = None
    user_id: str
    user_name: Optional[str] = None
    command: Optional[str] = None
    text: str = ""
    response_url: str
    trigger_id: Optional[str] = None
    api_app_id: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields from Slack


class SlackResponse(BaseModel):
    """Pydantic model for Slack responses"""
    text: str
    response_type: Optional[str] = "in_channel"  # "in_channel" or "ephemeral"
    blocks: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    
    @validator('response_type')
    def validate_response_type(cls, v):
        if v not in ["in_channel", "ephemeral"]:
            return "in_channel"
        return v


class SlackEventRequest(BaseModel):
    """Pydantic model for Slack Events API requests"""
    token: Optional[str] = None
    team_id: Optional[str] = None
    api_app_id: Optional[str] = None
    event: Dict[str, Any]
    type: str
    event_id: Optional[str] = None
    event_time: Optional[int] = None
    authed_users: Optional[List[str]] = None
    challenge: Optional[str] = None  # For URL verification


class SlackEventResponse(BaseModel):
    """Pydantic model for Slack Events API responses"""
    ok: bool = True
    challenge: Optional[str] = None  # For URL verification response


class BotProcessingRequest(BaseModel):
    """Internal model for bot processing requests"""
    domain: str
    emoji: str
    user_query: str
    channel_context: Dict[str, Any]
    thread_history: List[Dict[str, Any]] = []
    response_url: str
    correlation_id: str
    channel_id: Optional[str] = None
    user_id: Optional[str] = None
    initial_message_ts: Optional[str] = None


class BotResponse(BaseModel):
    """Model for bot AI responses"""
    text: str
    domain: str
    correlation_id: str
    processing_time: Optional[float] = None
    token_usage: Optional[int] = None
    
    
class HealthCheckResponse(BaseModel):
    """Model for health check responses"""
    status: str
    app_name: str
    version: str
    timestamp: float


class ReadinessCheckResponse(BaseModel):
    """Model for readiness check responses"""
    status: str
    checks: Dict[str, bool]
    timestamp: float