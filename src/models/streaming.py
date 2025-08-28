"""
Pydantic models for structured JSON messages sent over streams.
Critical for Vercel AI SDK integration and real-time communication.
"""

from typing import Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime


class StreamTextChunk(BaseModel):
    """Text content chunk for streaming responses"""
    type: Literal["text"] = "text"
    content: str = Field(..., description="Text content chunk")
    delta: Optional[str] = Field(default=None, description="Delta content for progressive updates")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    timestamp: datetime = Field(default_factory=datetime.now)


class StreamUIComponent(BaseModel):
    """UI component message for structured streaming updates"""
    type: Literal["ui"] = "ui"
    component: str = Field(..., description="Component name to render")
    props: Dict[str, Any] = Field(default_factory=dict, description="Component properties")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Component metadata")
    timestamp: datetime = Field(default_factory=datetime.now)


class StreamStatusUpdate(BaseModel):
    """Progress and status update message"""
    type: Literal["status"] = "status"
    status: str = Field(..., description="Current operation status")
    progress: Optional[float] = Field(default=None, description="Progress percentage (0-100)")
    operation: str = Field(..., description="Current operation description")
    details: Optional[str] = Field(default=None, description="Additional status details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Status metadata")
    timestamp: datetime = Field(default_factory=datetime.now)


class StreamError(BaseModel):
    """Error message for streaming"""
    type: Literal["error"] = "error"
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)


class StreamComplete(BaseModel):
    """Stream completion signal"""
    type: Literal["complete"] = "complete"
    final_response: Optional[str] = Field(default=None, description="Final response content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Completion metadata")
    timestamp: datetime = Field(default_factory=datetime.now)


# Union type for all possible streaming message types
StreamMessage = Union[
    StreamTextChunk,
    StreamUIComponent, 
    StreamStatusUpdate,
    StreamError,
    StreamComplete
]


class StreamConfig(BaseModel):
    """Configuration for streaming behavior"""
    enable_status_updates: bool = Field(default=True, description="Enable progress status updates")
    enable_ui_components: bool = Field(default=True, description="Enable UI component messages")
    chunk_size: int = Field(default=100, description="Text chunk size for streaming")
    max_stream_duration: int = Field(default=300, description="Max stream duration in seconds")
    buffer_timeout: float = Field(default=0.1, description="Buffer timeout between chunks")