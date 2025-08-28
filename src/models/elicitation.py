"""
Elicitation data models for MCP interactive parameter collection.
"""
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ElicitationRequest:
    """Represents an elicitation request stored in Azure Table Storage"""
    elicitation_id: str
    message: str
    response_type: str
    context: Dict[str, Any]
    created_at: str
    expires_at: str
    status: str  # pending, completed, timeout, cancelled
    response_data: Optional[Dict[str, Any]] = None
    
    def to_entity(self) -> Dict[str, Any]:
        """Convert to Azure Table Storage entity"""
        entity = {
            'PartitionKey': 'elicitations',
            'RowKey': self.elicitation_id,
            'elicitation_id': self.elicitation_id,
            'message': self.message,
            'response_type': self.response_type,
            'context': json.dumps(self.context),
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'status': self.status,
            'response_data': json.dumps(self.response_data) if self.response_data else ''
        }
        return entity
    
    @classmethod
    def from_entity(cls, entity: Dict[str, Any]) -> 'ElicitationRequest':
        """Create from Azure Table Storage entity"""
        try:
            context = json.loads(entity.get('context', '{}')) if entity.get('context') else {}
        except (json.JSONDecodeError, TypeError):
            context = {}
        
        try:
            response_data = json.loads(entity.get('response_data', '{}')) if entity.get('response_data') else None
        except (json.JSONDecodeError, TypeError):
            response_data = None
        
        return cls(
            elicitation_id=entity.get('elicitation_id', ''),
            message=entity.get('message', ''),
            response_type=entity.get('response_type', ''),
            context=context,
            created_at=entity.get('created_at', ''),
            expires_at=entity.get('expires_at', ''),
            status=entity.get('status', 'pending'),
            response_data=response_data
        )
    
    @classmethod
    def create_new(cls, elicitation_id: str, message: str, response_type: type, 
                   context: Dict[str, Any], timeout_seconds: int = 60) -> 'ElicitationRequest':
        """Create a new elicitation request"""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=timeout_seconds)
        
        return cls(
            elicitation_id=elicitation_id,
            message=message,
            response_type=str(response_type),
            context=context,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            status='pending'
        )
    
    def is_expired(self) -> bool:
        """Check if the elicitation request has expired"""
        try:
            expires_dt = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
            return datetime.now(timezone.utc) > expires_dt
        except:
            return True
    
    def complete(self, response_data: Dict[str, Any]) -> None:
        """Mark the elicitation as completed with response data"""
        self.status = 'completed'
        self.response_data = response_data
    
    def timeout(self) -> None:
        """Mark the elicitation as timed out"""
        self.status = 'timeout'
    
    def cancel(self) -> None:
        """Mark the elicitation as cancelled"""
        self.status = 'cancelled'


class ElicitationWaiter:
    """Helper class for waiting on elicitation responses"""
    
    def __init__(self):
        self._waiters: Dict[str, asyncio.Event] = {}
        self._responses: Dict[str, Dict[str, Any]] = {}
    
    def create_waiter(self, elicitation_id: str) -> None:
        """Create a waiter for the given elicitation ID"""
        self._waiters[elicitation_id] = asyncio.Event()
    
    async def wait_for_response(self, elicitation_id: str, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """Wait for a response to the elicitation"""
        if elicitation_id not in self._waiters:
            return None
        
        try:
            await asyncio.wait_for(self._waiters[elicitation_id].wait(), timeout=timeout)
            return self._responses.get(elicitation_id)
        except asyncio.TimeoutError:
            return None
        finally:
            # Clean up
            self._waiters.pop(elicitation_id, None)
            self._responses.pop(elicitation_id, None)
    
    def complete_elicitation(self, elicitation_id: str, response_data: Dict[str, Any]) -> None:
        """Complete an elicitation with response data"""
        self._responses[elicitation_id] = response_data
        if elicitation_id in self._waiters:
            self._waiters[elicitation_id].set()
    
    def cancel_elicitation(self, elicitation_id: str) -> None:
        """Cancel an elicitation"""
        if elicitation_id in self._waiters:
            self._waiters[elicitation_id].set()
        self._waiters.pop(elicitation_id, None)
        self._responses.pop(elicitation_id, None)