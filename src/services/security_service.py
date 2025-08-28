"""
Enhanced Security Service for Phase 1.
Provides PII detection, audit logging, rate limiting, and data encryption.
"""

import re
import hashlib
import hmac
import json
import time
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC as PBKDF2
import base64
import os

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Represents an audit event for security tracking."""
    timestamp: datetime
    event_type: str
    user_id: str
    client_name: Optional[str]
    channel_id: str
    action: str
    resource: str
    result: str
    metadata: Dict[str, Any]
    correlation_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PIIDetector:
    """Detects and masks PII in text."""
    
    # PII patterns
    PATTERNS = {
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'),
        'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        'aws_key': re.compile(r'AKIA[0-9A-Z]{16}'),
        'azure_key': re.compile(r'[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}'),
        'api_key': re.compile(r'[a-zA-Z0-9]{32,}'),
        'password': re.compile(r'(?i)(password|passwd|pwd|secret|token)[\s:=]+[\S]+'),
        'medical_record': re.compile(r'MRN[:\s]?\d{6,}'),
        'account_number': re.compile(r'\b\d{8,12}\b'),
        'routing_number': re.compile(r'\b\d{9}\b'),
        'driver_license': re.compile(r'\b[A-Z]\d{7,8}\b'),
    }
    
    # Health-specific PII for HIPAA compliance
    HEALTH_PATTERNS = {
        'patient_name': re.compile(r'(?i)(patient|member)[\s:]+[A-Z][a-z]+\s+[A-Z][a-z]+'),
        'dob': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
        'medical_code': re.compile(r'(?i)(ICD-10|CPT|DRG)[:\s]?[A-Z0-9]{3,}'),
    }
    
    def __init__(self, enable_health_detection: bool = False):
        self.enable_health_detection = enable_health_detection
        self._detection_stats = defaultdict(int)
    
    def detect_pii(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Detect PII in text.
        
        Returns:
            List of (pii_type, matched_text, start_pos, end_pos)
        """
        detections = []
        
        # Standard PII detection
        for pii_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                detections.append((
                    pii_type,
                    match.group(),
                    match.start(),
                    match.end()
                ))
                self._detection_stats[pii_type] += 1
        
        # Health-specific PII if enabled
        if self.enable_health_detection:
            for pii_type, pattern in self.HEALTH_PATTERNS.items():
                for match in pattern.finditer(text):
                    detections.append((
                        f"health_{pii_type}",
                        match.group(),
                        match.start(),
                        match.end()
                    ))
                    self._detection_stats[f"health_{pii_type}"] += 1
        
        return detections
    
    def mask_pii(self, text: str, mask_char: str = "*") -> Tuple[str, List[Dict]]:
        """
        Mask PII in text.
        
        Returns:
            Tuple of (masked_text, list_of_detections)
        """
        detections = self.detect_pii(text)
        
        if not detections:
            return text, []
        
        # Sort by position (reverse) to mask from end to start
        detections.sort(key=lambda x: x[2], reverse=True)
        
        masked_text = text
        detection_records = []
        
        for pii_type, matched_text, start, end in detections:
            # Create mask based on PII type
            if pii_type in ['ssn', 'credit_card', 'account_number']:
                # Show last 4 digits
                if len(matched_text) > 4:
                    mask = mask_char * (len(matched_text) - 4) + matched_text[-4:]
                else:
                    mask = mask_char * len(matched_text)
            elif pii_type == 'email':
                # Show domain
                parts = matched_text.split('@')
                if len(parts) == 2:
                    mask = mask_char * 3 + '@' + parts[1]
                else:
                    mask = mask_char * len(matched_text)
            else:
                # Full mask
                mask = mask_char * len(matched_text)
            
            masked_text = masked_text[:start] + mask + masked_text[end:]
            
            detection_records.append({
                'type': pii_type,
                'position': start,
                'length': end - start,
                'masked': True
            })
        
        return masked_text, detection_records
    
    def get_statistics(self) -> Dict[str, int]:
        """Get PII detection statistics."""
        return dict(self._detection_stats)


class RateLimiter:
    """Rate limiter for client and user requests."""
    
    def __init__(self):
        self._client_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._user_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, 
                              client_id: Optional[str],
                              user_id: str,
                              client_tier: str = "Standard") -> Tuple[bool, Optional[Dict]]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, limit_info)
        """
        async with self._lock:
            current_time = time.time()
            
            # Get rate limits based on tier
            limits = self._get_tier_limits(client_tier)
            
            # Check user rate limit
            user_bucket = self._user_buckets[user_id]
            user_recent = [t for t in user_bucket if current_time - t < limits['user_window']]
            
            if len(user_recent) >= limits['user_limit']:
                return False, {
                    'limited_by': 'user',
                    'limit': limits['user_limit'],
                    'window': limits['user_window'],
                    'retry_after': limits['user_window'] - (current_time - user_recent[0])
                }
            
            # Check client rate limit if client_id provided
            if client_id:
                client_bucket = self._client_buckets[client_id]
                client_recent = [t for t in client_bucket if current_time - t < limits['client_window']]
                
                if len(client_recent) >= limits['client_limit']:
                    return False, {
                        'limited_by': 'client',
                        'limit': limits['client_limit'],
                        'window': limits['client_window'],
                        'retry_after': limits['client_window'] - (current_time - client_recent[0])
                    }
                
                # Add to client bucket
                client_bucket.append(current_time)
            
            # Add to user bucket
            user_bucket.append(current_time)
            
            return True, None
    
    def _get_tier_limits(self, tier: str) -> Dict[str, Any]:
        """Get rate limits based on client tier."""
        tier_limits = {
            'Platinum': {
                'user_limit': 100,
                'user_window': 60,  # 100 requests per minute per user
                'client_limit': 1000,
                'client_window': 60  # 1000 requests per minute per client
            },
            'Gold': {
                'user_limit': 50,
                'user_window': 60,
                'client_limit': 500,
                'client_window': 60
            },
            'Silver': {
                'user_limit': 30,
                'user_window': 60,
                'client_limit': 300,
                'client_window': 60
            },
            'Bronze': {
                'user_limit': 20,
                'user_window': 60,
                'client_limit': 200,
                'client_window': 60
            },
            'Standard': {
                'user_limit': 10,
                'user_window': 60,
                'client_limit': 100,
                'client_window': 60
            }
        }
        
        return tier_limits.get(tier, tier_limits['Standard'])


class DataEncryption:
    """Handles encryption of sensitive data."""
    
    def __init__(self, key: Optional[str] = None):
        if key:
            self._key = key.encode()
        else:
            # Generate key from environment or use default (NOT for production!)
            key_source = os.environ.get('ENCRYPTION_KEY', 'default-key-change-me')
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'stable-salt',  # Should be random in production
                iterations=100000,
            )
            self._key = base64.urlsafe_b64encode(kdf.derive(key_source.encode()))
        
        self._cipher = Fernet(self._key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        return self._cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        return self._cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data."""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data."""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


class AuditLogger:
    """Handles security audit logging."""
    
    def __init__(self, encryption: Optional[DataEncryption] = None):
        self._events: deque = deque(maxlen=10000)
        self._encryption = encryption
        self._lock = asyncio.Lock()
    
    async def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        async with self._lock:
            # Encrypt sensitive fields if encryption is enabled
            if self._encryption and event.client_name:
                event_dict = event.to_dict()
                event_dict['client_name'] = self._encryption.encrypt(event.client_name)
                self._events.append(event_dict)
            else:
                self._events.append(event.to_dict())
            
            # Also log to standard logger
            logger.info(
                f"AUDIT: {event.event_type}",
                extra={
                    'audit': True,
                    'event_type': event.event_type,
                    'user_id': event.user_id,
                    'action': event.action,
                    'result': event.result,
                    'correlation_id': event.correlation_id
                }
            )
    
    async def log_client_access(self, 
                               user_id: str,
                               client_name: Optional[str],
                               channel_id: str,
                               action: str,
                               success: bool,
                               correlation_id: str,
                               metadata: Optional[Dict] = None) -> None:
        """Log client data access event."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type='client_access',
            user_id=user_id,
            client_name=client_name,
            channel_id=channel_id,
            action=action,
            resource='client_data',
            result='success' if success else 'failure',
            metadata=metadata or {},
            correlation_id=correlation_id
        )
        
        await self.log_event(event)
    
    async def log_pii_detection(self,
                               user_id: str,
                               detections: List[Dict],
                               action_taken: str,
                               correlation_id: str) -> None:
        """Log PII detection event."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type='pii_detection',
            user_id=user_id,
            client_name=None,
            channel_id='',
            action=action_taken,
            resource='user_input',
            result='pii_detected',
            metadata={'detections': detections},
            correlation_id=correlation_id
        )
        
        await self.log_event(event)
    
    async def log_rate_limit(self,
                            user_id: str,
                            client_name: Optional[str],
                            limit_info: Dict,
                            correlation_id: str) -> None:
        """Log rate limit event."""
        event = AuditEvent(
            timestamp=datetime.utcnow(),
            event_type='rate_limit',
            user_id=user_id,
            client_name=client_name,
            channel_id='',
            action='request_blocked',
            resource='api',
            result='rate_limited',
            metadata=limit_info,
            correlation_id=correlation_id
        )
        
        await self.log_event(event)
    
    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Get recent audit events."""
        return list(self._events)[-limit:]


class SecurityService:
    """Main security service coordinating all security features."""
    
    def __init__(self):
        self.pii_detector = PIIDetector(enable_health_detection=True)
        self.rate_limiter = RateLimiter()
        self.encryption = DataEncryption()
        self.audit_logger = AuditLogger(encryption=self.encryption)
    
    async def process_user_input(self,
                                user_input: str,
                                user_id: str,
                                client_name: Optional[str],
                                client_tier: str,
                                channel_id: str,
                                correlation_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process user input with security controls.
        
        Returns:
            Tuple of (processed_input, security_metadata)
        """
        security_metadata = {
            'pii_detected': False,
            'pii_masked': False,
            'rate_limited': False,
            'audit_logged': True
        }
        
        # Check rate limit
        allowed, limit_info = await self.rate_limiter.check_rate_limit(
            client_id=client_name,
            user_id=user_id,
            client_tier=client_tier
        )
        
        if not allowed:
            security_metadata['rate_limited'] = True
            security_metadata['rate_limit_info'] = limit_info
            
            await self.audit_logger.log_rate_limit(
                user_id=user_id,
                client_name=client_name,
                limit_info=limit_info,
                correlation_id=correlation_id
            )
            
            raise RateLimitExceeded(limit_info)
        
        # Detect and mask PII
        masked_input, detections = self.pii_detector.mask_pii(user_input)
        
        if detections:
            security_metadata['pii_detected'] = True
            security_metadata['pii_masked'] = True
            security_metadata['pii_types'] = list(set(d['type'] for d in detections))
            
            await self.audit_logger.log_pii_detection(
                user_id=user_id,
                detections=detections,
                action_taken='masked',
                correlation_id=correlation_id
            )
            
            logger.warning(
                f"PII detected and masked",
                extra={
                    'correlation_id': correlation_id,
                    'user_id': user_id,
                    'pii_types': security_metadata['pii_types']
                }
            )
        
        # Log client access
        await self.audit_logger.log_client_access(
            user_id=user_id,
            client_name=client_name,
            channel_id=channel_id,
            action='query_processing',
            success=True,
            correlation_id=correlation_id,
            metadata={'input_length': len(user_input)}
        )
        
        return masked_input, security_metadata
    
    def encrypt_sensitive_data(self, data: Any) -> str:
        """Encrypt sensitive data for storage."""
        if isinstance(data, dict):
            return self.encryption.encrypt_dict(data)
        return self.encryption.encrypt(str(data))
    
    def decrypt_sensitive_data(self, encrypted_data: str, as_dict: bool = False) -> Any:
        """Decrypt sensitive data."""
        if as_dict:
            return self.encryption.decrypt_dict(encrypted_data)
        return self.encryption.decrypt(encrypted_data)
    
    async def get_audit_trail(self, 
                             user_id: Optional[str] = None,
                             client_name: Optional[str] = None,
                             limit: int = 100) -> List[Dict]:
        """Get audit trail with optional filtering."""
        events = self.audit_logger.get_recent_events(limit * 2)  # Get more to filter
        
        # Filter if criteria provided
        if user_id:
            events = [e for e in events if e.get('user_id') == user_id]
        
        if client_name:
            # Encrypt client name for comparison if encryption is enabled
            if self.encryption:
                encrypted_name = self.encryption.encrypt(client_name)
                events = [e for e in events if e.get('client_name') == encrypted_name]
            else:
                events = [e for e in events if e.get('client_name') == client_name]
        
        return events[:limit]


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, limit_info: Dict):
        self.limit_info = limit_info
        super().__init__(f"Rate limit exceeded: {limit_info}")


# Global security service instance
_security_service: Optional[SecurityService] = None


def get_security_service() -> SecurityService:
    """Get or create security service instance."""
    global _security_service
    
    if _security_service is None:
        _security_service = SecurityService()
    
    return _security_service