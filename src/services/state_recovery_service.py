"""
State Recovery Service

Handles conversation state recovery, corruption detection, and emergency procedures.

Part of Epic 1: Core State Management & Infrastructure
Story 1.3: State Recovery & Rollback
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..models.conversation_state import (
    TechnicalConversationState,
    ConversationStateManager,
    ConversationStage
)
from ..services.langgraph_state_service import get_langgraph_state_service
from ..shared.telemetry import EnhancedTelemetryManager

logger = logging.getLogger(__name__)


class RecoveryStatus(str, Enum):
    """Recovery operation status"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    CORRUPTED = "corrupted"
    NOT_FOUND = "not_found"


class StateCorruptionType(str, Enum):
    """Types of state corruption detected"""
    INVALID_JSON = "invalid_json"
    MISSING_REQUIRED_FIELDS = "missing_required_fields"
    INVALID_FIELD_VALUES = "invalid_field_values"
    INCONSISTENT_STATE = "inconsistent_state"
    SCHEMA_MISMATCH = "schema_mismatch"


class StateRecoveryService:
    """
    Service for state recovery, corruption detection, and emergency procedures.
    
    Features:
    - Automatic corruption detection and repair
    - Multi-level fallback strategies
    - State reconstruction from conversation history
    - Emergency state creation with safe defaults
    - Comprehensive recovery logging and metrics
    """
    
    def __init__(self):
        self.state_service = get_langgraph_state_service()
        self.telemetry_manager = EnhancedTelemetryManager()
        
        # Recovery statistics
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'corruptions_detected': 0,
            'corruptions_repaired': 0,
            'emergency_states_created': 0
        }
        
        logger.info("StateRecoveryService initialized")
    
    async def recover_conversation_state(
        self,
        conversation_id: str,
        fallback_to_emergency: bool = True
    ) -> Tuple[Optional[TechnicalConversationState], RecoveryStatus, List[str]]:
        """
        Attempt to recover conversation state using multiple strategies.
        
        Args:
            conversation_id: Unique conversation identifier
            fallback_to_emergency: Whether to create emergency state if all recovery fails
            
        Returns:
            Tuple of (recovered_state, recovery_status, recovery_log)
        """
        recovery_log = []
        start_time = datetime.now(timezone.utc)
        
        self.recovery_stats['total_recoveries'] += 1
        
        try:
            # Strategy 1: Load current state and validate
            recovery_log.append(f"[{start_time.isoformat()}] Starting recovery for {conversation_id}")
            
            current_state = await self.state_service.load_state(conversation_id)
            if current_state:
                corruption_issues = await self._detect_state_corruption(current_state)
                
                if not corruption_issues:
                    recovery_log.append("Current state is valid - no recovery needed")
                    self.recovery_stats['successful_recoveries'] += 1
                    return current_state, RecoveryStatus.SUCCESS, recovery_log
                
                # Strategy 2: Attempt to repair corrupted state
                recovery_log.append(f"State corruption detected: {corruption_issues}")
                repaired_state = await self._attempt_state_repair(current_state, corruption_issues)
                
                if repaired_state:
                    recovery_log.append("State successfully repaired")
                    await self.state_service.save_state(conversation_id, repaired_state)
                    self.recovery_stats['successful_recoveries'] += 1
                    self.recovery_stats['corruptions_repaired'] += 1
                    return repaired_state, RecoveryStatus.SUCCESS, recovery_log
            
            # Strategy 3: Try to load from latest checkpoint
            recovery_log.append("Attempting recovery from latest checkpoint")
            checkpoint_state = await self._recover_from_latest_checkpoint(conversation_id)
            
            if checkpoint_state:
                recovery_log.append("Recovered from latest checkpoint")
                await self.state_service.save_state(conversation_id, checkpoint_state)
                self.recovery_stats['successful_recoveries'] += 1
                return checkpoint_state, RecoveryStatus.PARTIAL, recovery_log
            
            # Strategy 4: Reconstruct from conversation history
            recovery_log.append("Attempting reconstruction from conversation history")
            reconstructed_state = await self._reconstruct_from_history(conversation_id)
            
            if reconstructed_state:
                recovery_log.append("State reconstructed from conversation history")
                await self.state_service.save_state(conversation_id, reconstructed_state)
                self.recovery_stats['successful_recoveries'] += 1
                return reconstructed_state, RecoveryStatus.PARTIAL, recovery_log
            
            # Strategy 5: Create emergency fallback state
            if fallback_to_emergency:
                recovery_log.append("Creating emergency fallback state")
                emergency_state = await self._create_emergency_state(conversation_id)
                
                if emergency_state:
                    recovery_log.append("Emergency state created")
                    await self.state_service.save_state(conversation_id, emergency_state)
                    self.recovery_stats['emergency_states_created'] += 1
                    return emergency_state, RecoveryStatus.PARTIAL, recovery_log
            
            # All recovery strategies failed
            recovery_log.append("All recovery strategies failed")
            self.recovery_stats['failed_recoveries'] += 1
            
            self.telemetry_manager.log_custom_event(
                "state.recovery.failed",
                {
                    "conversation_id": conversation_id,
                    "recovery_log": recovery_log,
                    "duration_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                }
            )
            
            return None, RecoveryStatus.FAILED, recovery_log
            
        except Exception as e:
            error_msg = f"Recovery process failed with exception: {str(e)}"
            recovery_log.append(error_msg)
            logger.error(error_msg)
            
            self.recovery_stats['failed_recoveries'] += 1
            return None, RecoveryStatus.FAILED, recovery_log
    
    async def _detect_state_corruption(
        self,
        state: TechnicalConversationState
    ) -> List[StateCorruptionType]:
        """
        Detect various types of state corruption.
        
        Args:
            state: State to validate
            
        Returns:
            List of corruption types detected
        """
        corruption_issues = []
        
        try:
            # Test 1: Validate against schema
            try:
                ConversationStateManager.validate_state(state)
            except Exception as e:
                corruption_issues.append(StateCorruptionType.INVALID_FIELD_VALUES)
                logger.warning(f"Schema validation failed: {e}")
            
            # Test 2: Check required fields
            required_fields = [
                'conversation_id', 'initial_query', 'current_input', 'domain',
                'thread_id', 'intent', 'stage', 'user_id', 'channel_id',
                'correlation_id', 'created_at', 'updated_at', 'version'
            ]
            
            missing_fields = [field for field in required_fields if field not in state]
            if missing_fields:
                corruption_issues.append(StateCorruptionType.MISSING_REQUIRED_FIELDS)
                logger.warning(f"Missing required fields: {missing_fields}")
            
            # Test 3: Check field value consistency
            if state.get('stage') == ConversationStage.SUFFICIENT:
                if state.get('missing_fields', []):
                    corruption_issues.append(StateCorruptionType.INCONSISTENT_STATE)
                    logger.warning("Stage 'sufficient' has missing_fields")
            
            if state.get('confidence_score', 0) > 1.0 or state.get('confidence_score', 0) < 0.0:
                corruption_issues.append(StateCorruptionType.INVALID_FIELD_VALUES)
                logger.warning("Invalid confidence_score value")
            
            # Test 4: Check timestamp validity
            try:
                created_at = datetime.fromisoformat(state['created_at'].replace('Z', '+00:00'))
                updated_at = datetime.fromisoformat(state['updated_at'].replace('Z', '+00:00'))
                
                if updated_at < created_at:
                    corruption_issues.append(StateCorruptionType.INCONSISTENT_STATE)
                    logger.warning("updated_at is before created_at")
            except (ValueError, KeyError) as e:
                corruption_issues.append(StateCorruptionType.INVALID_FIELD_VALUES)
                logger.warning(f"Invalid timestamp format: {e}")
            
            # Test 5: Check version consistency
            version = state.get('version', 0)
            if not isinstance(version, int) or version < 1:
                corruption_issues.append(StateCorruptionType.INVALID_FIELD_VALUES)
                logger.warning(f"Invalid version number: {version}")
            
            if corruption_issues:
                self.recovery_stats['corruptions_detected'] += 1
                
                self.telemetry_manager.log_custom_event(
                    "state.corruption.detected",
                    {
                        "conversation_id": state.get('conversation_id', 'unknown'),
                        "corruption_types": [ct.value for ct in corruption_issues],
                        "state_version": state.get('version', 0)
                    }
                )
            
        except Exception as e:
            logger.error(f"Error during corruption detection: {e}")
            corruption_issues.append(StateCorruptionType.INVALID_JSON)
        
        return corruption_issues
    
    async def _attempt_state_repair(
        self,
        corrupted_state: TechnicalConversationState,
        corruption_issues: List[StateCorruptionType]
    ) -> Optional[TechnicalConversationState]:
        """
        Attempt to repair corrupted state.
        
        Args:
            corrupted_state: State with corruption issues
            corruption_issues: List of detected corruption types
            
        Returns:
            Repaired state if successful, None otherwise
        """
        try:
            repaired_state = corrupted_state.copy()
            
            # Repair missing required fields
            if StateCorruptionType.MISSING_REQUIRED_FIELDS in corruption_issues:
                now = datetime.now(timezone.utc).isoformat()
                
                defaults = {
                    'conversation_id': f"recovered_{now}",
                    'initial_query': "Recovered conversation",
                    'current_input': repaired_state.get('initial_query', 'Recovered conversation'),
                    'domain': 'cloud',
                    'thread_id': f"recovered_thread_{now}",
                    'intent': 'new_query',
                    'confidence_score': 0.5,
                    'extracted_info': {},
                    'accumulated_info': {},
                    'required_fields': [],
                    'missing_fields': [],
                    'stage': 'initial',
                    'questions_asked': [],
                    'response': '',
                    'response_type': 'question',
                    'user_id': 'recovered_user',
                    'channel_id': 'recovered_channel',
                    'correlation_id': f"recovered_corr_{now}",
                    'created_at': now,
                    'updated_at': now,
                    'version': 1
                }
                
                for field, default_value in defaults.items():
                    if field not in repaired_state:
                        repaired_state[field] = default_value
            
            # Repair invalid field values
            if StateCorruptionType.INVALID_FIELD_VALUES in corruption_issues:
                # Fix confidence score
                confidence = repaired_state.get('confidence_score', 0.5)
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    repaired_state['confidence_score'] = 0.5
                
                # Fix version
                version = repaired_state.get('version', 1)
                if not isinstance(version, int) or version < 1:
                    repaired_state['version'] = 1
                
                # Fix timestamps
                now = datetime.now(timezone.utc).isoformat()
                try:
                    datetime.fromisoformat(repaired_state['created_at'].replace('Z', '+00:00'))
                except (ValueError, KeyError):
                    repaired_state['created_at'] = now
                
                try:
                    datetime.fromisoformat(repaired_state['updated_at'].replace('Z', '+00:00'))
                except (ValueError, KeyError):
                    repaired_state['updated_at'] = now
            
            # Fix inconsistent state
            if StateCorruptionType.INCONSISTENT_STATE in corruption_issues:
                if repaired_state.get('stage') == ConversationStage.SUFFICIENT:
                    repaired_state['missing_fields'] = []
                
                # Ensure updated_at is not before created_at
                try:
                    created_at = datetime.fromisoformat(repaired_state['created_at'].replace('Z', '+00:00'))
                    updated_at = datetime.fromisoformat(repaired_state['updated_at'].replace('Z', '+00:00'))
                    
                    if updated_at < created_at:
                        repaired_state['updated_at'] = repaired_state['created_at']
                except ValueError:
                    pass
            
            # Validate repaired state
            ConversationStateManager.validate_state(repaired_state)
            
            # Add repair metadata
            repaired_state['metadata'] = repaired_state.get('metadata', {})
            repaired_state['metadata']['repaired_at'] = datetime.now(timezone.utc).isoformat()
            repaired_state['metadata']['corruption_issues'] = [ci.value for ci in corruption_issues]
            
            logger.info(f"Successfully repaired state for conversation {repaired_state['conversation_id']}")
            return repaired_state
            
        except Exception as e:
            logger.error(f"Failed to repair state: {e}")
            return None
    
    async def _recover_from_latest_checkpoint(
        self,
        conversation_id: str
    ) -> Optional[TechnicalConversationState]:
        """
        Attempt to recover from the latest valid checkpoint.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Recovered state from checkpoint if found, None otherwise
        """
        try:
            checkpoints = await self.state_service.list_checkpoints(conversation_id)
            
            # Try checkpoints in reverse chronological order
            for checkpoint_info in checkpoints:
                checkpoint_name = checkpoint_info['checkpoint_name']
                
                try:
                    checkpoint_state = await self.state_service.load_state(
                        conversation_id, checkpoint_name
                    )
                    
                    if checkpoint_state:
                        # Validate checkpoint state
                        corruption_issues = await self._detect_state_corruption(checkpoint_state)
                        
                        if not corruption_issues:
                            logger.info(f"Recovered from checkpoint: {checkpoint_name}")
                            return checkpoint_state
                        
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {checkpoint_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to recover from checkpoints: {e}")
            return None
    
    async def _reconstruct_from_history(
        self,
        conversation_id: str
    ) -> Optional[TechnicalConversationState]:
        """
        Attempt to reconstruct state from conversation history.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Reconstructed state if possible, None otherwise
        """
        try:
            # This would require access to conversation history
            # For now, return None - this could be implemented to analyze
            # conversation messages and reconstruct the state
            
            logger.info(f"State reconstruction from history not yet implemented for {conversation_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to reconstruct from history: {e}")
            return None
    
    async def _create_emergency_state(
        self,
        conversation_id: str
    ) -> Optional[TechnicalConversationState]:
        """
        Create an emergency fallback state with safe defaults.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Emergency state with safe defaults
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            emergency_state = ConversationStateManager.create_initial_state(
                conversation_id=conversation_id,
                initial_query="Emergency recovery - previous conversation state was lost",
                domain="cloud",  # Safe default
                thread_id=f"emergency_{conversation_id}",
                user_id="emergency_user",
                channel_id="emergency_channel",
                correlation_id=f"emergency_{now}",
                metadata={
                    "emergency_recovery": True,
                    "recovery_timestamp": now,
                    "recovery_reason": "State corruption or loss"
                }
            )
            
            logger.info(f"Created emergency state for conversation {conversation_id}")
            return emergency_state
            
        except Exception as e:
            logger.error(f"Failed to create emergency state: {e}")
            return None
    
    async def validate_state_integrity(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive state integrity validation.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Integrity validation report
        """
        report = {
            'conversation_id': conversation_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Load current state
            current_state = await self.state_service.load_state(conversation_id)
            
            if not current_state:
                report['status'] = 'not_found'
                report['issues'].append('No state found for conversation')
                report['recommendations'].append('Create new conversation state')
                return report
            
            # Check for corruption
            corruption_issues = await self._detect_state_corruption(current_state)
            
            if corruption_issues:
                report['status'] = 'corrupted'
                report['issues'].extend([f"Corruption: {ci.value}" for ci in corruption_issues])
                report['recommendations'].append('Run state recovery process')
            else:
                report['status'] = 'healthy'
            
            # Check state age
            try:
                created_at = datetime.fromisoformat(current_state['created_at'].replace('Z', '+00:00'))
                age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
                
                if age_hours > 48:  # Older than 48 hours
                    report['issues'].append(f'State is {age_hours:.1f} hours old')
                    report['recommendations'].append('Consider conversation cleanup')
            except ValueError:
                report['issues'].append('Invalid created_at timestamp')
            
            # Check version consistency
            version = current_state.get('version', 0)
            if version > 100:  # Unusually high version number
                report['issues'].append(f'High version number: {version}')
                report['recommendations'].append('Investigate version inflation')
            
            # Check for excessive node history
            node_history = current_state.get('node_history', [])
            if len(node_history) > 50:
                report['issues'].append(f'Large node history: {len(node_history)} entries')
                report['recommendations'].append('Consider node history cleanup')
            
            # Check for excessive errors
            error_history = current_state.get('error_history', [])
            if len(error_history) > 10:
                report['issues'].append(f'Many errors: {len(error_history)} entries')
                report['recommendations'].append('Investigate error patterns')
            
        except Exception as e:
            report['status'] = 'error'
            report['issues'].append(f'Validation failed: {str(e)}')
        
        return report
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery service statistics"""
        stats = self.recovery_stats.copy()
        
        # Calculate rates
        if stats['total_recoveries'] > 0:
            stats['success_rate'] = stats['successful_recoveries'] / stats['total_recoveries']
            stats['corruption_detection_rate'] = stats['corruptions_detected'] / stats['total_recoveries']
            
            if stats['corruptions_detected'] > 0:
                stats['repair_success_rate'] = stats['corruptions_repaired'] / stats['corruptions_detected']
            else:
                stats['repair_success_rate'] = 0.0
        else:
            stats['success_rate'] = 0.0
            stats['corruption_detection_rate'] = 0.0
            stats['repair_success_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset recovery statistics"""
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'corruptions_detected': 0,
            'corruptions_repaired': 0,
            'emergency_states_created': 0
        }
        
        logger.info("Recovery statistics reset")


# Global instance for dependency injection
_state_recovery_service = None

def get_state_recovery_service() -> StateRecoveryService:
    """Get singleton instance of StateRecoveryService"""
    global _state_recovery_service
    
    if _state_recovery_service is None:
        _state_recovery_service = StateRecoveryService()
    
    return _state_recovery_service