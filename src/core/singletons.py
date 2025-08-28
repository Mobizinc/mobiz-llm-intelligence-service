"""
Singleton instances for managers and services.

This module provides singleton patterns for various managers to ensure 
single instances across the application.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global singleton instances
_security_manager = None
_mcp_manager = None
_slack_manager = None
_telemetry_manager = None

def get_security_manager():
    """Get singleton SecurityManager instance"""
    global _security_manager
    
    if _security_manager is None:
        try:
            from ..shared.security import SecurityManager
            _security_manager = SecurityManager()
            logger.info("SecurityManager singleton initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SecurityManager: {e}")
            raise
    
    return _security_manager

def get_mcp_manager():
    """Get singleton MCPManager instance"""
    global _mcp_manager
    
    if _mcp_manager is None:
        try:
            from ..services.mcp_manager import MCPManager
            _mcp_manager = MCPManager()
            logger.info("MCPManager singleton initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MCPManager: {e}")
            raise
    
    return _mcp_manager

def get_slack_manager():
    """Get singleton SlackManager instance"""
    global _slack_manager
    
    if _slack_manager is None:
        try:
            # For now, create a simple placeholder
            class SlackManager:
                def __init__(self):
                    self.client = None
                    logger.info("SlackManager placeholder initialized")
                    
                def send_message(self, channel: str, message: str):
                    logger.info(f"Would send Slack message to {channel}: {message}")
                    
            _slack_manager = SlackManager()
            logger.info("SlackManager singleton initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SlackManager: {e}")
            raise
    
    return _slack_manager

def get_telemetry_manager():
    """Get singleton TelemetryManager instance"""
    global _telemetry_manager
    
    if _telemetry_manager is None:
        try:
            from ..telemetry.state_telemetry import get_state_tracer
            _telemetry_manager = get_state_tracer()
            logger.info("TelemetryManager singleton initialized")
        except Exception as e:
            logger.error(f"Failed to initialize TelemetryManager: {e}")
            raise
    
    return _telemetry_manager

def reset_singletons():
    """Reset all singleton instances (useful for testing)"""
    global _security_manager, _mcp_manager, _slack_manager, _telemetry_manager
    
    _security_manager = None
    _mcp_manager = None  
    _slack_manager = None
    _telemetry_manager = None
    
    logger.info("All singleton instances reset")