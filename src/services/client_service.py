"""
Client Service for loading client context based on channel mapping.
Provides client-aware information for MSP bot responses.
"""

import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class ClientContextCache:
    """Simple cache with TTL for client context data"""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minute default TTL
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached client context if not expired"""
        current_time = time.time()
        
        if key in self._cache:
            if current_time - self._timestamps[key] < self.ttl_seconds:
                logger.debug(f"Cache hit for client context: {key}")
                return self._cache[key]
            else:
                # Expired, remove from cache
                del self._cache[key]
                del self._timestamps[key]
                logger.debug(f"Cache expired for client context: {key}")
        
        return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set cached client context with current timestamp"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
        logger.debug(f"Cache set for client context: {key}")


# Global cache instance
_client_cache = ClientContextCache()


def load_client_context(channel_name: str) -> Dict[str, Any]:
    """
    Load client context based on channel name.
    
    Args:
        channel_name: Slack channel name to map to client
        
    Returns:
        Dict containing client context or empty dict if not found
    """
    start_time = time.time()
    
    try:
        # Check cache first
        cached_context = _client_cache.get(channel_name)
        if cached_context is not None:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Client context loaded from cache",
                extra={
                    "channel_name": channel_name,
                    "duration_ms": duration_ms,
                    "cache_hit": True
                }
            )
            return cached_context
        
        # Load from file
        config_path = Path(__file__).parent.parent.parent / "config" / "clients.yaml"
        
        if not config_path.exists():
            logger.warning(f"Client configuration file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            clients_config = yaml.safe_load(f)
        
        # Get client context for this channel
        client_context = clients_config.get('clients', {}).get(channel_name, {})
        
        # Cache the result (even if empty)
        _client_cache.set(channel_name, client_context)
        
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Client context loaded from file",
            extra={
                "channel_name": channel_name,
                "duration_ms": duration_ms,
                "cache_hit": False,
                "client_found": bool(client_context),
                "client_name": client_context.get('metadata', {}).get('client_name') if client_context else None
            }
        )
        
        return client_context
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Error loading client context for channel: {channel_name}",
            extra={
                "channel_name": channel_name,
                "duration_ms": duration_ms,
                "error": str(e)
            },
            exc_info=True
        )
        return {}


def get_client_infrastructure_summary(client_context: Dict[str, Any]) -> str:
    """
    Generate a concise infrastructure summary for AI prompts.
    
    Args:
        client_context: Client context dictionary
        
    Returns:
        String summary of client infrastructure
    """
    if not client_context:
        return "No client context available"
    
    metadata = client_context.get('metadata', {})
    infrastructure = client_context.get('infrastructure', {})
    
    summary_parts = []
    
    # Basic client info
    client_name = metadata.get('client_name', 'Unknown Client')
    industry = metadata.get('industry', 'Unknown Industry')
    tier = metadata.get('tier', 'Unknown Tier')
    
    summary_parts.append(f"Client: {client_name} ({industry}, {tier} tier)")
    
    # Network infrastructure
    network = infrastructure.get('network', {})
    if network:
        firewall = network.get('firewall', {})
        if firewall:
            fw_vendor = firewall.get('vendor', 'Unknown')
            fw_model = firewall.get('model', '')
            summary_parts.append(f"Firewall: {fw_vendor} {fw_model}".strip())
        
        switches = network.get('switches')
        if switches:
            summary_parts.append(f"Switches: {switches}")
    
    # Cloud infrastructure
    cloud = infrastructure.get('cloud', {})
    if cloud:
        primary_cloud = cloud.get('primary')
        if primary_cloud:
            summary_parts.append(f"Primary Cloud: {primary_cloud}")
    
    # Compliance requirements
    compliance = client_context.get('compliance', {})
    frameworks = compliance.get('frameworks', [])
    if frameworks:
        summary_parts.append(f"Compliance: {', '.join(frameworks)}")
    
    return "; ".join(summary_parts)


def enrich_query_with_client_context(user_query: str, client_context: Dict[str, Any]) -> str:
    """
    Enrich user query with client context for better AI responses.
    
    Args:
        user_query: Original user query
        client_context: Client context dictionary
        
    Returns:
        Enriched query string with client context
    """
    if not client_context:
        return user_query
    
    infrastructure_summary = get_client_infrastructure_summary(client_context)
    
    enriched_query = f"""
CLIENT CONTEXT: {infrastructure_summary}

USER QUESTION: {user_query}

Please provide a response specific to this client's infrastructure and requirements.
""".strip()
    
    return enriched_query


@lru_cache(maxsize=10)
def get_available_clients() -> Dict[str, str]:
    """
    Get list of available clients and their names.
    Cached to avoid repeated file reads.
    
    Returns:
        Dict mapping channel names to client names
    """
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "clients.yaml"
        
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            clients_config = yaml.safe_load(f)
        
        clients = {}
        for channel_name, client_data in clients_config.get('clients', {}).items():
            client_name = client_data.get('metadata', {}).get('client_name', channel_name)
            clients[channel_name] = client_name
        
        return clients
        
    except Exception as e:
        logger.error(f"Error loading available clients: {e}", exc_info=True)
        return {}


def is_client_channel(channel_name: str) -> bool:
    """
    Check if a channel name corresponds to a known client.
    
    Args:
        channel_name: Slack channel name
        
    Returns:
        True if channel is mapped to a client
    """
    available_clients = get_available_clients()
    return channel_name in available_clients