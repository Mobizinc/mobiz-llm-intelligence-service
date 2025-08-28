"""
Enhanced Client Service with pattern-based channel mapping.
Fixes the architectural issues identified in the review.
"""

import yaml
import time
import logging
import fnmatch
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class ChannelMapping:
    """Represents a channel to client mapping."""
    channel_pattern: str
    client_id: str
    mapping_type: str  # 'pattern' or 'explicit'
    description: Optional[str] = None
    channel_type: Optional[str] = None
    default_urgency: Optional[str] = None


class EnhancedClientContextCache:
    """Thread-safe cache with TTL, invalidation, and metrics."""
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached client context if not expired (thread-safe)."""
        async with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                value, timestamp = self._cache[key]
                if current_time - timestamp < self.ttl_seconds:
                    self._hits += 1
                    logger.debug(f"Cache hit for: {key} (hit rate: {self.hit_rate:.2%})")
                    # Update LRU by moving to end
                    del self._cache[key]
                    self._cache[key] = (value, timestamp)
                    return value
                else:
                    # Expired
                    del self._cache[key]
                    logger.debug(f"Cache expired for: {key}")
            
            self._misses += 1
            return None
    
    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set cached client context with LRU eviction (thread-safe)."""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
                logger.debug(f"Cache eviction: {oldest_key}")
            
            self._cache[key] = (value, time.time())
            logger.debug(f"Cache set for: {key} (size: {len(self._cache)}/{self.max_size})")
    
    async def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate specific key or entire cache."""
        async with self._lock:
            if key:
                self._cache.pop(key, None)
                logger.info(f"Cache invalidated for: {key}")
            else:
                self._cache.clear()
                logger.info("Entire cache invalidated")
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics for monitoring."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self.hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class ChannelMappingService:
    """Service for managing channel to client mappings."""
    
    def __init__(self):
        self._mappings_cache: Optional[Dict[str, Any]] = None
        self._mappings_hash: Optional[str] = None
        self._patterns: List[ChannelMapping] = []
        self._explicit: Dict[str, str] = {}
        self._excluded: set = set()
        self._channel_types: Dict[str, Dict] = {}
        
    async def load_mappings(self) -> None:
        """Load channel mappings from configuration file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "channel_mappings.yaml"
        
        try:
            # Check if file has changed
            async with aiofiles.open(config_path, 'r') as f:
                content = await f.read()
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                if content_hash == self._mappings_hash:
                    logger.debug("Channel mappings unchanged, using cached version")
                    return
                
                self._mappings_hash = content_hash
                mappings = yaml.safe_load(content)
                
                # Parse patterns
                self._patterns = []
                for pattern_config in mappings.get('patterns', []):
                    self._patterns.append(ChannelMapping(
                        channel_pattern=pattern_config['pattern'],
                        client_id=pattern_config['client'],
                        mapping_type='pattern',
                        description=pattern_config.get('description')
                    ))
                
                # Parse explicit mappings
                self._explicit = mappings.get('explicit', {})
                
                # Parse excluded channels
                self._excluded = set(mappings.get('excluded_channels', []))
                
                # Parse channel types
                self._channel_types = mappings.get('channel_types', {})
                
                logger.info(f"Loaded {len(self._patterns)} patterns, "
                          f"{len(self._explicit)} explicit mappings, "
                          f"{len(self._excluded)} exclusions")
                
        except Exception as e:
            logger.error(f"Error loading channel mappings: {e}", exc_info=True)
            # Use defaults if loading fails
            self._patterns = []
            self._explicit = {}
            self._excluded = set()
    
    def get_client_for_channel(self, channel_name: str) -> Optional[str]:
        """Get client ID for a given channel name."""
        # Strip # if present
        channel_name = channel_name.lstrip('#')
        
        # Check if excluded
        if channel_name in self._excluded:
            logger.debug(f"Channel {channel_name} is explicitly excluded")
            return None
        
        # Check explicit mappings first (highest priority)
        if channel_name in self._explicit:
            client_id = self._explicit[channel_name]
            logger.debug(f"Channel {channel_name} mapped to {client_id} (explicit)")
            return client_id
        
        # Check pattern mappings
        for mapping in self._patterns:
            if fnmatch.fnmatch(channel_name, mapping.channel_pattern):
                logger.debug(f"Channel {channel_name} matched pattern "
                           f"{mapping.channel_pattern} -> {mapping.client_id}")
                return mapping.client_id
        
        logger.debug(f"No mapping found for channel {channel_name}")
        return None
    
    def get_channel_type(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """Determine channel type and properties."""
        channel_name = channel_name.lstrip('#')
        
        for type_name, type_config in self._channel_types.items():
            patterns = type_config.get('patterns', [])
            for pattern in patterns:
                if fnmatch.fnmatch(channel_name, pattern):
                    return {
                        'type': type_name,
                        'default_urgency': type_config.get('default_urgency', 'medium'),
                        'auto_escalate': type_config.get('auto_escalate', False)
                    }
        
        return None


class EnhancedClientService:
    """Enhanced client service with improved architecture."""
    
    def __init__(self):
        self._cache = EnhancedClientContextCache()
        self._mapping_service = ChannelMappingService()
        self._clients_data: Optional[Dict[str, Any]] = None
        self._clients_hash: Optional[str] = None
        
    async def initialize(self) -> None:
        """Initialize the service and load configurations."""
        await self._mapping_service.load_mappings()
        await self._load_clients_data()
        logger.info("Enhanced client service initialized")
    
    async def _load_clients_data(self) -> None:
        """Load client data from configuration file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "clients.yaml"
        
        try:
            async with aiofiles.open(config_path, 'r') as f:
                content = await f.read()
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                if content_hash != self._clients_hash:
                    self._clients_hash = content_hash
                    self._clients_data = yaml.safe_load(content)
                    # Invalidate cache when client data changes
                    await self._cache.invalidate()
                    logger.info("Client data reloaded, cache invalidated")
                    
        except Exception as e:
            logger.error(f"Error loading client data: {e}", exc_info=True)
            self._clients_data = {'clients': {}}
    
    async def load_client_context(self, channel_name: str, 
                                 include_channel_type: bool = True) -> Dict[str, Any]:
        """
        Load client context based on channel name with pattern matching.
        
        Args:
            channel_name: Slack channel name
            include_channel_type: Whether to include channel type info
            
        Returns:
            Dict containing client context with metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{channel_name}:{include_channel_type}"
            cached_context = await self._cache.get(cache_key)
            if cached_context is not None:
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Client context loaded from cache in {duration_ms:.2f}ms",
                          extra={"channel": channel_name, "cache_hit": True})
                return cached_context
            
            # Ensure mappings are loaded
            if not self._mapping_service._patterns and not self._mapping_service._explicit:
                await self._mapping_service.load_mappings()
            
            # Get client ID from channel mapping
            client_id = self._mapping_service.get_client_for_channel(channel_name)
            
            if not client_id:
                logger.debug(f"No client mapping for channel: {channel_name}")
                result = {"_metadata": {"channel": channel_name, "mapped": False}}
                await self._cache.set(cache_key, result)
                return result
            
            # Load client data if needed
            if not self._clients_data:
                await self._load_clients_data()
            
            # Get client context
            client_context = self._clients_data.get('clients', {}).get(client_id, {})
            
            # Add metadata
            result = {
                **client_context,
                "_metadata": {
                    "channel": channel_name,
                    "client_id": client_id,
                    "mapped": True,
                    "loaded_at": datetime.utcnow().isoformat()
                }
            }
            
            # Add channel type info if requested
            if include_channel_type:
                channel_type = self._mapping_service.get_channel_type(channel_name)
                if channel_type:
                    result["_channel_type"] = channel_type
            
            # Cache the result
            await self._cache.set(cache_key, result)
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Client context loaded in {duration_ms:.2f}ms",
                      extra={
                          "channel": channel_name,
                          "client_id": client_id,
                          "cache_hit": False,
                          "duration_ms": duration_ms
                      })
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading client context: {e}",
                       extra={"channel": channel_name},
                       exc_info=True)
            return {"_metadata": {"channel": channel_name, "error": str(e)}}
    
    def get_infrastructure_summary(self, client_context: Dict[str, Any]) -> str:
        """Generate infrastructure summary with enhanced formatting."""
        if not client_context or not client_context.get("_metadata", {}).get("mapped"):
            return "No client context available"
        
        metadata = client_context.get('metadata', {})
        infrastructure = client_context.get('infrastructure', {})
        compliance = client_context.get('compliance', {})
        
        summary_parts = []
        
        # Client info with tier emoji
        tier_emojis = {
            "Platinum": "💎",
            "Gold": "🥇",
            "Silver": "🥈",
            "Bronze": "🥉",
            "Standard": "⭐"
        }
        
        client_name = metadata.get('client_name', 'Unknown')
        tier = metadata.get('tier', 'Standard')
        industry = metadata.get('industry', 'Unknown')
        tier_emoji = tier_emojis.get(tier, "")
        
        summary_parts.append(f"{tier_emoji} {client_name} ({industry}, {tier})")
        
        # Infrastructure details
        network = infrastructure.get('network', {})
        if firewall := network.get('firewall', {}):
            summary_parts.append(
                f"FW: {firewall.get('vendor', 'Unknown')} {firewall.get('model', '')}".strip()
            )
        
        if cloud := infrastructure.get('cloud', {}):
            if primary := cloud.get('primary'):
                regions = cloud.get('regions', [])
                region_str = f" ({', '.join(regions[:2])})" if regions else ""
                summary_parts.append(f"Cloud: {primary}{region_str}")
        
        # Compliance
        if frameworks := compliance.get('frameworks', []):
            summary_parts.append(f"Compliance: {', '.join(frameworks[:3])}")
        
        # Channel type info if present
        if channel_type := client_context.get('_channel_type'):
            if channel_type.get('auto_escalate'):
                summary_parts.append("⚠️ Auto-escalation enabled")
        
        return " | ".join(summary_parts)
    
    async def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        return self._cache.get_metrics()
    
    async def invalidate_cache(self, channel: Optional[str] = None) -> None:
        """Invalidate cache for specific channel or all."""
        await self._cache.invalidate(channel)
    
    async def reload_configurations(self) -> None:
        """Reload all configurations and invalidate cache."""
        await self._mapping_service.load_mappings()
        await self._load_clients_data()
        await self._cache.invalidate()
        logger.info("Configurations reloaded and cache cleared")


# Global service instance with lazy initialization
_enhanced_service: Optional[EnhancedClientService] = None


async def get_enhanced_client_service() -> EnhancedClientService:
    """Get or create the enhanced client service instance."""
    global _enhanced_service
    
    if _enhanced_service is None:
        _enhanced_service = EnhancedClientService()
        await _enhanced_service.initialize()
    
    return _enhanced_service


# Compatibility wrapper for existing code
async def load_client_context_async(channel_name: str) -> Dict[str, Any]:
    """
    Load client context from Azure Table Storage ONLY.
    Single source of truth - no YAML fallback.
    """
    try:
        from .client_storage_service import get_client_storage_service
        
        storage_service = await get_client_storage_service()
        client_id = await storage_service.get_client_for_channel(channel_name)
        
        if client_id:
            client_data = await storage_service.get_client_data(client_id)
            if client_data:
                logger.info(f"Loaded client context from Table Storage: {channel_name} → {client_id}")
                return client_data
        
        # No mapping found
        logger.debug(f"No Table Storage mapping for channel: {channel_name}")
        return {"_metadata": {"channel": channel_name, "mapped": False}}
        
    except Exception as e:
        logger.error(f"Table Storage failed for {channel_name}: {e}")
        return {"_metadata": {"channel": channel_name, "mapped": False, "error": str(e)}}


def load_client_context(channel_name: str) -> Dict[str, Any]:
    """Synchronous wrapper for backward compatibility."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(load_client_context_async(channel_name))
    finally:
        loop.close()


def get_client_infrastructure_summary(client_context: Dict[str, Any]) -> str:
    """
    Generate a concise infrastructure summary for AI prompts.
    
    Args:
        client_context: Client context dictionary from Azure Table Storage
        
    Returns:
        String summary of client infrastructure
    """
    if not client_context or '_metadata' in client_context:
        return "No client context available"
    
    summary_parts = []
    
    # Basic client info
    client_name = client_context.get('ClientName', 'Unknown Client')
    industry = client_context.get('Industry', 'Unknown Industry')
    tier = client_context.get('Tier', 'Unknown Tier')
    
    summary_parts.append(f"Client: {client_name} ({industry}, {tier} tier)")
    
    # Parse infrastructure JSON
    infrastructure_str = client_context.get('Infrastructure', '{}')
    try:
        infrastructure = json.loads(infrastructure_str) if isinstance(infrastructure_str, str) else infrastructure_str
    except (json.JSONDecodeError, TypeError):
        infrastructure = {}
    
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
    compliance_str = client_context.get('Compliance', '[]')
    try:
        compliance = json.loads(compliance_str) if isinstance(compliance_str, str) else compliance_str
        if compliance:
            summary_parts.append(f"Compliance: {', '.join(compliance)}")
    except (json.JSONDecodeError, TypeError):
        pass
    
    return "; ".join(summary_parts)


def enrich_query_with_client_context(user_query: str, client_context: Dict[str, Any]) -> str:
    """
    Enrich user query with selective client context for better AI responses.
    Only includes relevant infrastructure details based on the query content.
    
    Args:
        user_query: Original user query
        client_context: Client context dictionary from Azure Table Storage
        
    Returns:
        Enriched query string with relevant client context
    """
    if not client_context or '_metadata' in client_context:
        return user_query
    
    # Get selective infrastructure summary based on query
    relevant_infrastructure = get_relevant_infrastructure_for_query(user_query, client_context)
    
    if not relevant_infrastructure:
        # For non-infrastructure queries, just return the original query
        return user_query
    
    enriched_query = f"""
CLIENT CONTEXT: {relevant_infrastructure}

USER QUESTION: {user_query}

Please provide a response specific to this client's infrastructure and requirements.
""".strip()
    
    return enriched_query


def get_relevant_infrastructure_for_query(user_query: str, client_context: Dict[str, Any]) -> str:
    """
    Analyze the user query and return only relevant infrastructure details.
    
    Args:
        user_query: The user's question
        client_context: Full client context from Azure Table Storage
        
    Returns:
        Relevant infrastructure summary or empty string if no infrastructure context needed
    """
    if not client_context:
        return ""
    
    query_lower = user_query.lower()
    summary_parts = []
    
    # Basic client info (always include for infrastructure queries)
    client_name = client_context.get('ClientName', 'Unknown Client')
    industry = client_context.get('Industry', 'Unknown Industry')
    tier = client_context.get('Tier', 'Unknown Tier')
    
    # Parse infrastructure JSON
    infrastructure_str = client_context.get('Infrastructure', '{}')
    try:
        infrastructure = json.loads(infrastructure_str) if isinstance(infrastructure_str, str) else infrastructure_str
    except (json.JSONDecodeError, TypeError):
        infrastructure = {}
    
    # Network-related queries
    network_keywords = ['vpn', 'firewall', 'network', 'switch', 'router', 'vlan', 'subnet', 'connectivity', 'tunnel', 'ipsec']
    if any(keyword in query_lower for keyword in network_keywords):
        summary_parts.append(f"Client: {client_name} ({industry}, {tier} tier)")
        
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
        
        # Include cloud info for VPN scenarios
        cloud = infrastructure.get('cloud', {})
        if cloud and cloud.get('primary'):
            summary_parts.append(f"Primary Cloud: {cloud.get('primary')}")
    
    # Cloud-related queries
    cloud_keywords = ['azure', 'aws', 'cloud', 'vm', 'virtual machine', 'storage', 'server', 'instance', 'container', 'kubernetes']
    if any(keyword in query_lower for keyword in cloud_keywords):
        if not summary_parts:  # Avoid duplication if already added for network
            summary_parts.append(f"Client: {client_name} ({industry}, {tier} tier)")
        
        cloud = infrastructure.get('cloud', {})
        if cloud:
            primary_cloud = cloud.get('primary')
            if primary_cloud:
                regions = cloud.get('regions', [])
                region_str = f" ({', '.join(regions[:2])})" if regions else ""
                summary_parts.append(f"Primary Cloud: {primary_cloud}{region_str}")
    
    # DevOps/deployment queries
    devops_keywords = ['deploy', 'deployment', 'ci/cd', 'pipeline', 'docker', 'kubernetes', 'container', 'automation']
    if any(keyword in query_lower for keyword in devops_keywords):
        if not summary_parts:  # Avoid duplication
            summary_parts.append(f"Client: {client_name} ({industry}, {tier} tier)")
        
        # Include relevant cloud and technology stack info
        cloud = infrastructure.get('cloud', {})
        if cloud and cloud.get('primary'):
            summary_parts.append(f"Primary Cloud: {cloud.get('primary')}")
    
    # Compliance-related queries
    compliance_keywords = ['compliance', 'hipaa', 'sox', 'pci', 'gdpr', 'security', 'audit']
    if any(keyword in query_lower for keyword in compliance_keywords):
        if not summary_parts:  # Avoid duplication
            summary_parts.append(f"Client: {client_name} ({industry}, {tier} tier)")
        
        compliance_str = client_context.get('Compliance', '[]')
        try:
            compliance = json.loads(compliance_str) if isinstance(compliance_str, str) else compliance_str
            if compliance:
                summary_parts.append(f"Compliance Requirements: {', '.join(compliance)}")
        except (json.JSONDecodeError, TypeError):
            pass
    
    # For queries that don't match infrastructure patterns, return empty
    infrastructure_related = any(keyword in query_lower for keyword in 
                               network_keywords + cloud_keywords + devops_keywords + compliance_keywords)
    
    if not infrastructure_related:
        return ""
    
    return "; ".join(summary_parts) if summary_parts else ""