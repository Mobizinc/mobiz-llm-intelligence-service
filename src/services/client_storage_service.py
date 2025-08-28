"""
Client Storage Service - Azure Table Storage Implementation
Replaces YAML-based client configuration with dynamic Azure Table Storage
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from azure.data.tables import TableServiceClient, TableClient
from azure.data.tables.aio import TableServiceClient as AsyncTableServiceClient, TableClient as AsyncTableClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
import fnmatch
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ClientMappingRule:
    """Represents a client mapping rule."""
    channel_pattern: str
    client_id: str
    mapping_type: str  # 'explicit' or 'pattern'
    description: str
    priority: int
    timestamp: datetime


@dataclass
class ClientInfo:
    """Represents client information."""
    client_id: str
    client_name: str
    industry: str
    tier: str
    monthly_revenue: int
    account_manager: str
    infrastructure: Dict[str, Any]
    compliance: List[str]
    support: Dict[str, Any]
    timestamp: datetime


class ClientStorageCache:
    """In-memory cache for client storage data with TTL."""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default
        self.ttl_seconds = ttl_seconds
        self._mappings_cache: Dict[str, str] = {}
        self._clients_cache: Dict[str, ClientInfo] = {}
        self._patterns_cache: List[ClientMappingRule] = []
        self._cache_timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get_mapping(self, channel: str) -> Optional[str]:
        """Get cached client mapping."""
        async with self._lock:
            if self._is_expired('mappings'):
                return None
            return self._mappings_cache.get(channel)
    
    async def set_mapping(self, channel: str, client_id: str):
        """Cache client mapping."""
        async with self._lock:
            self._mappings_cache[channel] = client_id
            self._cache_timestamps['mappings'] = time.time()
    
    async def get_client(self, client_id: str) -> Optional[ClientInfo]:
        """Get cached client info."""
        async with self._lock:
            if self._is_expired('clients'):
                return None
            return self._clients_cache.get(client_id)
    
    async def set_client(self, client_id: str, client_info: ClientInfo):
        """Cache client info."""
        async with self._lock:
            self._clients_cache[client_id] = client_info
            self._cache_timestamps['clients'] = time.time()
    
    async def get_patterns(self) -> List[ClientMappingRule]:
        """Get cached patterns."""
        async with self._lock:
            if self._is_expired('patterns'):
                return []
            return self._patterns_cache.copy()
    
    async def set_patterns(self, patterns: List[ClientMappingRule]):
        """Cache patterns."""
        async with self._lock:
            self._patterns_cache = patterns.copy()
            self._cache_timestamps['patterns'] = time.time()
    
    async def invalidate(self, cache_type: Optional[str] = None):
        """Invalidate cache."""
        async with self._lock:
            if cache_type:
                self._cache_timestamps.pop(cache_type, None)
                if cache_type == 'mappings':
                    self._mappings_cache.clear()
                elif cache_type == 'clients':
                    self._clients_cache.clear()
                elif cache_type == 'patterns':
                    self._patterns_cache.clear()
            else:
                self._cache_timestamps.clear()
                self._mappings_cache.clear()
                self._clients_cache.clear()
                self._patterns_cache.clear()
    
    def _is_expired(self, cache_type: str) -> bool:
        """Check if cache is expired."""
        timestamp = self._cache_timestamps.get(cache_type, 0)
        return time.time() - timestamp > self.ttl_seconds


class ClientStorageService:
    """Azure Table Storage service for client configuration management."""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string
        self._async_table_service: Optional[AsyncTableServiceClient] = None
        self._mappings_table: Optional[AsyncTableClient] = None
        self._patterns_table: Optional[AsyncTableClient] = None
        self._clients_table: Optional[AsyncTableClient] = None
        self._cache = ClientStorageCache()
        self._initialized = False
        
        # Table names
        self.mappings_table_name = "ClientMappings"
        self.patterns_table_name = "ClientMappingPatterns"
        self.clients_table_name = "ClientData"
    
    async def initialize(self):
        """Initialize table clients."""
        if self._initialized:
            return
        
        try:
            if not self.connection_string:
                # Try environment variable first for local testing
                import os
                self.connection_string = os.environ.get("TABLE_STORAGE_CONNECTION")
                
                # If not in environment, try security manager
                if not self.connection_string:
                    try:
                        from ..shared.security import get_security_manager
                        security_manager = get_security_manager()
                        self.connection_string = await security_manager.get_secret("table-storage-connection")
                    except ImportError:
                        # Fallback for local testing
                        pass
            
            if not self.connection_string:
                raise ValueError("Table storage connection string not available")
            
            # Initialize async table service
            self._async_table_service = AsyncTableServiceClient.from_connection_string(
                self.connection_string
            )
            
            # Initialize table clients
            self._mappings_table = self._async_table_service.get_table_client(
                table_name=self.mappings_table_name
            )
            self._patterns_table = self._async_table_service.get_table_client(
                table_name=self.patterns_table_name
            )
            self._clients_table = self._async_table_service.get_table_client(
                table_name=self.clients_table_name
            )
            
            # Create tables if they don't exist
            await self._ensure_tables_exist()
            
            self._initialized = True
            logger.info("Client storage service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize client storage service: {e}")
            raise
    
    async def _ensure_tables_exist(self):
        """Create tables if they don't exist."""
        try:
            await self._mappings_table.create_table()
            logger.info(f"Created table: {self.mappings_table_name}")
        except HttpResponseError as e:
            if e.status_code != 409:  # Table already exists
                raise
        
        try:
            await self._patterns_table.create_table()
            logger.info(f"Created table: {self.patterns_table_name}")
        except HttpResponseError as e:
            if e.status_code != 409:  # Table already exists
                raise
        
        try:
            await self._clients_table.create_table()
            logger.info(f"Created table: {self.clients_table_name}")
        except HttpResponseError as e:
            if e.status_code != 409:  # Table already exists
                raise
    
    async def get_client_for_channel(self, channel_name: str) -> Optional[str]:
        """
        Get client ID for a given channel name.
        
        Args:
            channel_name: Slack channel name
            
        Returns:
            Client ID or None if no mapping found
        """
        await self.initialize()
        
        # Check cache first
        cached_mapping = await self._cache.get_mapping(channel_name)
        if cached_mapping is not None:
            return cached_mapping
        
        try:
            # 1. Check explicit mappings first (higher priority)
            try:
                entity = await self._mappings_table.get_entity(
                    partition_key="mapping",
                    row_key=channel_name
                )
                client_id = entity.get("ClientId")
                if client_id:
                    await self._cache.set_mapping(channel_name, client_id)
                    logger.debug(f"Found explicit mapping: {channel_name} → {client_id}")
                    return client_id
            except ResourceNotFoundError:
                pass  # No explicit mapping found
            
            # 2. Check pattern mappings
            patterns = await self._get_mapping_patterns()
            
            # Sort by priority (lower number = higher priority)
            patterns.sort(key=lambda x: x.priority)
            
            for pattern_rule in patterns:
                if fnmatch.fnmatch(channel_name, pattern_rule.channel_pattern):
                    client_id = pattern_rule.client_id
                    # Cache the resolved mapping
                    await self._cache.set_mapping(channel_name, client_id)
                    logger.debug(f"Found pattern mapping: {channel_name} matches {pattern_rule.channel_pattern} → {client_id}")
                    return client_id
            
            # No mapping found
            await self._cache.set_mapping(channel_name, "")  # Cache negative result
            logger.debug(f"No mapping found for channel: {channel_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting client for channel {channel_name}: {e}")
            return None
    
    async def _get_mapping_patterns(self) -> List[ClientMappingRule]:
        """Get all mapping patterns from cache or table."""
        # Check cache first
        cached_patterns = await self._cache.get_patterns()
        if cached_patterns:
            return cached_patterns
        
        try:
            patterns = []
            async for entity in self._patterns_table.list_entities():
                pattern_rule = ClientMappingRule(
                    channel_pattern=entity.get("Pattern", ""),
                    client_id=entity.get("ClientId", ""),
                    mapping_type="pattern",
                    description=entity.get("Description", ""),
                    priority=entity.get("Priority", 999),
                    timestamp=entity.get("Timestamp", datetime.utcnow())
                )
                patterns.append(pattern_rule)
            
            await self._cache.set_patterns(patterns)
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting mapping patterns: {e}")
            return []
    
    async def get_client_data(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        Get client data for a given client ID.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client data dictionary or None if not found
        """
        await self.initialize()
        
        # Check cache first
        cached_client = await self._cache.get_client(client_id)
        if cached_client:
            return self._client_info_to_dict(cached_client)
        
        try:
            entity = await self._clients_table.get_entity(
                partition_key="client",
                row_key=client_id
            )
            
            # Parse JSON fields
            infrastructure = json.loads(entity.get("Infrastructure", "{}"))
            compliance = json.loads(entity.get("Compliance", "[]"))
            support = json.loads(entity.get("Support", "{}"))
            
            client_info = ClientInfo(
                client_id=client_id,
                client_name=entity.get("ClientName", ""),
                industry=entity.get("Industry", ""),
                tier=entity.get("Tier", ""),
                monthly_revenue=entity.get("MonthlyRevenue", 0),
                account_manager=entity.get("AccountManager", ""),
                infrastructure=infrastructure,
                compliance=compliance,
                support=support,
                timestamp=entity.get("Timestamp", datetime.utcnow())
            )
            
            await self._cache.set_client(client_id, client_info)
            
            return self._client_info_to_dict(client_info)
            
        except ResourceNotFoundError:
            logger.debug(f"Client data not found: {client_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting client data for {client_id}: {e}")
            return None
    
    def _client_info_to_dict(self, client_info: ClientInfo) -> Dict[str, Any]:
        """Convert ClientInfo to dictionary format."""
        return {
            "metadata": {
                "client_name": client_info.client_name,
                "industry": client_info.industry,
                "tier": client_info.tier,
                "monthly_revenue": client_info.monthly_revenue,
                "account_manager": client_info.account_manager
            },
            "infrastructure": client_info.infrastructure,
            "compliance": {"frameworks": client_info.compliance},
            "support": client_info.support
        }
    
    async def add_channel_mapping(self, channel_name: str, client_id: str, 
                                mapping_type: str = "explicit", 
                                description: str = "", priority: int = 1) -> bool:
        """
        Add or update a channel mapping.
        
        Args:
            channel_name: Channel name to map
            client_id: Client ID to map to
            mapping_type: 'explicit' or 'pattern'
            description: Description of the mapping
            priority: Priority for conflict resolution (lower = higher priority)
            
        Returns:
            True if successful, False otherwise
        """
        await self.initialize()
        
        try:
            entity = {
                "PartitionKey": "mapping",
                "RowKey": channel_name,
                "ClientId": client_id,
                "MappingType": mapping_type,
                "Description": description,
                "Priority": priority,
                "Timestamp": datetime.utcnow()
            }
            
            await self._mappings_table.upsert_entity(entity)
            
            # Invalidate cache
            await self._cache.invalidate("mappings")
            
            logger.info(f"Added channel mapping: {channel_name} → {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding channel mapping: {e}")
            return False
    
    async def add_pattern_mapping(self, pattern: str, client_id: str,
                                description: str = "", priority: int = 10) -> bool:
        """
        Add or update a pattern mapping.
        
        Args:
            pattern: Channel pattern (e.g., 'allcare-*')
            client_id: Client ID to map to
            description: Description of the pattern
            priority: Priority for pattern matching (lower = higher priority)
            
        Returns:
            True if successful, False otherwise
        """
        await self.initialize()
        
        try:
            # Generate a unique row key for the pattern
            pattern_id = f"{client_id}_{pattern.replace('*', 'wildcard').replace('-', '_')}"
            
            entity = {
                "PartitionKey": "pattern",
                "RowKey": pattern_id,
                "Pattern": pattern,
                "ClientId": client_id,
                "Description": description,
                "Priority": priority,
                "Timestamp": datetime.utcnow()
            }
            
            await self._patterns_table.upsert_entity(entity)
            
            # Invalidate cache
            await self._cache.invalidate("patterns")
            
            logger.info(f"Added pattern mapping: {pattern} → {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding pattern mapping: {e}")
            return False
    
    async def update_client_data(self, client_id: str, client_data: Dict[str, Any]) -> bool:
        """
        Update client data.
        
        Args:
            client_id: Client ID to update
            client_data: Client data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        await self.initialize()
        
        try:
            metadata = client_data.get("metadata", {})
            infrastructure = client_data.get("infrastructure", {})
            compliance = client_data.get("compliance", {})
            support = client_data.get("support", {})
            
            entity = {
                "PartitionKey": "client",
                "RowKey": client_id,
                "ClientName": metadata.get("client_name", ""),
                "Industry": metadata.get("industry", ""),
                "Tier": metadata.get("tier", ""),
                "MonthlyRevenue": metadata.get("monthly_revenue", 0),
                "AccountManager": metadata.get("account_manager", ""),
                "Infrastructure": json.dumps(infrastructure),
                "Compliance": json.dumps(compliance.get("frameworks", [])),
                "Support": json.dumps(support),
                "Timestamp": datetime.utcnow()
            }
            
            await self._clients_table.upsert_entity(entity)
            
            # Invalidate cache
            await self._cache.invalidate("clients")
            
            logger.info(f"Updated client data: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating client data: {e}")
            return False
    
    async def delete_channel_mapping(self, channel_name: str) -> bool:
        """Delete a channel mapping."""
        await self.initialize()
        
        try:
            await self._mappings_table.delete_entity(
                partition_key="mapping",
                row_key=channel_name
            )
            
            # Invalidate cache
            await self._cache.invalidate("mappings")
            
            logger.info(f"Deleted channel mapping: {channel_name}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"Channel mapping not found: {channel_name}")
            return False
        except Exception as e:
            logger.error(f"Error deleting channel mapping: {e}")
            return False
    
    async def delete_pattern_mapping(self, pattern: str) -> bool:
        """Delete a pattern mapping."""
        await self.initialize()
        
        try:
            await self._patterns_table.delete_entity(
                partition_key="pattern",
                row_key=pattern
            )
            
            # Invalidate cache
            await self._cache.invalidate("patterns")
            
            logger.info(f"Deleted pattern mapping: {pattern}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"Pattern mapping not found: {pattern}")
            return False
        except Exception as e:
            logger.error(f"Error deleting pattern mapping: {e}")
            return False
    
    async def list_all_mappings(self) -> List[Dict[str, Any]]:
        """List all channel mappings."""
        await self.initialize()
        
        try:
            mappings = []
            
            # Get explicit mappings
            async for entity in self._mappings_table.list_entities():
                mappings.append({
                    "key": entity["RowKey"],
                    "channel_name": entity["RowKey"],
                    "client_id": entity.get("ClientId", ""),
                    "type": "explicit",
                    "description": entity.get("Description", ""),
                    "priority": entity.get("Priority", 1)
                })
            
            # Get pattern mappings
            async for entity in self._patterns_table.list_entities():
                pattern = entity["RowKey"]
                mappings.append({
                    "key": pattern,
                    "pattern": pattern,
                    "client_id": entity.get("ClientId", ""),
                    "type": "pattern",
                    "description": entity.get("Description", ""),
                    "priority": entity.get("Priority", 10)
                })
            
            return mappings
            
        except Exception as e:
            logger.error(f"Error listing mappings: {e}")
            return []
    
    async def list_all_clients(self) -> List[Dict[str, Any]]:
        """List all clients."""
        await self.initialize()
        
        try:
            clients = []
            async for entity in self._clients_table.list_entities():
                clients.append({
                    "client_id": entity["RowKey"],
                    "client_name": entity.get("ClientName", ""),
                    "industry": entity.get("Industry", ""),
                    "tier": entity.get("Tier", ""),
                    "monthly_revenue": entity.get("MonthlyRevenue", 0),
                    "account_manager": entity.get("AccountManager", "")
                })
            
            return clients
            
        except Exception as e:
            logger.error(f"Error listing clients: {e}")
            return []
    
    async def refresh_cache(self):
        """Refresh all caches."""
        await self._cache.invalidate()
        logger.info("Client storage cache refreshed")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "mappings_cached": len(self._cache._mappings_cache),
            "clients_cached": len(self._cache._clients_cache),
            "patterns_cached": len(self._cache._patterns_cache),
            "ttl_seconds": self._cache.ttl_seconds
        }


# Global instance
_client_storage_service: Optional[ClientStorageService] = None


async def get_client_storage_service() -> ClientStorageService:
    """Get or create global client storage service instance."""
    global _client_storage_service
    
    if _client_storage_service is None:
        _client_storage_service = ClientStorageService()
        await _client_storage_service.initialize()
    
    return _client_storage_service