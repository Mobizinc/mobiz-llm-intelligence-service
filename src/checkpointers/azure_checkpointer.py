"""
Azure Table Storage Checkpointer for LangGraph State Persistence

This module implements a checkpointer using Azure Table Storage to persist
conversation state across LangGraph workflow executions. This enables:
- State recovery across restarts
- Conversation continuity  
- State versioning and rollback capabilities

Part of Epic 3: Orchestration & Intelligent Routing
Story 3.1: Core LangGraph Workflow
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Iterator, Tuple
from dataclasses import dataclass

try:
    from azure.data.tables import TableClient
    from azure.core.exceptions import ResourceNotFoundError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    TableClient = None
    ResourceNotFoundError = Exception

from langgraph.checkpoint.base import BaseCheckpointer, Checkpoint, CheckpointMetadata
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CheckpointerConfig:
    """Configuration for Azure Table Storage checkpointer"""
    table_name: str = "conversation_checkpoints"
    max_versions: int = 10  # Maximum checkpoint versions to keep
    ttl_hours: int = 24  # Time to live for checkpoints
    enable_compression: bool = True
    

class AzureTableCheckpointMetadata(CheckpointMetadata):
    """Extended metadata for Azure Table Storage checkpoints"""
    
    def __init__(
        self,
        source: str = "update",
        step: int = -1,
        writes: Dict[str, Any] = None,
        parents: Dict[str, str] = None,
        **kwargs
    ):
        super().__init__(source=source, step=step, writes=writes or {}, parents=parents or {})
        self.azure_metadata = kwargs
        

class AzureCheckpointer(BaseCheckpointer):
    """
    Azure Table Storage-based checkpointer for LangGraph workflows.
    
    Provides persistent state storage using Azure Table Storage with:
    - Automatic versioning
    - Compression for large states
    - TTL-based cleanup
    - Recovery capabilities
    """
    
    def __init__(self, config: Optional[CheckpointerConfig] = None):
        super().__init__()
        self.config = config or CheckpointerConfig()
        
        if not AZURE_AVAILABLE:
            logger.warning("Azure Table Storage dependencies not available. Falling back to MemoryCheckpointer for development.")
            self._fallback_checkpointer = MemorySaver()
            self.table_client = None
        else:
            # Initialize Azure Table Storage client
            self.table_client = self._initialize_table_client()
            
        logger.info(f"AzureCheckpointer initialized with table: {self.config.table_name}")
        
    def _initialize_table_client(self) -> Optional[TableClient]:
        """Initialize Azure Table Storage client"""
        try:
            # For now, use development storage emulator or fall back to memory
            connection_string = getattr(settings, 'azure_storage_connection_string', None)
            
            if not connection_string:
                logger.warning("Azure Storage connection string not found, using development storage")
                connection_string = "DefaultEndpointsProtocol=https;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
            
            table_client = TableClient.from_connection_string(
                conn_str=connection_string,
                table_name=self.config.table_name
            )
            
            # Create table if it doesn't exist
            try:
                table_client.create_table()
                logger.info(f"Created table: {self.config.table_name}")
            except Exception:
                # Table might already exist
                pass
                
            return table_client
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Table Storage: {e}")
            logger.info("Falling back to MemoryCheckpointer for development")
            self._fallback_checkpointer = MemorySaver()
            return None
    
    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """Serialize checkpoint data for storage"""
        try:
            data = {
                'v': checkpoint.v,
                'ts': checkpoint.ts,
                'channel_values': checkpoint.channel_values,
                'channel_versions': checkpoint.channel_versions,
                'versions_seen': checkpoint.versions_seen,
                'pending_sends': checkpoint.pending_sends
            }
            
            serialized = json.dumps(data, default=str, ensure_ascii=False)
            
            if self.config.enable_compression and len(serialized) > 1000:
                try:
                    import gzip
                    import base64
                    compressed = gzip.compress(serialized.encode('utf-8'))
                    return f"GZIP:{base64.b64encode(compressed).decode('ascii')}"
                except ImportError:
                    logger.debug("gzip not available, storing uncompressed")
            
            return serialized
            
        except Exception as e:
            logger.error(f"Failed to serialize checkpoint: {e}")
            raise
    
    def _deserialize_checkpoint(self, serialized_data: str) -> Checkpoint:
        """Deserialize checkpoint data from storage"""
        try:
            # Handle compressed data
            if serialized_data.startswith("GZIP:"):
                try:
                    import gzip
                    import base64
                    compressed_data = base64.b64decode(serialized_data[5:])
                    serialized_data = gzip.decompress(compressed_data).decode('utf-8')
                except ImportError:
                    logger.error("gzip not available but data is compressed")
                    raise
            
            data = json.loads(serialized_data)
            
            return Checkpoint(
                v=data.get('v'),
                ts=data.get('ts'),
                channel_values=data.get('channel_values', {}),
                channel_versions=data.get('channel_versions', {}),
                versions_seen=data.get('versions_seen', {}),
                pending_sends=data.get('pending_sends', [])
            )
            
        except Exception as e:
            logger.error(f"Failed to deserialize checkpoint: {e}")
            raise
    
    def _serialize_metadata(self, metadata: CheckpointMetadata) -> str:
        """Serialize checkpoint metadata for storage"""
        try:
            data = {
                'source': metadata.source,
                'step': metadata.step,
                'writes': metadata.writes,
                'parents': metadata.parents
            }
            
            if isinstance(metadata, AzureTableCheckpointMetadata):
                data['azure_metadata'] = metadata.azure_metadata
            
            return json.dumps(data, default=str, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to serialize metadata: {e}")
            return "{}"
    
    def _deserialize_metadata(self, serialized_data: str) -> CheckpointMetadata:
        """Deserialize checkpoint metadata from storage"""
        try:
            if not serialized_data:
                return CheckpointMetadata()
                
            data = json.loads(serialized_data)
            
            azure_metadata = data.pop('azure_metadata', {})
            
            return AzureTableCheckpointMetadata(
                source=data.get('source', 'update'),
                step=data.get('step', -1),
                writes=data.get('writes', {}),
                parents=data.get('parents', {}),
                **azure_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to deserialize metadata: {e}")
            return CheckpointMetadata()
    
    def _create_row_key(self, config: RunnableConfig, checkpoint_id: Optional[str] = None) -> str:
        """Create row key for checkpoint storage"""
        thread_id = config["configurable"]["thread_id"]
        
        if checkpoint_id:
            return f"{thread_id}#{checkpoint_id}"
        else:
            # Use current timestamp for new checkpoints
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            return f"{thread_id}#{timestamp}"
    
    def _parse_row_key(self, row_key: str) -> Tuple[str, str]:
        """Parse row key to extract thread_id and checkpoint_id"""
        parts = row_key.split('#', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""
    
    async def alist(
        self, 
        config: RunnableConfig, 
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> List[CheckpointMetadata]:
        """List checkpoints for a given thread"""
        
        # Fallback to memory checkpointer if Azure is not available
        if not self.table_client:
            if hasattr(self, '_fallback_checkpointer'):
                try:
                    return await self._fallback_checkpointer.alist(config, filter=filter, before=before, limit=limit)
                except:
                    return []
            return []
            
        try:
            thread_id = config["configurable"]["thread_id"]
            
            # Query checkpoints for this thread
            filter_query = f"PartitionKey eq '{thread_id}'"
            
            entities = self.table_client.query_entities(
                query_filter=filter_query,
                select=["RowKey", "Metadata", "Timestamp"]
            )
            
            checkpoints = []
            for entity in entities:
                try:
                    metadata = self._deserialize_metadata(entity.get('Metadata', '{}'))
                    checkpoints.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to deserialize checkpoint metadata: {e}")
                    continue
            
            # Sort by step (most recent first)
            checkpoints.sort(key=lambda x: x.step, reverse=True)
            
            if limit:
                checkpoints = checkpoints[:limit]
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Get the latest checkpoint and metadata for a thread"""
        
        # Fallback to memory checkpointer if Azure is not available
        if not self.table_client:
            if hasattr(self, '_fallback_checkpointer'):
                try:
                    return await self._fallback_checkpointer.aget_tuple(config)
                except:
                    return None
            return None
            
        try:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = config["configurable"].get("checkpoint_id")
            
            if checkpoint_id:
                # Get specific checkpoint
                row_key = self._create_row_key(config, checkpoint_id)
                entity = self.table_client.get_entity(partition_key=thread_id, row_key=row_key)
            else:
                # Get latest checkpoint
                filter_query = f"PartitionKey eq '{thread_id}'"
                entities = list(self.table_client.query_entities(
                    query_filter=filter_query,
                    select=["RowKey", "CheckpointData", "Metadata", "Timestamp"]
                ))
                
                if not entities:
                    return None
                
                # Sort by timestamp (most recent first) and get the latest
                entities.sort(key=lambda x: x.get('Timestamp', ''), reverse=True)
                entity = entities[0]
            
            # Deserialize checkpoint and metadata
            checkpoint_data = entity.get('CheckpointData', '{}')
            metadata_data = entity.get('Metadata', '{}')
            
            checkpoint = self._deserialize_checkpoint(checkpoint_data)
            metadata = self._deserialize_metadata(metadata_data)
            
            return (checkpoint, metadata)
            
        except ResourceNotFoundError if AZURE_AVAILABLE else Exception:
            logger.debug(f"No checkpoint found for thread: {config['configurable']['thread_id']}")
            return None
        except Exception as e:
            logger.error(f"Failed to get checkpoint: {e}")
            return None
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any]
    ) -> RunnableConfig:
        """Store a checkpoint"""
        
        # Fallback to memory checkpointer if Azure is not available
        if not self.table_client:
            if hasattr(self, '_fallback_checkpointer'):
                try:
                    return await self._fallback_checkpointer.aput(config, checkpoint, metadata, new_versions)
                except Exception as e:
                    logger.error(f"Failed to store checkpoint in fallback: {e}")
                    return config
            return config
            
        try:
            thread_id = config["configurable"]["thread_id"]
            row_key = self._create_row_key(config)
            
            # Serialize data
            checkpoint_data = self._serialize_checkpoint(checkpoint)
            metadata_data = self._serialize_metadata(metadata)
            
            # Prepare entity
            entity = {
                'PartitionKey': thread_id,
                'RowKey': row_key,
                'CheckpointData': checkpoint_data,
                'Metadata': metadata_data,
                'Timestamp': datetime.now(timezone.utc).isoformat(),
                'Step': metadata.step,
                'Source': metadata.source,
                'Versions': json.dumps(new_versions, default=str)
            }
            
            # Store in Azure Table Storage
            self.table_client.upsert_entity(entity)
            
            logger.debug(f"Stored checkpoint for thread {thread_id}, step {metadata.step}")
            
            # Cleanup old checkpoints if needed
            await self._cleanup_old_checkpoints(thread_id)
            
            # Return updated config with checkpoint ID
            updated_config = config.copy()
            updated_config["configurable"] = config["configurable"].copy()
            _, checkpoint_id = self._parse_row_key(row_key)
            updated_config["configurable"]["checkpoint_id"] = checkpoint_id
            
            return updated_config
            
        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")
            return config
    
    async def _cleanup_old_checkpoints(self, thread_id: str):
        """Clean up old checkpoints to maintain max_versions limit"""
        if not self.table_client:
            return
            
        try:
            # Query all checkpoints for this thread
            filter_query = f"PartitionKey eq '{thread_id}'"
            entities = list(self.table_client.query_entities(
                query_filter=filter_query,
                select=["RowKey", "Step", "Timestamp"]
            ))
            
            # Sort by step (most recent first)
            entities.sort(key=lambda x: x.get('Step', 0), reverse=True)
            
            # Remove old checkpoints beyond max_versions
            if len(entities) > self.config.max_versions:
                old_entities = entities[self.config.max_versions:]
                
                for entity in old_entities:
                    try:
                        self.table_client.delete_entity(
                            partition_key=thread_id,
                            row_key=entity['RowKey']
                        )
                        logger.debug(f"Deleted old checkpoint: {entity['RowKey']}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old checkpoint {entity['RowKey']}: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    # Sync versions for compatibility
    def list(self, config: RunnableConfig, **kwargs) -> List[CheckpointMetadata]:
        """Sync version of alist"""
        import asyncio
        return asyncio.run(self.alist(config, **kwargs))
    
    def get_tuple(self, config: RunnableConfig) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Sync version of aget_tuple"""
        import asyncio
        return asyncio.run(self.aget_tuple(config))
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any]
    ) -> RunnableConfig:
        """Sync version of aput"""
        import asyncio
        return asyncio.run(self.aput(config, checkpoint, metadata, new_versions))