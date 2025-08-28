"""
Client Context Service

This service is responsible for loading and merging client data from YAML files
and dynamic JSON notes, providing rich client context to the reasoning nodes.
"""

import asyncio
import json
import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ClientContextService:
    """
    Service for managing client context data from YAML configurations and JSON notes.
    
    Supports:
    - Loading client data from YAML configuration files
    - Loading dynamic notes from JSON files
    - Merging and caching client context
    - Client note management (add, view, delete)
    """
    
    def __init__(self, config_path: Optional[str] = None, notes_path: Optional[str] = None):
        """
        Initialize the client context service.
        
        Args:
            config_path: Path to clients.yaml file
            notes_path: Path to client_notes directory
        """
        # Set default paths relative to the project root
        project_root = Path(__file__).parent.parent.parent
        self.config_path = Path(config_path) if config_path else project_root / "config" / "clients.yaml"
        self.notes_path = Path(notes_path) if notes_path else project_root / "config" / "client_notes"
        
        # Create directories if they don't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for client contexts
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 300  # 5 minutes cache TTL
    
    async def load_client_context(self, client_id: str) -> Dict[str, Any]:
        """
        Load and merge client context from YAML configuration and JSON notes.
        
        Args:
            client_id: The client identifier
            
        Returns:
            Dict containing merged client context data
        """
        try:
            # Check cache first
            if self._is_cache_valid(client_id):
                logger.debug(f"Using cached context for client {client_id}")
                return self._context_cache[client_id]
            
            logger.info(f"Loading context for client {client_id}")
            
            # Load base configuration from YAML
            yaml_context = await self._load_yaml_context(client_id)
            
            # Load dynamic notes from JSON files
            json_notes = await self._load_json_notes(client_id)
            
            # Merge contexts
            merged_context = self._merge_contexts(yaml_context, json_notes)
            
            # Add metadata
            merged_context["_metadata"] = {
                "client_id": client_id,
                "loaded_at": datetime.now().isoformat(),
                "sources": {
                    "yaml_loaded": bool(yaml_context),
                    "notes_loaded": bool(json_notes),
                    "total_notes": len(json_notes.get("notes", [])) if json_notes else 0
                }
            }
            
            # Cache the result
            self._context_cache[client_id] = merged_context
            self._cache_timestamps[client_id] = datetime.now()
            
            logger.info(
                f"Client context loaded for {client_id}",
                extra={
                    "client_id": client_id,
                    "yaml_loaded": bool(yaml_context),
                    "notes_count": len(json_notes.get("notes", [])) if json_notes else 0,
                    "context_keys": list(merged_context.keys())
                }
            )
            
            return merged_context
            
        except Exception as e:
            logger.error(
                f"Error loading client context for {client_id}: {e}",
                extra={
                    "client_id": client_id,
                    "exception_type": type(e).__name__
                },
                exc_info=True
            )
            
            # Return minimal context on error
            return {
                "client_id": client_id,
                "error": f"Failed to load context: {str(e)}",
                "_metadata": {
                    "client_id": client_id,
                    "loaded_at": datetime.now().isoformat(),
                    "error": True
                }
            }
    
    async def _load_yaml_context(self, client_id: str) -> Dict[str, Any]:
        """Load client context from the YAML configuration file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Clients YAML file not found: {self.config_path}")
                await self._create_default_yaml_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}
            
            clients = yaml_data.get('clients', {})
            client_data = clients.get(client_id, {})
            
            if client_data:
                logger.debug(f"Loaded YAML context for {client_id}")
            else:
                logger.debug(f"No YAML context found for {client_id}")
            
            return client_data
            
        except Exception as e:
            logger.error(f"Error loading YAML context: {e}")
            return {}
    
    async def _load_json_notes(self, client_id: str) -> Dict[str, Any]:
        """Load client notes from JSON files."""
        try:
            notes_file = self.notes_path / f"{client_id}.json"
            
            if not notes_file.exists():
                logger.debug(f"No notes file found for {client_id}")
                return {}
            
            with open(notes_file, 'r', encoding='utf-8') as f:
                notes_raw = json.load(f)
            
            # Handle different note file formats
            if isinstance(notes_raw, list):
                # Legacy format: direct array of notes
                notes_data = {
                    "client_id": client_id,
                    "notes": notes_raw,
                    "format": "legacy"
                }
            elif isinstance(notes_raw, dict):
                # New format: structured with metadata
                notes_data = notes_raw
            else:
                logger.warning(f"Unexpected notes format for {client_id}")
                return {}
            
            note_count = len(notes_data.get('notes', []))
            logger.debug(f"Loaded {note_count} notes for {client_id}")
            return notes_data
            
        except Exception as e:
            logger.error(f"Error loading JSON notes for {client_id}: {e}")
            return {}
    
    def _merge_contexts(self, yaml_context: Dict[str, Any], json_notes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge YAML configuration and JSON notes into a unified context.
        
        JSON notes take precedence over YAML configuration for overlapping keys.
        """
        merged = yaml_context.copy()
        
        if json_notes:
            # Merge top-level fields from JSON notes
            for key, value in json_notes.items():
                if key == "notes":
                    # Special handling for notes array
                    merged["notes"] = value
                elif key == "updated_at":
                    # Keep track of when notes were last updated
                    merged["notes_updated_at"] = value
                else:
                    # Override YAML values with JSON values
                    merged[key] = value
        
        return merged
    
    def _is_cache_valid(self, client_id: str) -> bool:
        """Check if cached context is still valid."""
        if client_id not in self._context_cache:
            return False
        
        if client_id not in self._cache_timestamps:
            return False
        
        cache_age = (datetime.now() - self._cache_timestamps[client_id]).total_seconds()
        return cache_age < self._cache_ttl_seconds
    
    async def add_client_note(
        self, 
        client_id: str, 
        note_content: str, 
        category: str = "general",
        author: str = "system"
    ) -> Dict[str, Any]:
        """
        Add a new note for a client.
        
        Args:
            client_id: The client identifier
            note_content: The note content
            category: Note category (e.g., "technical", "business", "contact")
            author: Who added the note
            
        Returns:
            Dict containing the operation result
        """
        try:
            notes_file = self.notes_path / f"{client_id}.json"
            
            # Load existing notes or create new structure
            if notes_file.exists():
                with open(notes_file, 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
            else:
                notes_data = {
                    "client_id": client_id,
                    "created_at": datetime.now().isoformat(),
                    "notes": []
                }
            
            # Add new note
            new_note = {
                "id": f"note_{datetime.now().timestamp()}",
                "content": note_content,
                "category": category,
                "author": author,
                "created_at": datetime.now().isoformat()
            }
            
            notes_data["notes"].append(new_note)
            notes_data["updated_at"] = datetime.now().isoformat()
            
            # Save updated notes
            with open(notes_file, 'w', encoding='utf-8') as f:
                json.dump(notes_data, f, indent=2, ensure_ascii=False)
            
            # Invalidate cache
            if client_id in self._context_cache:
                del self._context_cache[client_id]
                del self._cache_timestamps[client_id]
            
            logger.info(
                f"Added note for client {client_id}",
                extra={
                    "client_id": client_id,
                    "category": category,
                    "author": author,
                    "note_id": new_note["id"]
                }
            )
            
            return {
                "success": True,
                "note_id": new_note["id"],
                "message": f"Note added for client {client_id}"
            }
            
        except Exception as e:
            logger.error(f"Error adding note for client {client_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to add note for client {client_id}"
            }
    
    async def view_client_notes(self, client_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        View recent notes for a client.
        
        Args:
            client_id: The client identifier
            limit: Maximum number of notes to return
            
        Returns:
            Dict containing the client's notes
        """
        try:
            notes_file = self.notes_path / f"{client_id}.json"
            
            if not notes_file.exists():
                return {
                    "success": True,
                    "client_id": client_id,
                    "notes": [],
                    "message": f"No notes found for client {client_id}"
                }
            
            with open(notes_file, 'r', encoding='utf-8') as f:
                notes_data = json.load(f)
            
            # Sort notes by creation date (most recent first) and limit
            notes = notes_data.get("notes", [])
            sorted_notes = sorted(
                notes, 
                key=lambda x: x.get("created_at", ""), 
                reverse=True
            )[:limit]
            
            return {
                "success": True,
                "client_id": client_id,
                "notes": sorted_notes,
                "total_notes": len(notes),
                "showing": len(sorted_notes),
                "last_updated": notes_data.get("updated_at"),
                "message": f"Found {len(sorted_notes)} recent notes for client {client_id}"
            }
            
        except Exception as e:
            logger.error(f"Error viewing notes for client {client_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to view notes for client {client_id}"
            }
    
    async def delete_client_note(self, client_id: str, note_id: str) -> Dict[str, Any]:
        """
        Delete a specific note for a client.
        
        Args:
            client_id: The client identifier
            note_id: The note ID to delete
            
        Returns:
            Dict containing the operation result
        """
        try:
            notes_file = self.notes_path / f"{client_id}.json"
            
            if not notes_file.exists():
                return {
                    "success": False,
                    "message": f"No notes file found for client {client_id}"
                }
            
            with open(notes_file, 'r', encoding='utf-8') as f:
                notes_data = json.load(f)
            
            # Find and remove the note
            notes = notes_data.get("notes", [])
            original_count = len(notes)
            
            notes_data["notes"] = [note for note in notes if note.get("id") != note_id]
            new_count = len(notes_data["notes"])
            
            if original_count == new_count:
                return {
                    "success": False,
                    "message": f"Note {note_id} not found for client {client_id}"
                }
            
            # Update timestamp
            notes_data["updated_at"] = datetime.now().isoformat()
            
            # Save updated notes
            with open(notes_file, 'w', encoding='utf-8') as f:
                json.dump(notes_data, f, indent=2, ensure_ascii=False)
            
            # Invalidate cache
            if client_id in self._context_cache:
                del self._context_cache[client_id]
                del self._cache_timestamps[client_id]
            
            logger.info(
                f"Deleted note {note_id} for client {client_id}",
                extra={
                    "client_id": client_id,
                    "note_id": note_id
                }
            )
            
            return {
                "success": True,
                "message": f"Note {note_id} deleted for client {client_id}"
            }
            
        except Exception as e:
            logger.error(f"Error deleting note for client {client_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete note for client {client_id}"
            }
    
    async def _create_default_yaml_config(self):
        """Create a default clients.yaml configuration file."""
        try:
            default_config = {
                "clients": {
                    "example-client": {
                        "name": "Example Client",
                        "industry": "Technology",
                        "size": "Medium",
                        "technologies": ["Azure", "Microsoft 365"],
                        "contact": {
                            "primary": "admin@example-client.com",
                            "technical": "tech@example-client.com"
                        },
                        "preferences": {
                            "communication_style": "technical",
                            "response_detail": "comprehensive"
                        },
                        "sla": {
                            "response_time": "4 hours",
                            "resolution_time": "24 hours"
                        }
                    }
                }
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Created default clients configuration at {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error creating default YAML config: {e}")
    
    async def clear_cache(self, client_id: Optional[str] = None):
        """
        Clear the context cache for a specific client or all clients.
        
        Args:
            client_id: Optional specific client ID to clear. If None, clears all cache.
        """
        if client_id:
            if client_id in self._context_cache:
                del self._context_cache[client_id]
            if client_id in self._cache_timestamps:
                del self._cache_timestamps[client_id]
            logger.debug(f"Cleared cache for client {client_id}")
        else:
            self._context_cache.clear()
            self._cache_timestamps.clear()
            logger.debug("Cleared all client context cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        now = datetime.now()
        valid_entries = sum(
            1 for client_id in self._context_cache
            if self._is_cache_valid(client_id)
        )
        
        return {
            "total_cached": len(self._context_cache),
            "valid_cached": valid_entries,
            "expired_cached": len(self._context_cache) - valid_entries,
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None
        }