"""
Dynamic MCP Manager for handling multiple MCP server connections.
Provides scalable, configuration-driven MCP integration without hardcoding servers or tools.
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ..core.config import settings
from ..core.logging_config import get_logger

logger = get_logger(__name__)

# Import MCP components
try:
    from fastmcp import Client
    from mcp.client.sse import create_mcp_http_client
    from mcp.client.stdio import stdio_client
    import httpx
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_AVAILABLE = False
    Client = None
    create_mcp_http_client = None
    stdio_client = None
    httpx = None
    logger.warning(f"MCP import failed: {e}")

# Import telemetry if available
try:
    from ..shared.telemetry import EnhancedTelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    EnhancedTelemetryManager = None


class MCPToolError(Exception):
    """Error calling MCP tool"""
    def __init__(self, message: str, server_name: str = None, tool_name: str = None):
        super().__init__(message)
        self.server_name = server_name
        self.tool_name = tool_name


class MCPManager:
    """Manages multiple MCP server connections dynamically"""
    
    def __init__(self):
        self.config = {}
        self.clients: Dict[str, Any] = {}  # server_name -> Client instance
        self.tools_registry: Dict[str, Dict] = {}  # Complete tool registry
        self.server_metadata: Dict[str, Dict] = {}  # Server capabilities and restrictions
        self.server_health: Dict[str, str] = {}  # Server health status
        self._refresh_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._last_config_load = 0
        
        # Check if MCP is available
        if not MCP_AVAILABLE:
            logger.warning("MCP packages not available, MCP functionality disabled")
    
    async def initialize(self) -> bool:
        """Initialize all configured MCP servers"""
        logger.info("Starting MCP Manager initialization")
        
        if not MCP_AVAILABLE:
            logger.warning("Cannot initialize MCP manager - MCP packages not available")
            logger.debug("Missing packages: fastmcp, mcp.client.sse, mcp.client.stdio, httpx")
            return False
        
        if not settings.mcp_global_enabled:
            logger.info("MCP globally disabled via configuration (mcp_global_enabled=False)")
            return False
        
        start_time = time.time()
        
        try:
            logger.debug("Step 1: Loading MCP configuration")
            await self._load_configuration()
            
            logger.debug("Step 2: Discovering and connecting to servers")
            await self._discover_and_connect_servers()
            
            logger.debug("Step 3: Discovering all available tools")
            await self._discover_all_tools()
            
            # Start refresh task
            if settings.mcp_dynamic_discovery:
                logger.debug("Step 4: Starting dynamic discovery refresh task")
                self._start_refresh_task()
            else:
                logger.debug("Step 4: Skipping refresh task (dynamic discovery disabled)")
            
            self._initialized = True
            initialization_time = time.time() - start_time
            
            logger.info(
                f"MCP Manager initialization completed successfully",
                extra={
                    "servers_connected": len(self.clients),
                    "tools_discovered": len(self.tools_registry),
                    "initialization_time": f"{initialization_time:.2f}s",
                    "dynamic_discovery_enabled": settings.mcp_dynamic_discovery
                }
            )
            
            # Log summary of available servers and tools
            if self.clients:
                server_names = list(self.clients.keys())
                logger.info(f"Connected servers: {', '.join(server_names)}")
            
            if self.tools_registry:
                logger.debug(f"Available tools: {', '.join(self.tools_registry.keys())}")
            
            return True
            
        except Exception as e:
            initialization_time = time.time() - start_time
            logger.error(
                f"Failed to initialize MCP manager after {initialization_time:.2f}s: {e}",
                extra={
                    "initialization_time": f"{initialization_time:.2f}s",
                    "servers_attempted": len(self.config.get('mcpServers', {})),
                    "servers_connected": len(self.clients)
                },
                exc_info=True
            )
            return False
    
    async def _load_configuration(self):
        """Load MCP configuration from file and environment"""
        try:
            config_path = Path(settings.mcp_config_path)
            
            if not config_path.exists():
                logger.error(f"MCP configuration file not found: {config_path}")
                self.config = {"global_settings": {}, "mcpServers": {}}
                return
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Apply environment overrides
            self._apply_environment_overrides()
            
            # Apply secrets from security manager if available
            await self._apply_security_overrides()
            
            self._last_config_load = time.time()
            logger.info(f"Loaded MCP configuration with {len(self.config.get('mcpServers', {}))} servers")
            
        except Exception as e:
            logger.error(f"Failed to load MCP configuration: {e}", exc_info=True)
            self.config = {"global_settings": {}, "mcpServers": {}}
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides to configuration"""
        # Override global settings from environment
        global_settings = self.config.setdefault("global_settings", {})
        
        if hasattr(settings, 'mcp_tool_refresh_interval'):
            global_settings["tool_refresh_interval"] = settings.mcp_tool_refresh_interval
        if hasattr(settings, 'mcp_default_timeout'):
            global_settings["default_timeout"] = settings.mcp_default_timeout
        if hasattr(settings, 'mcp_max_concurrent_requests'):
            global_settings["max_concurrent_requests"] = settings.mcp_max_concurrent_requests
    
    async def _apply_security_overrides(self):
        """Apply secrets from security manager if available"""
        try:
            from ..core.singletons import get_security_manager
            security_manager = get_security_manager()
            
            # Check for MCP-specific secrets in Key Vault
            for server_name in self.config.get("mcpServers", {}):
                # Look for server-specific API keys or tokens
                api_key = security_manager.get_secret(f"mcp-{server_name}-api-key", default=None)
                if api_key:
                    self.config["mcpServers"][server_name].setdefault("auth", {})["api_key"] = api_key
                    
        except Exception as e:
            logger.debug(f"Could not load MCP secrets from security manager: {e}")
    
    async def _discover_and_connect_servers(self):
        """Dynamically connect to all enabled servers from config"""
        servers = self.config.get("mcpServers", {})
        
        if not servers:
            logger.warning("No MCP servers configured")
            return False
        
        # Connect to servers concurrently
        connection_tasks = []
        server_names = []
        
        for server_name, server_config in servers.items():
            if server_config.get("enabled", True):
                task = asyncio.create_task(
                    self._connect_to_server(server_name, server_config)
                )
                connection_tasks.append(task)
                server_names.append(server_name)
        
        # Wait for all connections
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # Log results
        successful_connections = 0
        for i, result in enumerate(results):
            server_name = server_names[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to connect to {server_name}: {result}")
                self.server_health[server_name] = f"connection_failed: {result}"
            elif result:  # result is True for successful connection
                successful_connections += 1
                self.server_health[server_name] = "connected"
            else:
                logger.warning(f"Connection to {server_name} returned False")
                self.server_health[server_name] = "connection_failed"
        
        logger.info(f"Successfully connected to {successful_connections}/{len(connection_tasks)} MCP servers")
        return successful_connections > 0
    
    async def _connect_to_server(self, server_name: str, server_config: Dict[str, Any]):
        """Connect to a single MCP server based on configuration"""
        start_time = time.time()
        
        try:
            transport_type = server_config.get("transport", "http")
            
            if transport_type == "http":
                url = server_config.get("url")
                if not url:
                    raise ValueError(f"HTTP transport requires 'url' for server {server_name}")
                
                # Use FastMCP Client with direct URL
                timeout = server_config.get("settings", {}).get("timeout", settings.mcp_default_timeout)
                
                # Create client instance (don't enter context yet)
                client = Client(transport=url, timeout=timeout)
                
                # Test connection using proper async with pattern
                async with client:
                    # Test connection by listing tools
                    tools = await client.list_tools()
                    tools_count = len(tools) if isinstance(tools, list) else len(getattr(tools, 'tools', []))
                    
                    # Store un-entered client instance and metadata after successful test
                    self.clients[server_name] = client
                    self.server_metadata[server_name] = {
                        **server_config.get("settings", {}),
                        "url": url,
                        "transport": transport_type,
                        "tools_count": tools_count,
                        "domains": server_config.get("domains", [])
                    }
                    
                    # Log telemetry for successful connection
                    duration = time.time() - start_time
                    if TELEMETRY_AVAILABLE:
                        telemetry = EnhancedTelemetryManager()
                        telemetry.log_custom_event(
                            "mcp.connection.lifecycle",
                            {
                                "server_name": server_name,
                                "status": "success",
                                "duration": duration,
                                "tools_count": tools_count,
                                "transport": transport_type
                            }
                        )
                    
                    logger.info(f"Successfully connected to MCP server {server_name} with {tools_count} tools")
                    return True
                
            else:
                logger.warning(f"Unsupported transport type '{transport_type}' for server {server_name}")
                return False
            
        except Exception as e:
            # Log telemetry for failed connection
            duration = time.time() - start_time
            if TELEMETRY_AVAILABLE:
                telemetry = EnhancedTelemetryManager()
                telemetry.log_custom_event(
                    "mcp.connection.lifecycle",
                    {
                        "server_name": server_name,
                        "status": "failed",
                        "duration": duration,
                        "error": str(e),
                        "transport": transport_type
                    }
                )
            
            logger.error(f"Failed to connect to MCP server {server_name}: {e}")
            return False
    
    async def _discover_all_tools(self):
        """Discover tools from all connected servers"""
        discovery_tasks = []
        
        for server_name in self.clients.keys():
            task = asyncio.create_task(
                self._discover_server_tools(server_name)
            )
            discovery_tasks.append(task)
        
        # Wait for all discoveries
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        # Log results
        total_tools = 0
        for i, result in enumerate(results):
            server_name = list(self.clients.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to discover tools from {server_name}: {result}")
            else:
                server_tools = [tool_id for tool_id in self.tools_registry.keys() 
                              if tool_id.startswith(f"{server_name}.")]
                total_tools += len(server_tools)
                logger.info(f"Discovered {len(server_tools)} tools from {server_name}")
        
        logger.info(f"Total tools discovered: {total_tools}")
    
    async def _discover_server_tools(self, server_name: str):
        """Discover all tools from a specific server"""
        try:
            client = self.clients[server_name]
            server_config = self.config["mcpServers"][server_name]
            server_settings = server_config.get("settings", {})
            
            # Use async with context manager for tool discovery
            async with client:
                tools_result = await client.list_tools()
                
                # Handle the tools list correctly - FastMCP returns list directly
                tools_list = tools_result if isinstance(tools_result, list) else getattr(tools_result, 'tools', [])
                
                for tool in tools_list:
                    tool_id = f"{server_name}.{tool.name}"
                    self.tools_registry[tool_id] = {
                        "server": server_name,
                        "tool": tool.name,
                        "description": tool.description,
                        "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                        "restrictions": server_settings,
                        "domains": server_config.get("domains", []),
                        "tags": server_config.get("tags", []),
                        "discovered_at": time.time()
                    }
                
                logger.debug(f"Discovered {len(tools_list)} tools from {server_name}")
            
        except Exception as e:
            logger.error(f"Failed to discover tools from {server_name}: {e}")
            raise
    
    def _unwrap_result(self, result):
        """Unwrap MCP result properly handling FastMCP's structure"""
        logger.info(f"Unwrapping result type: {type(result)}")
        
        # Handle FastMCP CallToolResult structure
        if hasattr(result, 'content') and isinstance(result.content, list):
            logger.info(f"Result has content list with {len(result.content)} items")
            if result.content and hasattr(result.content[0], 'text'):
                # Extract text from first content item
                text_content = result.content[0].text
                logger.info(f"Extracted text content, length: {len(text_content)}")
                
                # Try to parse as JSON if it looks like JSON
                if isinstance(text_content, str) and (text_content.startswith('[') or text_content.startswith('{')):
                    try:
                        parsed = json.loads(text_content)
                        logger.info(f"Successfully parsed JSON, type: {type(parsed)}")
                        return parsed
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse as JSON: {e}, returning raw text")
                        return text_content
                else:
                    return text_content
        
        # Fallback to other unwrapping methods
        if hasattr(result, 'data'):
            logger.info(f"Using result.data: {type(result.data)}")
            return result.data
        elif hasattr(result, 'content'):
            # This would be for non-list content
            logger.info(f"Using result.content directly: {type(result.content)}")
            return result.content
        else:
            logger.info(f"Using result directly: {result}")
            return result
    
    async def call_tool(self, tool_identifier: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on any connected server"""
        if not self._initialized:
            raise RuntimeError("MCP Manager not initialized")
        
        if tool_identifier not in self.tools_registry:
            raise ValueError(f"Tool not found: {tool_identifier}")
        
        tool_info = self.tools_registry[tool_identifier]
        server_name = tool_info["server"]
        tool_name = tool_info["tool"]
        
        if server_name not in self.clients:
            raise RuntimeError(f"Server not connected: {server_name}")
        
        start_time = time.time()
        
        try:
            client = self.clients[server_name]
            
            # Apply restrictions
            restrictions = tool_info["restrictions"]
            timeout = restrictions.get("timeout", settings.mcp_default_timeout)
            
            # Use async with context manager for tool call
            async with client:
                # Call tool with timeout
                result = await asyncio.wait_for(
                    client.call_tool(tool_name, arguments),
                    timeout=timeout
                )
                
                # Check for error using is_error property
                if hasattr(result, 'is_error') and result.is_error:
                    error_message = getattr(result, 'error_message', str(result))
                    
                    # Log telemetry for tool error
                    if TELEMETRY_AVAILABLE:
                        telemetry = EnhancedTelemetryManager()
                        telemetry.log_custom_event(
                            "mcp.tool.error",
                            {
                                "tool_identifier": tool_identifier,
                                "server_name": server_name,
                                "tool_name": tool_name,
                                "error_type": "tool_error",
                                "error_message": error_message
                            }
                        )
                    
                    logger.error(f"MCP Tool error for {tool_identifier}: {error_message}")
                    raise MCPToolError(error_message, server_name=server_name, tool_name=tool_name)
                
                # Unwrap result using .data property
                unwrapped_result = self._unwrap_result(result)
                
                # Log telemetry for successful result processing
                duration = time.time() - start_time
                if TELEMETRY_AVAILABLE:
                    data_type = type(unwrapped_result).__name__
                    data_size = len(str(unwrapped_result)) if unwrapped_result else 0
                    telemetry = EnhancedTelemetryManager()
                    telemetry.log_custom_event(
                        "mcp.result.processed",
                        {
                            "tool_identifier": tool_identifier,
                            "server_name": server_name,
                            "data_type": data_type,
                            "size": data_size,
                            "duration": duration
                        }
                    )
                
                return unwrapped_result
            
        except MCPToolError:
            # Re-raise MCP tool errors as-is
            raise
        except Exception as e:
            # Log telemetry for general tool error
            if TELEMETRY_AVAILABLE:
                telemetry = EnhancedTelemetryManager()
                telemetry.log_custom_event(
                    "mcp.tool.error",
                    {
                        "tool_identifier": tool_identifier,
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "error_type": "exception",
                        "error_message": str(e)
                    }
                )
            
            logger.error(f"Failed to call tool {tool_identifier}: {e}")
            raise MCPToolError(f"Tool call failed: {str(e)}", server_name=server_name, tool_name=tool_name)
    
    async def search_tools_by_domain(self, domain: str) -> List[str]:
        """Get tools available for a specific domain"""
        matching_tools = []
        
        for tool_id, tool_info in self.tools_registry.items():
            if domain in tool_info.get("domains", []):
                matching_tools.append(tool_id)
        
        return matching_tools
    
    async def search_tools_by_query(self, query: str, domain: Optional[str] = None) -> List[str]:
        """Search tools by description content"""
        matching_tools = []
        query_lower = query.lower()
        
        for tool_id, tool_info in self.tools_registry.items():
            # Check domain filter
            if domain and domain not in tool_info.get("domains", []):
                continue
                
            # Search in description and tags
            description = tool_info.get("description", "").lower()
            tags = tool_info.get("tags", [])
            
            if (query_lower in description or 
                any(query_lower in tag.lower() for tag in tags)):
                matching_tools.append(tool_id)
        
        return matching_tools
    
    def _start_refresh_task(self):
        """Start background task for periodic tool refresh"""
        if self._refresh_task:
            return
            
        async def refresh_loop():
            while True:
                try:
                    await asyncio.sleep(settings.mcp_tool_refresh_interval)
                    await self.refresh_tools()
                    logger.info("Successfully refreshed MCP tools")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Tool refresh failed: {e}")
        
        self._refresh_task = asyncio.create_task(refresh_loop())
        logger.info(f"Started MCP tool refresh task (interval: {settings.mcp_tool_refresh_interval}s)")
    
    async def refresh_tools(self):
        """Refresh tool lists from all servers"""
        if not self._initialized:
            return
        
        old_tools = set(self.tools_registry.keys())
        
        # Clear current registry
        self.tools_registry.clear()
        
        # Rediscover all tools
        await self._discover_all_tools()
        
        new_tools = set(self.tools_registry.keys())
        
        # Log changes
        added = new_tools - old_tools
        removed = old_tools - new_tools
        
        if added:
            logger.info(f"New tools discovered: {sorted(added)}")
        if removed:
            logger.info(f"Tools removed: {sorted(removed)}")
        
        return {
            "added": list(added),
            "removed": list(removed),
            "total": len(new_tools)
        }
    
    async def get_server_health(self) -> Dict[str, str]:
        """Check health status of all MCP servers"""
        health_status = {}
        
        for server_name, client in self.clients.items():
            try:
                # Use async with for each health check operation
                async with client:
                    # Use ping() as the standard health check method
                    await asyncio.wait_for(client.ping(), timeout=5)
                    health_status[server_name] = "healthy"
            except Exception as e:
                health_status[server_name] = f"unhealthy: {str(e)[:100]}"
        
        return health_status
    
    async def reload_configuration(self) -> Dict[str, Any]:
        """Reload MCP configuration without restart"""
        try:
            # Stop refresh task
            if self._refresh_task:
                self._refresh_task.cancel()
                self._refresh_task = None
            
            # Clear existing client references (no need to call __aexit__ on un-entered clients)
            for server_name in list(self.clients.keys()):
                logger.info(f"Clearing reference to MCP server: {server_name}")
                del self.clients[server_name]
            
            # Clear state
            self.clients.clear()
            self.tools_registry.clear()
            self.server_metadata.clear()
            self.server_health.clear()
            self._initialized = False
            
            # Reinitialize
            success = await self.initialize()
            
            return {
                "success": success,
                "servers": list(self.clients.keys()),
                "tools": len(self.tools_registry),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to reload MCP configuration: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get complete MCP manager status"""
        return {
            "initialized": self._initialized,
            "mcp_available": MCP_AVAILABLE,
            "global_enabled": settings.mcp_global_enabled,
            "servers": {
                "connected": len(self.clients),
                "configured": len(self.config.get("mcpServers", {})),
                "health": await self.get_server_health()
            },
            "tools": {
                "total": len(self.tools_registry),
                "by_server": {
                    server: len([t for t in self.tools_registry.keys() if t.startswith(f"{server}.")])
                    for server in self.clients.keys()
                }
            },
            "last_config_load": self._last_config_load,
            "refresh_task_running": self._refresh_task and not self._refresh_task.done()
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            # Stop refresh task
            if self._refresh_task:
                self._refresh_task.cancel()
                try:
                    await self._refresh_task
                except asyncio.CancelledError:
                    pass
            
            # Clear client references (no need to call __aexit__ on un-entered clients)
            for server_name in list(self.clients.keys()):
                logger.info(f"Clearing reference to MCP server: {server_name}")
                del self.clients[server_name]
            
            # Clear state
            self.clients.clear()
            self.tools_registry.clear()
            self.server_metadata.clear()
            self.server_health.clear()
            self._initialized = False
            
            logger.info("MCP Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during MCP Manager cleanup: {e}")