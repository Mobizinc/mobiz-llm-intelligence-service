import asyncio
import logging
from typing import Dict, Any, List, Optional

from ..core.config import settings
from ..core.cache import cache_in_request
from ..shared.langchain_config import ChainManager
from ..shared.simple_conversation_state import get_simple_conversation_manager
from ..core.singletons import get_mcp_manager

logger = logging.getLogger(__name__)


class AIService:
    """Service for handling AI processing with LangGraph stateful conversations and MCP integration"""
    
    def __init__(self):
        self.chain_manager = ChainManager()  # Keep for backward compatibility
        self.simple_conversation_manager = get_simple_conversation_manager()  # Simple stateful conversations
        self.mcp_manager = get_mcp_manager()
    
    async def process_query(
        self,
        domain: str,
        user_query: str,
        channel_context: Dict[str, Any],
        thread_history: List[Dict[str, Any]],
        conversation_context: str = "No previous conversation context available.",
        thread_id: str = None
    ) -> str:
        """
        Process AI query using LangGraph stateful conversations with MCP context enhancement
        """
        try:
            logger.info(
                f"Processing AI query for {domain}",
                extra={
                    "domain": domain,
                    "query_length": len(user_query),
                    "has_thread_history": len(thread_history) > 0,
                    "has_conversation_context": len(conversation_context) > 50,
                    "mcp_enabled": settings.mcp_global_enabled,
                    "thread_id": thread_id
                }
            )
            
            # Check if this query needs disambiguation (contains ambiguous terms)
            needs_disambiguation = self._needs_disambiguation(user_query)
            
            if needs_disambiguation:
                logger.info(f"Query needs disambiguation, using simple conversation manager for {domain}")
                
                # Use simple conversation manager for stateful conversation handling
                # Convert thread_history to string format
                thread_history_strings = []
                if thread_history:
                    for msg in thread_history:
                        if isinstance(msg, dict) and 'text' in msg:
                            thread_history_strings.append(msg['text'])
                        else:
                            thread_history_strings.append(str(msg))
                
                response = await self.simple_conversation_manager.process_message(
                    user_input=user_query,
                    domain=domain,
                    channel_context=channel_context,
                    thread_id=thread_id,
                    thread_history=thread_history_strings,
                    channel_id=channel_context.get('id')  # Pass actual channel ID
                )
                
                logger.info(f"Simple conversation manager processed query successfully for {domain}")
                return response
            
            else:
                logger.info(f"Query doesn't need disambiguation, using simple chain for {domain}")
                
                # For non-ambiguous queries, use the original LangChain approach
                return await self._process_with_langchain(
                    domain, user_query, channel_context, thread_history, conversation_context
                )
            
        except Exception as e:
            logger.error(
                f"AI processing failed for {domain}: {e}",
                extra={
                    "domain": domain,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            # Return a fallback response
            return f"I'm sorry, I encountered an error processing your {domain} question. Please try rephrasing your request or contact support if this issue persists."
    
    def _needs_disambiguation(self, user_query: str) -> bool:
        """Determine if a query needs disambiguation via LangGraph."""
        
        query_lower = user_query.lower()
        
        # Ambiguous terms that typically need clarification
        ambiguous_terms = [
            # VPN/Connectivity
            'vpn', 'tunnel', 'site to site', 'connectivity', 'connect',
            
            # Cloud deployment (when not specific)
            'setup', 'configure', 'deploy', 'create', 'build',
            
            # Network (when not specific)
            'network', 'firewall', 'router', 'switch',
            
            # Development (when not specific)
            'code', 'app', 'application', 'system'
        ]
        
        # Specific terms that usually don't need disambiguation
        specific_terms = [
            # Specific technologies
            'azure virtual network gateway', 'palo alto panorama', 
            'cisco asa', 'fortinet fortigate', 'docker container',
            'kubernetes pod', 'react component', 'sql query',
            
            # Specific error messages
            'error', 'exception', 'failed', 'not working',
            
            # Specific commands/configurations
            'command', 'config', 'setting', 'parameter'
        ]
        
        # Check for specific terms first (don't disambiguate)
        if any(term in query_lower for term in specific_terms):
            return False
        
        # Check for ambiguous terms (do disambiguate)
        if any(term in query_lower for term in ambiguous_terms):
            return True
        
        # Short queries are often ambiguous
        if len(user_query.split()) <= 6:
            return True
        
        return False
    
    async def _process_with_langchain(
        self,
        domain: str,
        user_query: str,
        channel_context: Dict[str, Any],
        thread_history: List[Dict[str, Any]],
        conversation_context: str
    ) -> str:
        """Process query with original LangChain approach (for non-ambiguous queries)."""
        
        # Gather MCP documentation context if available
        mcp_context = ""
        if settings.mcp_global_enabled and self.mcp_manager._initialized:
            try:
                mcp_context = await self._gather_mcp_context(user_query, domain)
                logger.info(
                    f"Gathered MCP context for {domain}",
                    extra={
                        "domain": domain,
                        "mcp_context_length": len(mcp_context),
                        "has_mcp_context": bool(mcp_context)
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to gather MCP context for {domain}: {e}")
                mcp_context = ""
        
        # Get the appropriate chain for the domain
        chain = self.chain_manager.get_chain(domain)
        
        # Check if we can use lightweight processing for simple queries
        if len(user_query) < 50 and not any(keyword in user_query.lower() for keyword in ['error', 'debug', 'troubleshoot', 'configure', 'install']):
            # For simple queries, try direct processing without thread pool
            try:
                response = await self._process_simple_query(chain, user_query, channel_context, thread_history, conversation_context, mcp_context)
                if response:
                    logger.info(f"Simple query processed directly for {domain}")
                    return response
            except Exception:
                # Fall back to thread pool if direct processing fails
                pass
        
        # For complex queries or fallback, use thread pool to avoid blocking
        # Note: This is necessary because LangChain chains are synchronous
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            chain.process_query,
            user_query,
            channel_context,
            thread_history,
            conversation_context,
            mcp_context  # Pass MCP context to the chain
        )
        
        logger.info(
            f"AI query processed successfully for {domain}",
            extra={
                "domain": domain,
                "response_length": len(response) if response else 0,
                "used_mcp_context": bool(mcp_context)
            }
        )
        
        return response
    
    async def _process_simple_query(self, chain, user_query: str, channel_context: Dict[str, Any], 
                                   thread_history: List[Dict[str, Any]], conversation_context: str, mcp_context: str) -> Optional[str]:
        """Process simple queries without thread pool for better performance"""
        try:
            # For simple queries like "help", "status", "what is X", provide direct responses
            query_lower = user_query.lower().strip()
            
            if query_lower in ['help', '?']:
                return "I'm here to help! Ask me technical questions and I'll provide expert guidance based on my specialized knowledge."
            
            if query_lower in ['status', 'health']:
                return "I'm running and ready to assist with your technical questions!"
            
            # For other simple patterns, still use the full chain but indicate it's a simple query
            return None  # Signal to use the full processing
            
        except Exception:
            return None
    
    @cache_in_request(ttl_seconds=180, key_prefix="mcp_context_")  # 3 minute cache
    async def _gather_mcp_context(self, user_query: str, domain: str) -> str:
        """Gather relevant documentation context from MCP servers with request-scoped caching"""
        try:
            # Get tools available for this domain
            domain_tools = await self.mcp_manager.search_tools_by_domain(domain)
            
            if not domain_tools:
                logger.debug(f"No MCP tools available for domain: {domain}")
                return ""
            
            # Search for tools that might help with this query
            search_tools = await self.mcp_manager.search_tools_by_query(user_query, domain)
            
            # Combine domain tools and search results
            relevant_tools = list(set(domain_tools + search_tools))
            
            if not relevant_tools:
                logger.debug(f"No relevant MCP tools found for query: {user_query[:50]}...")
                return ""
            
            # Gather context from relevant tools
            contexts = []
            for tool_id in relevant_tools[:3]:  # Limit to top 3 tools to avoid token overuse
                try:
                    tool_info = self.mcp_manager.tools_registry.get(tool_id)
                    if not tool_info:
                        continue
                    
                    # Apply token limits from tool restrictions
                    token_limit = tool_info.get("restrictions", {}).get("token_limit", 1000)
                    
                    # Prepare search arguments based on tool type
                    search_args = self._prepare_search_arguments(user_query, domain, token_limit)
                    
                    # Call the MCP tool
                    result = await self.mcp_manager.call_tool(tool_id, search_args)
                    
                    if result:
                        contexts.append({
                            "source": tool_id,
                            "content": str(result)[:token_limit],
                            "priority": tool_info.get("restrictions", {}).get("priority", 0)
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to get context from tool {tool_id}: {e}")
                    continue
            
            if not contexts:
                return ""
            
            # Sort by priority and combine contexts
            contexts.sort(key=lambda x: x["priority"], reverse=True)
            
            # Format the combined context
            formatted_context = self._format_mcp_context(contexts, user_query)
            
            logger.debug(f"Gathered MCP context from {len(contexts)} tools, total length: {len(formatted_context)}")
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error gathering MCP context: {e}")
            return ""
    
    def _prepare_search_arguments(self, user_query: str, domain: str, token_limit: int) -> Dict[str, Any]:
        """Prepare search arguments for MCP tools based on query and domain"""
        # Extract key terms from the query for more targeted search
        key_terms = self._extract_key_terms(user_query, domain)
        
        # Common argument patterns for documentation search tools
        return {
            "query": key_terms,
            "limit": min(token_limit // 100, 20),  # Reasonable limit based on token budget
            "include_code": True,
            "include_examples": True,
            "domain": domain
        }
    
    def _extract_key_terms(self, user_query: str, domain: str) -> str:
        """Extract key terms from user query for more targeted MCP searches"""
        # Simple keyword extraction - could be enhanced with NLP
        # Remove common stop words and focus on technical terms
        stop_words = {"how", "what", "where", "when", "why", "can", "could", "should", "would", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        words = user_query.lower().split()
        key_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add domain-specific keywords
        domain_keywords = {
            "cloud": ["azure", "aws", "cloud", "virtual", "storage", "compute"],
            "devops": ["docker", "kubernetes", "ci/cd", "pipeline", "deployment"],
            "dev": ["python", "javascript", "api", "database", "framework"],
            "network": ["firewall", "router", "switch", "vlan", "vpn"]
        }
        
        if domain in domain_keywords:
            key_words.extend([kw for kw in domain_keywords[domain] if kw in user_query.lower()])
        
        return " ".join(key_words[:10])  # Limit to top 10 key terms
    
    def _format_mcp_context(self, contexts: List[Dict[str, Any]], user_query: str) -> str:
        """Format MCP contexts into a coherent documentation section"""
        if not contexts:
            return ""
        
        formatted_parts = [
            "## Latest Documentation Context",
            f"*Based on your query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}*",
            ""
        ]
        
        for i, context in enumerate(contexts):
            source_parts = context["source"].split(".")
            server_name = source_parts[0] if len(source_parts) > 1 else "Documentation"
            
            formatted_parts.extend([
                f"### {server_name.replace('_', ' ').title()} Documentation",
                context["content"],
                ""
            ])
        
        formatted_parts.extend([
            "*Note: This information comes from up-to-date documentation sources.*",
            ""
        ])
        
        return "\n".join(formatted_parts)


# Global AI service instance
ai_service = AIService()