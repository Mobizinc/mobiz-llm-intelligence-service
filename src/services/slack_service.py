import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from ..core.config import settings
from ..core.singletons import get_slack_manager
from ..core.cache import cache_in_request
from ..models.bot import ConversationContext, ChannelContext

logger = logging.getLogger(__name__)


class SlackService:
    """Service for handling Slack API interactions"""
    
    def __init__(self):
        self.slack_manager = get_slack_manager()
    
    @cache_in_request(ttl_seconds=300, key_prefix="slack_channel_")  # 5 minute cache
    async def get_channel_context(self, channel_id: str, channel_name: str = None) -> Dict[str, Any]:
        """Get channel information for context awareness with request-scoped caching"""
        try:
            # The @cache_in_request decorator will handle caching automatically
            # This will only be called once per request for the same channel_id
            # Pass channel_name to avoid unnecessary API call
            context = await self.slack_manager.get_channel_context(channel_id, channel_name)
            
            logger.debug(f"Retrieved channel context for {channel_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get channel context: {e}")
            return {
                "id": channel_id,
                "name": channel_name or "unknown",
                "is_technical": False,
                "is_general": True
            }
    
    async def get_thread_history(self, channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
        """Get thread history for conversation context"""
        try:
            # Direct async call - no thread pool needed
            history = await self.slack_manager.get_thread_history(channel_id, thread_ts)
            
            logger.debug(f"Retrieved thread history: {len(history)} messages")
            return history
            
        except Exception as e:
            logger.error(f"Failed to get thread history: {e}")
            return []
    
    async def should_route_to_domain(self, user_query: str, current_domain: str) -> Optional[str]:
        """Check if query should be routed to a different domain"""
        try:
            # This method doesn't use Slack API - call directly
            suggested_domain = self.slack_manager.should_route_to_domain(user_query, current_domain)
            
            if suggested_domain:
                logger.info(f"Domain routing suggestion: {current_domain} -> {suggested_domain}")
            
            return suggested_domain
            
        except Exception as e:
            logger.error(f"Failed to check domain routing: {e}")
            return None
    
    async def analyze_conversation_context(
        self,
        user_query: str,
        channel_id: str,
        thread_ts: str,
        correlation_id: str
    ) -> ConversationContext:
        """Analyze conversation context for enhanced responses"""
        try:
            # This method doesn't use Slack API - call directly
            analysis = self.slack_manager.analyze_conversation_context(
                user_query,
                channel_id,
                thread_ts,
                correlation_id
            )
            
            # Convert to ConversationContext model
            context = ConversationContext(
                is_follow_up=analysis.get("is_follow_up", False),
                confidence=analysis.get("confidence", 0.0),
                context_messages_count=analysis.get("context_messages_count", 0),
                formatted_context=analysis.get("formatted_context", "No previous conversation context available."),
                conversation_history=analysis.get("conversation_history", [])
            )
            
            logger.debug(f"Analyzed conversation context: follow_up={context.is_follow_up}, confidence={context.confidence}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to analyze conversation context: {e}")
            return ConversationContext()
    
    async def format_technical_response(self, response: str, channel_context: Dict[str, Any]) -> str:
        """Format response based on channel context"""
        try:
            # This method doesn't use Slack API - call directly
            formatted = self.slack_manager.format_technical_response(
                response,
                channel_context
            )
            
            logger.debug(f"Formatted response: {len(formatted)} characters")
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return response  # Return original response if formatting fails
    
    async def format_ai_response_intelligently(
        self, 
        response: str, 
        domain: str, 
        channel_context: Dict[str, Any]
    ) -> str:
        """Optimized AI response formatting with fast path for simple responses"""
        try:
            # Fast path: if response is short and has no formatting markers, skip processing
            if len(response) < 100 and not any(marker in response for marker in ['**', '```', '# ', '## ']):
                domain_emoji_map = {"cloud": "☁️", "devops": "⚙️", "dev": "💻", "network": "🌐"}
                domain_emoji = domain_emoji_map.get(domain, "🤖")
                return f"{domain_emoji} {response}"
            
            # For medium complexity, do basic formatting without thread pool
            if len(response) < 500 and response.count('\n') < 10:
                domain_emoji_map = {"cloud": "☁️", "devops": "⚙️", "dev": "💻", "network": "🌐"}
                domain_emoji = domain_emoji_map.get(domain, "🤖")
                
                # Basic formatting adjustments
                formatted = response
                if channel_context and (channel_context.get("is_general") or not channel_context.get("is_technical")):
                    # For general channels, make it more concise
                    lines = formatted.split('\n')
                    if len(lines) > 8:  # If response is too long for general channel
                        summary_lines = lines[:6]  # Take first 6 lines
                        summary_lines.append("\n_Use this command in a technical channel for detailed guidance._")
                        formatted = '\n'.join(summary_lines)
                
                return f"{domain_emoji} {formatted}"
            
            # Direct call - no thread pool needed for formatting
            formatted = self.slack_manager.format_ai_response_intelligently(
                response,
                domain
            )
            
            # Apply channel-specific adjustments if needed
            if channel_context and (channel_context.get("is_general") or not channel_context.get("is_technical")):
                # For general channels, make it more concise
                lines = formatted.split('\n')
                if len(lines) > 10:  # If response is too long for general channel
                    summary_lines = lines[:8]  # Take first 8 lines
                    summary_lines.append("\n_Use this command in a technical channel for detailed guidance._")
                    formatted = '\n'.join(summary_lines)
            
            logger.debug(f"AI response formatted: {len(formatted)} characters")
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format AI response: {e}")
            return response  # Return original response if formatting fails  # Return original response if formatting fails
    
    async def post_thread_reply(
        self,
        channel_id: str,
        thread_ts: str,
        text: str,
        domain: str,
        is_ephemeral: bool = False,
        correlation_id: str = ""
    ) -> bool:
        """Post a reply to a Slack thread"""
        try:
            # Direct async call - no thread pool needed
            success = await self.slack_manager.post_thread_reply(
                channel_id,
                thread_ts,
                text,
                domain,
                is_ephemeral,
                correlation_id
            )
            
            logger.info(f"Thread reply posted: success={success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to post thread reply: {e}")
            return False
    
    async def post_message(
        self,
        channel_id: str,
        text: str,
        domain: str,
        correlation_id: str = ""
    ) -> Optional[str]:
        """Post a message to a Slack channel"""
        try:
            # Direct async call - no thread pool needed
            message_ts = await self.slack_manager.post_message(
                channel_id,
                text,
                domain,
                correlation_id
            )
            
            logger.info(f"Message posted: ts={message_ts}")
            return message_ts
            
        except Exception as e:
            logger.error(f"Failed to post message: {e}")
            return None
    
    async def send_response(
        self,
        response_url: str = None,
        channel_id: str = None,
        thread_ts: str = None,
        text: str = "",
        is_ephemeral: bool = False,
        correlation_id: str = ""
    ) -> bool:
        """Send response via Slack response URL with threading support"""
        try:
            # Direct async call - no thread pool needed
            success = await self.slack_manager.send_response(
                response_url=response_url,
                channel_id=channel_id,
                thread_ts=thread_ts,
                text=text,
                is_ephemeral=is_ephemeral,
                correlation_id=correlation_id
            )
            
            logger.info(f"Response sent (threading={bool(channel_id and thread_ts)}): success={success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to send response: {e}")
            return False
    
    async def save_conversation_exchange(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        bot_domain: str,
        user_query: str,
        bot_response: str,
        correlation_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Save conversation exchange for context in future interactions"""
        try:
            # This method doesn't use Slack API - call directly
            success = self.slack_manager.save_conversation_exchange(
                channel_id,
                thread_ts,
                user_id,
                bot_domain,
                user_query,
                bot_response,
                correlation_id,
                metadata
            )
            
            logger.debug(f"Conversation exchange saved: success={success}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to save conversation exchange: {e}")
            return False


# Global Slack service instance
slack_service = SlackService()