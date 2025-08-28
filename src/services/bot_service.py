import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal

from ..core.config import settings
from ..core.logging_config import get_logger
from ..models.bot import BotDomain, ConversationContext, ChannelContext, AIProcessingMetrics
from ..models.slack import BotProcessingRequest
from ..models.conversation import ConversationRequest, ConversationMessage
from ..orchestration.streaming_executor import StreamingGraphExecutor
from ..security.input_validator import input_validator

logger = get_logger(__name__)


class BotService:
    """Service class for handling bot processing logic"""
    
    def __init__(self):
        self.processing_tasks: Dict[str, AIProcessingMetrics] = {}
        self.streaming_executor = StreamingGraphExecutor()
    
    async def process_event(
        self,
        user_query: str,
        bot_domain: str,
        user_id: str,
        correlation_id: str,
        channel_context: Dict[str, Any],
        thread_history: Optional[List[str]] = None,
        conversation_context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        High-level interface for processing events from various sources (Slack, etc.).
        
        This method encapsulates the entire lifecycle of creating a ConversationRequest,
        streaming the response, and buffering it for non-streaming clients.
        
        Args:
            user_query: The user's input text
            bot_domain: Domain for the bot (cloud, devops, network, dev)
            user_id: Identifier for the user
            correlation_id: Unique ID for tracking this request
            channel_context: Context about the channel/environment
            thread_history: Optional list of previous messages in thread
            conversation_context: Optional conversation context string
            metadata: Optional additional metadata for the request
            
        Returns:
            Complete response text suitable for the calling interface
        """
        # Build ConversationRequest for unified processing
        conversation_request = ConversationRequest(
            messages=[
                ConversationMessage(
                    role="user",
                    content=user_query,
                    metadata={}
                )
            ],
            user_id=user_id,
            conversation_id=correlation_id,
            client_id=channel_context.get("client_name"),
            bot_type=bot_domain,
            metadata={
                "channel_id": channel_context.get("channel_id"),
                "channel_name": channel_context.get("name"),
                "is_technical": channel_context.get("is_technical", False),
                "conversation_context": conversation_context,
                **(metadata or {})
            }
        )
        
        # Security validation to prevent prompt injection attacks
        validation_result = input_validator.validate_conversation_request(conversation_request)
        
        if not validation_result.is_valid:
            logger.error(
                f"SECURITY: Input validation failed for Slack user {user_id}",
                extra={
                    "user_id": user_id,
                    "correlation_id": correlation_id,
                    "channel_name": channel_context.get("name"),
                    "risk_level": validation_result.risk_level,
                    "violations": validation_result.violations,
                    "requires_review": validation_result.requires_human_review
                }
            )
            
            # Return an appropriate security message for Slack
            if validation_result.risk_level == "critical":
                return "🚨 **Security Alert**: Your message has been blocked for security reasons. Please contact an administrator if you believe this is an error."
            else:
                return "⚠️ **Input Error**: Your message contains content that couldn't be processed. Please rephrase your question and try again."
        
        # Add thread history to messages if available
        if thread_history:
            for msg_text in thread_history[-5:]:  # Limit to last 5 messages
                if msg_text:
                    # Use a simple heuristic to determine message role
                    role = self._determine_message_role(msg_text)
                    conversation_request.messages.insert(-1, ConversationMessage(
                        role=role,
                        content=msg_text,
                        metadata={"from_thread": True}
                    ))
        
        # Use the buffering adapter for consistent response
        return await self.__buffer_stream_response(conversation_request)
    
    def _determine_message_role(self, message_text: str) -> str:
        """
        Determine the role of a message based on its content.
        
        This is a simple heuristic that should be enhanced over time.
        
        Args:
            message_text: The text content of the message
            
        Returns:
            Either "user" or "assistant"
        """
        # Simple heuristic: if the message contains bot-like language, consider it assistant
        bot_indicators = ["bot", "assistant", "I'll help", "I can", "Here's", "To answer"]
        message_lower = message_text.lower()
        
        if any(indicator in message_lower for indicator in bot_indicators):
            return "assistant"
        return "user"

    async def __buffer_stream_response(self, request: ConversationRequest) -> str:
        """
        Buffer streaming response into complete text for Slack.
        
        This method acts as a "buffering adapter" that collects all streaming chunks
        from the StreamingGraphExecutor into a single complete response suitable for Slack.
        
        Args:
            request: ConversationRequest formatted for the StreamingGraphExecutor
            
        Returns:
            Complete buffered response text
        """
        response_chunks = []
        
        try:
            # Stream from the unified StreamingGraphExecutor
            async for stream_message in self.streaming_executor.stream_execution(request):
                # Only collect text chunks for the final response - ignore status updates for Slack
                if hasattr(stream_message, 'type') and stream_message.type == "text":
                    if hasattr(stream_message, 'content') and stream_message.content:
                        response_chunks.append(stream_message.content)
                
                # Handle error messages
                elif hasattr(stream_message, 'type') and stream_message.type == "error":
                    error_msg = f"Error: {stream_message.message}" if hasattr(stream_message, 'message') else "An error occurred during processing."
                    response_chunks.append(error_msg)
                    break
        
        except Exception as e:
            logger.error(f"Error during stream buffering: {e}")
            response_chunks.append(f"I encountered an error while processing your request: {str(e)}")
        
        # Join all chunks into final response
        final_response = "".join(response_chunks).strip()
        
        # Fallback if no content was generated
        if not final_response:
            final_response = "I apologize, but I wasn't able to generate a response for your request. Please try again or rephrase your question."
        
        return final_response
    
    async def process_bot_request_background(
        self,
        domain: str,
        emoji: str,
        user_query: str,
        channel_context: Dict[str, Any],
        thread_history: List[Dict[str, Any]],
        response_url: str,
        correlation_id: str,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        initial_message_ts: Optional[str] = None
    ) -> bool:
        """
        Background processing function that handles AI processing and sends response via response_url
        This replaces the Azure Functions threading approach with FastAPI background tasks
        """
        start_time = time.time()
        success = False
        
        try:
            logger.info(
                f"Starting background processing for {domain} bot",
                extra={
                    "correlation_id": correlation_id,
                    "domain": domain,
                    "query_length": len(user_query),
                    "channel_id": channel_id
                }
            )
            
            # Track processing start
            metrics = AIProcessingMetrics(
                domain=domain,
                correlation_id=correlation_id,
                start_time=start_time
            )
            self.processing_tasks[correlation_id] = metrics
            
            # Import services here to avoid circular imports
            from ..services.slack_service import SlackService
            
            slack_service = SlackService()
            
            # Check for domain routing
            suggested_domain = await slack_service.should_route_to_domain(user_query, domain)
            if suggested_domain:
                route_message = f"Your question seems more related to {suggested_domain}. Try `/{suggested_domain}-bot {user_query}` for specialized help.\\n\\nI'll still try to help from a {domain} perspective:"
                user_query = f"{route_message}\\n\\nOriginal question: {user_query}"
                
                logger.info(
                    f"Routing suggestion provided: {suggested_domain}",
                    extra={
                        "correlation_id": correlation_id,
                        "current_domain": domain,
                        "suggested_domain": suggested_domain
                    }
                )
            
            # Get conversation context for enhanced responses
            conversation_analysis = ConversationContext()
            conversation_context_str = "No previous conversation context available."
            
            if channel_id and initial_message_ts:
                try:
                    conversation_analysis = await slack_service.analyze_conversation_context(
                        user_query, channel_id, initial_message_ts, correlation_id
                    )
                    conversation_context_str = conversation_analysis.formatted_context
                    
                    logger.info(
                        f"Conversation context analyzed for {domain}",
                        extra={
                            "correlation_id": correlation_id,
                            "domain": domain,
                            "is_follow_up": conversation_analysis.is_follow_up,
                            "confidence": conversation_analysis.confidence,
                            "context_messages_count": conversation_analysis.context_messages_count
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to analyze conversation context: {e}")
            
            # Process the AI request
            ai_start_time = time.time()
            logger.info(
                f"Starting AI processing for {domain} query",
                extra={
                    "correlation_id": correlation_id,
                    "domain": domain,
                    "ai_processing_start": ai_start_time,
                    "has_conversation_context": bool(conversation_analysis.conversation_history)
                }
            )
            
            # Build ConversationRequest for the unified StreamingGraphExecutor
            conversation_request = ConversationRequest(
                messages=[
                    ConversationMessage(
                        role="user",
                        content=user_query,
                        timestamp=datetime.now().isoformat(),
                        metadata={}
                    )
                ],
                user_id=user_id or f"slack_user_{correlation_id}",
                conversation_id=correlation_id,
                client_id=channel_context.get("client_id"),
                bot_type=domain,  # cloud, devops, network, dev
                metadata={
                    "channel_id": channel_id,
                    "channel_name": channel_context.get("name"),
                    "is_technical": channel_context.get("is_technical", False),
                    "thread_ts": initial_message_ts,
                    "conversation_context": conversation_context_str,
                    "is_follow_up": conversation_analysis.is_follow_up,
                    "follow_up_confidence": conversation_analysis.confidence
                }
            )

            # Add thread history to messages if available (insert before user query)
            if thread_history:
                for i, msg in enumerate(thread_history[-5:]):  # Limit to last 5 messages
                    if isinstance(msg, dict) and 'text' in msg and msg['text']:
                        # Determine role based on message content or bot indicator
                        role = "assistant" if msg.get("bot") or "bot" in msg.get("text", "").lower() else "user"
                        conversation_request.messages.insert(-1, ConversationMessage(  # Insert before the latest user message
                            role=role,
                            content=msg['text'],
                            timestamp=datetime.now().isoformat(),
                            metadata={"from_thread": True}
                        ))

            # Use the unified StreamingGraphExecutor with buffering
            response = await self._buffer_stream_response(conversation_request)
            
            ai_duration = time.time() - ai_start_time
            logger.info(
                f"AI processing completed for {domain}",
                extra={
                    "correlation_id": correlation_id,
                    "domain": domain,
                    "ai_duration": ai_duration,
                    "response_length": len(response) if response else 0
                }
            )
            
            # Format the response intelligently - handles any OpenAI response structure
            formatted_response = await slack_service.format_ai_response_intelligently(
                response, domain, channel_context
            )
            
            logger.info(
                f"Response formatted for {domain}",
                extra={
                    "correlation_id": correlation_id,
                    "domain": domain,
                    "formatted_length": len(formatted_response),
                    "is_ephemeral": channel_context.get('is_general', False)
                }
            )
            
            # Send response via thread reply if initial_message_ts available, otherwise use response_url
            logger.info(
                f"Threading decision for {domain} bot",
                extra={
                    "correlation_id": correlation_id,
                    "has_initial_message_ts": bool(initial_message_ts),
                    "has_channel_id": bool(channel_id),
                    "initial_message_ts": initial_message_ts,
                    "channel_id": channel_id,
                    "will_use_threading": bool(initial_message_ts and channel_id)
                }
            )
            
            if initial_message_ts and channel_id:
                logger.info(f"Using threading path for {domain} bot response", extra={"correlation_id": correlation_id})
                success = await slack_service.post_thread_reply(
                    channel_id=channel_id,
                    thread_ts=initial_message_ts,
                    text=formatted_response,
                    domain=domain,
                    is_ephemeral=channel_context.get('is_general', False),
                    correlation_id=correlation_id
                )
            else:
                logger.info(f"Using fallback response_url path for {domain} bot response", extra={"correlation_id": correlation_id})
                # Fallback to response_url method (but still try threading if we have the info)
                success = await slack_service.send_response(
                    response_url=response_url,
                    channel_id=channel_id,
                    thread_ts=initial_message_ts,
                    text=formatted_response,
                    is_ephemeral=channel_context.get('is_general', False),
                    correlation_id=correlation_id
                )
            
            # Save conversation exchange if response was sent successfully
            if success and channel_id and initial_message_ts and user_id:
                try:
                    conversation_saved = await slack_service.save_conversation_exchange(
                        channel_id=channel_id,
                        thread_ts=initial_message_ts,
                        user_id=user_id,
                        bot_domain=domain,
                        user_query=user_query,
                        bot_response=response,
                        correlation_id=correlation_id,
                        metadata={
                            "channel_name": channel_context.get("name", "unknown"),
                            "is_technical": channel_context.get("is_technical", False),
                            "response_time": ai_duration,
                            "was_follow_up": conversation_analysis.is_follow_up,
                            "follow_up_confidence": conversation_analysis.confidence
                        }
                    )
                    
                    if conversation_saved:
                        logger.debug(f"Conversation exchange saved for {correlation_id}")
                    else:
                        logger.warning(f"Failed to save conversation exchange for {correlation_id}")
                        
                except Exception as e:
                    logger.error(f"Error saving conversation exchange: {e}")
            
            # Update metrics
            total_response_time = time.time() - start_time
            metrics.end_time = time.time()
            metrics.duration = total_response_time
            metrics.success = success
            
            if success:
                logger.info(
                    f"Successfully completed background processing for {domain} bot",
                    extra={
                        "correlation_id": correlation_id,
                        "domain": domain,
                        "total_duration": total_response_time,
                        "ai_duration": ai_duration
                    }
                )
            else:
                logger.error(
                    f"Failed to send {domain} bot response",
                    extra={
                        "correlation_id": correlation_id,
                        "domain": domain,
                        "total_duration": total_response_time
                    }
                )
                
        except Exception as e:
            error_msg = f"{domain.title()} Bot streaming background processing error: {str(e)}"
            logger.error(
                error_msg,
                extra={
                    "correlation_id": correlation_id,
                    "domain": domain,
                    "exception_type": type(e).__name__,
                    "using_streaming_executor": True
                },
                exc_info=True
            )
            
            # Update metrics with error
            if correlation_id in self.processing_tasks:
                self.processing_tasks[correlation_id].success = False
                self.processing_tasks[correlation_id].error_message = str(e)
            
            # Send error response via response_url
            try:
                from ..services.slack_service import SlackService
                slack_service = SlackService()
                
                error_response = f"⚠️ I encountered an error processing your {domain} question. Please try rephrasing or contact support if this persists."
                await slack_service.send_response(
                    response_url=response_url,
                    text=error_response,
                    correlation_id=correlation_id
                )
            except Exception as send_error:
                logger.error(
                    f"Failed to send error response via response_url",
                    extra={
                        "correlation_id": correlation_id,
                        "domain": domain,
                        "send_error": str(send_error)
                    },
                    exc_info=True
                )
        
        return success
    
    async def handle_command(
        self,
        user_query: str,
        user_id: str,
        channel_id: Optional[str] = None,
        bot_type: str = "general",
        client_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Handle command using streaming adapter pattern.
        
        This method acts as a "Slack Adapter" for the new streaming core.
        It buffers streaming output into a complete response for Slack delivery.
        
        Args:
            user_query: The user's query/command
            user_id: ID of the user making the request
            channel_id: Optional channel ID for context
            bot_type: Type of bot (cloud, devops, network, dev)
            client_id: Optional client ID for context loading
            conversation_id: Optional conversation ID for threading
            
        Returns:
            str: Complete buffered response from streaming execution
        """
        try:
            # Create conversation request for streaming executor
            conversation_request = ConversationRequest(
                messages=[
                    ConversationMessage(
                        role="user",
                        content=user_query,
                        metadata={"source": "slack"}
                    )
                ],
                user_id=user_id,
                channel_id=channel_id,
                client_id=client_id,
                bot_type=bot_type,
                conversation_id=conversation_id,
                metadata={
                    "adapter": "slack",
                    "timestamp": time.time()
                }
            )
            
            # Stream execution and buffer all chunks
            response_buffer = []
            
            logger.info(
                f"Starting streaming execution for {bot_type} bot",
                extra={
                    "user_id": user_id,
                    "bot_type": bot_type,
                    "client_id": client_id,
                    "query_length": len(user_query)
                }
            )
            
            async for stream_message in self.streaming_executor.stream_execution(conversation_request):
                # Only collect text chunks for the final response
                if hasattr(stream_message, 'type') and stream_message.type == "text":
                    if hasattr(stream_message, 'content') and stream_message.content:
                        response_buffer.append(stream_message.content)
                
                # Log status updates for debugging
                elif hasattr(stream_message, 'type') and stream_message.type == "status":
                    logger.debug(
                        f"Streaming status: {stream_message.operation}",
                        extra={
                            "user_id": user_id,
                            "progress": getattr(stream_message, 'progress', None),
                            "status": getattr(stream_message, 'status', 'unknown')
                        }
                    )
                
                # Handle errors in stream
                elif hasattr(stream_message, 'type') and stream_message.type == "error":
                    logger.error(
                        f"Streaming error: {stream_message.message}",
                        extra={
                            "user_id": user_id,
                            "error_code": getattr(stream_message, 'error_code', 'unknown'),
                            "details": getattr(stream_message, 'details', {})
                        }
                    )
                    response_buffer.append(f"⚠️ {stream_message.message}")
            
            # Combine all buffered chunks into final response
            final_response = "".join(response_buffer).strip()
            
            if not final_response:
                final_response = f"I apologize, but I wasn't able to generate a response to your {bot_type} question. Please try rephrasing or provide more details."
            
            logger.info(
                f"Streaming execution completed for {bot_type} bot",
                extra={
                    "user_id": user_id,
                    "bot_type": bot_type,
                    "response_length": len(final_response)
                }
            )
            
            return final_response
            
        except Exception as e:
            logger.error(
                f"Error in handle_command: {e}",
                extra={
                    "user_id": user_id,
                    "bot_type": bot_type,
                    "exception_type": type(e).__name__
                },
                exc_info=True
            )
            
            return f"⚠️ I encountered an error processing your {bot_type} question. Please try again or contact support if this persists."

    async def process_bot_request_langgraph(
        self,
        user_query: str,
        user_id: str,
        channel_id: str,
        channel_name: str,
        correlation_id: str,
        interaction_type: Literal["slash_command", "app_mention", "thread_reply"],
        response_url: Optional[str] = None,
        thread_history: Optional[List[Dict[str, Any]]] = None,
        channel_context: Optional[Dict[str, Any]] = None,
        initial_message_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process bot request using LangGraph workflow for intelligent routing and client context.
        
        This is the new Phase 1 processing method that replaces direct chain processing
        with intelligent graph-based routing and client awareness.
        """
        start_time = time.time()
        
        try:
            logger.info(
                f"Starting LangGraph processing",
                extra={
                    "correlation_id": correlation_id,
                    "interaction_type": interaction_type,
                    "channel_name": channel_name,
                    "query_length": len(user_query)
                }
            )
            
            # Import LangGraph workflow
            from ..agents.workflow import process_msp_request
            
            # Process through LangGraph workflow
            final_state = await process_msp_request(
                user_query=user_query,
                user_id=user_id,
                channel_id=channel_id,
                channel_name=channel_name,
                correlation_id=correlation_id,
                interaction_type=interaction_type,
                thread_history=thread_history or [],
                channel_context=channel_context or {}
            )
            
            # Extract results from final state
            ai_response = final_state.get("ai_response", "I apologize, but I couldn't process your request.")
            selected_domain = final_state.get("selected_domain", "general")
            ai_processing_time = final_state.get("ai_processing_time", 0.0)
            routing_confidence = final_state.get("routing_confidence", 0.0)
            errors = final_state.get("errors", [])
            
            logger.info(
                f"LangGraph processing completed",
                extra={
                    "correlation_id": correlation_id,
                    "selected_domain": selected_domain,
                    "routing_confidence": routing_confidence,
                    "ai_processing_time": ai_processing_time,
                    "errors_count": len(errors),
                    "total_time": time.time() - start_time
                }
            )
            
            # If response_url is provided, send response back to Slack
            if response_url:
                await self._send_langgraph_response(
                    final_state=final_state,
                    response_url=response_url,
                    channel_id=channel_id,
                    initial_message_ts=initial_message_ts,
                    correlation_id=correlation_id
                )
            
            return {
                "success": True,
                "ai_response": ai_response,
                "selected_domain": selected_domain,
                "routing_confidence": routing_confidence,
                "ai_processing_time": ai_processing_time,
                "client_name": final_state.get("client_name"),
                "urgency_level": final_state.get("urgency_level"),
                "errors": errors,
                "fallback_used": final_state.get("fallback_used", False),
                "requires_follow_up": final_state.get("requires_follow_up", False),
                "should_escalate": final_state.get("should_escalate", False)
            }
            
        except Exception as e:
            logger.error(
                f"LangGraph processing failed: {e}",
                extra={"correlation_id": correlation_id},
                exc_info=True
            )
            
            # Fallback error response
            if response_url:
                try:
                    from ..services.slack_service import SlackService
                    slack_service = SlackService()
                    
                    error_response = "⚠️ I encountered an error processing your request. Please try again or contact support."
                    await slack_service.send_response(
                        response_url=response_url,
                        text=error_response,
                        correlation_id=correlation_id
                    )
                except Exception as send_error:
                    logger.error(f"Failed to send error response: {send_error}")
            
            return {
                "success": False,
                "ai_response": "I encountered an error processing your request.",
                "selected_domain": "general",
                "errors": [f"langgraph_processing_error: {str(e)}"],
                "fallback_used": True
            }
    
    async def _send_langgraph_response(
        self,
        final_state: Dict[str, Any],
        response_url: str,
        channel_id: Optional[str] = None,
        initial_message_ts: Optional[str] = None,
        correlation_id: str = ""
    ) -> bool:
        """
        Send LangGraph response back to Slack using the appropriate method.
        """
        try:
            from ..services.slack_service import SlackService
            
            slack_service = SlackService()
            ai_response = final_state.get("ai_response", "No response generated.")
            selected_domain = final_state.get("selected_domain", "general")
            channel_context = final_state.get("channel_context", {})
            
            # Format response for Slack
            formatted_response = await slack_service.format_ai_response_intelligently(
                ai_response, selected_domain, channel_context
            )
            
            # Prefer threading if we have the necessary information
            if initial_message_ts and channel_id:
                logger.info(
                    f"Sending LangGraph response via thread",
                    extra={
                        "correlation_id": correlation_id,
                        "domain": selected_domain,
                        "thread_ts": initial_message_ts
                    }
                )
                
                success = await slack_service.post_thread_reply(
                    channel_id=channel_id,
                    thread_ts=initial_message_ts,
                    text=formatted_response,
                    domain=selected_domain,
                    is_ephemeral=channel_context.get('is_general', False),
                    correlation_id=correlation_id
                )
            else:
                logger.info(
                    f"Sending LangGraph response via response_url",
                    extra={
                        "correlation_id": correlation_id,
                        "domain": selected_domain
                    }
                )
                
                success = await slack_service.send_response(
                    response_url=response_url,
                    channel_id=channel_id,
                    thread_ts=initial_message_ts,
                    text=formatted_response,
                    is_ephemeral=channel_context.get('is_general', False),
                    correlation_id=correlation_id
                )
            
            # Save conversation if successful and we have the necessary info
            if success and channel_id and initial_message_ts and final_state.get("engineer_id"):
                try:
                    conversation_saved = await slack_service.save_conversation_exchange(
                        channel_id=channel_id,
                        thread_ts=initial_message_ts,
                        user_id=final_state["engineer_id"],
                        bot_domain=selected_domain,
                        user_query=final_state.get("user_query", ""),
                        bot_response=ai_response,
                        correlation_id=correlation_id,
                        metadata={
                            "channel_name": channel_context.get("name", "unknown"),
                            "is_technical": channel_context.get("is_technical", False),
                            "response_time": final_state.get("ai_processing_time", 0.0),
                            "client_name": final_state.get("client_name"),
                            "routing_confidence": final_state.get("routing_confidence", 0.0),
                            "urgency_level": final_state.get("urgency_level"),
                            "via_langgraph": True
                        }
                    )
                    
                    if conversation_saved:
                        logger.debug(f"LangGraph conversation saved for {correlation_id}")
                    else:
                        logger.warning(f"Failed to save LangGraph conversation for {correlation_id}")
                        
                except Exception as e:
                    logger.error(f"Error saving LangGraph conversation: {e}")
            
            return success
            
        except Exception as e:
            logger.error(
                f"Error sending LangGraph response: {e}",
                extra={"correlation_id": correlation_id},
                exc_info=True
            )
            return False
    
    def get_task_status(self, correlation_id: str) -> Optional[AIProcessingMetrics]:
        """Get the status of a background processing task"""
        return self.processing_tasks.get(correlation_id)
    
    def cleanup_completed_tasks(self, max_age_seconds: int = 3600):
        """Clean up completed tasks older than max_age_seconds"""
        current_time = time.time()
        to_remove = []
        
        for correlation_id, metrics in self.processing_tasks.items():
            if metrics.end_time and (current_time - metrics.end_time) > max_age_seconds:
                to_remove.append(correlation_id)
        
        for correlation_id in to_remove:
            del self.processing_tasks[correlation_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed tasks")


# Global bot service instance
bot_service = BotService()