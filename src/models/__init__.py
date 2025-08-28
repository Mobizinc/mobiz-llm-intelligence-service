"""Models package"""

# Bot models
from .bot import (
    BotDomain,
    BotHelpResponse,
    ConversationContext,
    ChannelContext,
    AIProcessingMetrics,
    BackgroundTaskStatus
)

# Slack models
from .slack import *

# Elicitation models
from .elicitation import *

# LangGraph Conversation State Models (Epic 1: Story 1.1)
from .conversation_state import (
    TechnicalConversationState,
    TechnicalConversationStateValidator,
    ConversationStateManager,
    ConversationStage,
    IntentType,
    ResponseType,
    DomainType,
    is_conversation_complete,
    get_conversation_age_minutes,
    should_expire_conversation
)

__all__ = [
    # Bot models
    'BotDomain',
    'BotHelpResponse',
    'ConversationContext',
    'ChannelContext',
    'AIProcessingMetrics',
    'BackgroundTaskStatus',
    
    # LangGraph Conversation State (Epic 1)
    'TechnicalConversationState',
    'TechnicalConversationStateValidator',
    'ConversationStateManager',
    'ConversationStage',
    'IntentType',
    'ResponseType',
    'DomainType',
    'is_conversation_complete',
    'get_conversation_age_minutes',
    'should_expire_conversation'
]