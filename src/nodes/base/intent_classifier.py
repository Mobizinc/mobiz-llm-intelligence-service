"""
Intent Classifier Node Implementation

This module implements the IntentClassifierNode that analyzes user messages
to classify their intent and determine the appropriate processing path.

Part of Epic 2: Agent Nodes Implementation
Story 2.1: Implement IntentClassifierNode
"""

import json
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from .base import BaseNode
from ..models.conversation_state import TechnicalConversationState, IntentType, DomainType


class ClassifiedIntent(BaseModel):
    """Structured output for intent classification"""
    
    intent: IntentType = Field(description="The classified intent type")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Brief explanation of why this intent was chosen")
    secondary_intents: List[IntentType] = Field(
        default=[],
        description="Alternative intents with lower confidence"
    )
    requires_clarification: bool = Field(
        default=False,
        description="Whether the intent is ambiguous and needs clarification"
    )
    disambiguation_questions: List[str] = Field(
        default=[],
        description="Suggested questions to clarify ambiguous intent"
    )


class IntentClassifierNode(BaseNode):
    """
    Node that classifies user intent with confidence scoring and disambiguation.
    
    Supports:
    - Multi-class intent classification
    - Confidence threshold configuration
    - Few-shot learning with domain examples
    - Intent disambiguation for ambiguous requests
    - Fallback handling for low confidence
    """
    
    def __init__(self, confidence_threshold: float = 0.7, **kwargs):
        super().__init__(node_name="IntentClassifier", **kwargs)
        self.confidence_threshold = confidence_threshold
        self.output_parser = PydanticOutputParser(pydantic_object=ClassifiedIntent)
    
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process state through intent classification.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with intent classification results
        """
        # Check cache first
        cache_key = self._get_cache_key(state, [str(self.confidence_threshold)])
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return self._apply_cached_result(state, cached_result)
        
        # Classify intent
        classified_intent = await self._classify_intent(state)
        
        # Update state with classification results
        updated_state = self._update_state_with_intent(state, classified_intent)
        
        # Cache the result
        self._set_cache(cache_key, classified_intent)
        
        return updated_state
    
    async def _classify_intent(self, state: TechnicalConversationState) -> ClassifiedIntent:
        """Classify the intent of the current message"""
        
        system_prompt = self._build_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        classified_intent = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_parser=self.output_parser
        )
        
        return classified_intent
    
    def _build_system_prompt(self, domain: str) -> str:
        """Build system prompt for intent classification"""
        
        domain_context = self.get_domain_context(domain)
        
        base_prompt = f"""You are an expert intent classifier for technical conversations in the {domain} domain.

Your task is to classify the user's intent and provide confidence scoring.

INTENT TYPES:
1. NEW_QUERY - Brand new technical question or request
2. PROVIDING_INFO - User is providing requested information or answering questions  
3. ASKING_FOLLOWUP - Follow-up question to a previous response
4. CLARIFICATION - Asking for clarification of a previous response
5. IMPLEMENTATION - Requesting specific implementation details or code

DOMAIN CONTEXT:
- Technologies: {', '.join(domain_context.get('technologies', []))}
- Focus Areas: {', '.join(domain_context.get('focus_areas', []))}
- Common Patterns: {', '.join(domain_context.get('common_patterns', []))}

CLASSIFICATION GUIDELINES:

NEW_QUERY indicators:
- Fresh technical questions
- "How do I...", "What is...", "Can you help me..."
- No reference to previous conversation
- Specific technical requests

PROVIDING_INFO indicators:  
- Direct answers to questions
- "Yes", "No", followed by details
- Technical specifications or constraints
- "I have...", "My setup is...", "I'm using..."

ASKING_FOLLOWUP indicators:
- "What about...", "How does that work with..."
- Questions building on previous responses
- Requests for more detail on specific points

CLARIFICATION indicators:
- "What did you mean by...", "Can you explain..."
- Confusion about previous responses
- Requests to rephrase or simplify

IMPLEMENTATION indicators:
- "Show me the code", "How do I implement..."
- Requests for step-by-step instructions
- "What's the configuration for..."

CONFIDENCE SCORING:
- 0.9-1.0: Very clear intent with strong indicators
- 0.7-0.8: Clear intent with good indicators
- 0.5-0.6: Somewhat ambiguous, multiple possible intents
- 0.3-0.4: Ambiguous, requires clarification
- 0.0-0.2: Very unclear intent

DISAMBIGUATION:
- If confidence < 0.7, set requires_clarification = true
- Provide specific questions to clarify intent
- Consider context from conversation history

{self.output_parser.get_format_instructions()}"""
        
        return base_prompt
    
    def _build_user_message(self, state: TechnicalConversationState) -> str:
        """Build user message for intent classification"""
        
        message_parts = [
            f"CURRENT MESSAGE: {state['current_input']}",
            f"DOMAIN: {state['domain']}",
        ]
        
        # Add conversation history context
        if state.get('conversation_history'):
            recent_history = state['conversation_history'][-3:]  # Last 3 exchanges
            history_text = []
            for exchange in recent_history:
                if isinstance(exchange, dict):
                    if 'user' in exchange and 'bot' in exchange:
                        history_text.append(f"User: {exchange['user']}")
                        history_text.append(f"Bot: {exchange['bot']}")
            
            if history_text:
                message_parts.append(f"RECENT CONVERSATION:\n" + "\n".join(history_text[-6:]))  # Last 6 lines
        
        # Add current thread context
        if state.get('thread_id') and len(state.get('questions_asked', [])) > 0:
            message_parts.append(f"QUESTIONS PREVIOUSLY ASKED: {', '.join(state['questions_asked'])}")
        
        # Add any extracted info context
        if state.get('extracted_info'):
            extracted_summary = []
            for key, value in state['extracted_info'].items():
                if value:
                    extracted_summary.append(f"{key}: {str(value)[:50]}")
            if extracted_summary:
                message_parts.append(f"PREVIOUSLY EXTRACTED INFO:\n" + "\n".join(extracted_summary[:5]))
        
        return "\n\n".join(message_parts)
    
    def _update_state_with_intent(
        self, 
        state: TechnicalConversationState, 
        classified_intent: ClassifiedIntent
    ) -> TechnicalConversationState:
        """Update state with intent classification results"""
        
        updated_state = state.copy()
        
        # Update intent information
        updated_state['intent'] = classified_intent.intent
        updated_state['confidence_score'] = classified_intent.confidence
        
        # Update intent metadata
        intent_metadata = updated_state.get('intent_metadata', {}).copy()
        intent_metadata.update({
            'reasoning': classified_intent.reasoning,
            'secondary_intents': [intent.value for intent in classified_intent.secondary_intents],
            'requires_clarification': classified_intent.requires_clarification,
            'disambiguation_questions': classified_intent.disambiguation_questions,
            'classification_timestamp': state['updated_at']
        })
        updated_state['intent_metadata'] = intent_metadata
        
        # Handle low confidence scenarios
        if classified_intent.confidence < self.confidence_threshold:
            if classified_intent.requires_clarification and classified_intent.disambiguation_questions:
                # Add disambiguation questions to clarifying questions
                updated_state['questions_asked'] = updated_state.get('questions_asked', [])
                # Don't duplicate questions
                new_questions = [
                    q for q in classified_intent.disambiguation_questions 
                    if q not in updated_state['questions_asked']
                ]
                updated_state['questions_asked'].extend(new_questions[:2])  # Limit to 2 new questions
                
                # Set response to disambiguation questions
                updated_state['response'] = self._format_disambiguation_response(
                    classified_intent.disambiguation_questions[:2], 
                    state['domain']
                )
                updated_state['response_type'] = "clarification"
            else:
                # Generic low confidence handling
                updated_state['response'] = self._format_low_confidence_response(
                    classified_intent.intent,
                    classified_intent.confidence,
                    state['domain']
                )
                updated_state['response_type'] = "clarification"
        
        # Update token usage tracking
        token_usage = updated_state.get('token_usage', {}).copy()
        token_usage['intent_classifier'] = token_usage.get('intent_classifier', 0) + 150  # Estimated tokens
        updated_state['token_usage'] = token_usage
        
        return updated_state
    
    def _apply_cached_result(
        self, 
        state: TechnicalConversationState, 
        cached_intent: ClassifiedIntent
    ) -> TechnicalConversationState:
        """Apply cached classification result to state"""
        
        # Record cache hit
        cache_hits = state.get('cache_hits', []).copy()
        cache_hits.append(f"intent_classifier:{state['updated_at']}")
        
        updated_state = self._update_state_with_intent(state, cached_intent)
        updated_state['cache_hits'] = cache_hits
        
        return updated_state
    
    def _format_disambiguation_response(self, questions: List[str], domain: str) -> str:
        """Format disambiguation questions as a response"""
        
        domain_context = self.get_domain_context(domain)
        
        response_parts = [
            f"I'd like to better understand your {domain} request. Could you help clarify:",
            ""
        ]
        
        for i, question in enumerate(questions, 1):
            response_parts.append(f"{i}. {question}")
        
        response_parts.extend([
            "",
            f"This will help me provide more targeted {domain} guidance for your specific needs."
        ])
        
        return "\n".join(response_parts)
    
    def _format_low_confidence_response(self, intent: IntentType, confidence: float, domain: str) -> str:
        """Format response for low confidence classification"""
        
        if confidence < 0.3:
            return f"""I'm not quite sure I understand your {domain} request. Could you provide a bit more context or rephrase your question? 

For example:
- Are you looking for implementation guidance?
- Do you need help troubleshooting an issue? 
- Are you asking about best practices or recommendations?

The more specific you can be, the better I can assist you."""
        
        else:
            intent_descriptions = {
                IntentType.NEW_QUERY: "asking a new technical question",
                IntentType.PROVIDING_INFO: "providing additional information", 
                IntentType.ASKING_FOLLOWUP: "asking a follow-up question",
                IntentType.CLARIFICATION: "asking for clarification",
                IntentType.IMPLEMENTATION: "requesting implementation details"
            }
            
            description = intent_descriptions.get(intent, "making a technical request")
            
            return f"""I think you're {description}, but I want to make sure I understand correctly. Could you confirm or provide a bit more detail about what you're looking for?

This will help me give you the most relevant {domain} guidance."""
    
    def _validate_output_state(self, state: TechnicalConversationState) -> None:
        """Validate output state for intent classifier"""
        super()._validate_output_state(state)
        
        # Ensure intent classification results are present
        if 'intent' not in state or not state['intent']:
            raise NodeValidationError("Intent classification result missing")
        
        if 'confidence_score' not in state:
            raise NodeValidationError("Confidence score missing")
        
        if not (0.0 <= state['confidence_score'] <= 1.0):
            raise NodeValidationError(f"Invalid confidence score: {state['confidence_score']}")
        
        # Validate intent metadata
        if 'intent_metadata' not in state:
            raise NodeValidationError("Intent metadata missing")
        
        required_metadata = ['reasoning', 'requires_clarification']
        for field in required_metadata:
            if field not in state['intent_metadata']:
                raise NodeValidationError(f"Required metadata field '{field}' missing")