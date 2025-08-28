"""
Direct Answer Node Implementation

This module implements the DirectAnswerNode that provides immediate answers
to simple technical queries without requiring the full implementation planning process.

Part of Epic 2: Agent Nodes Implementation
Story 2.5: Build ImplementationGenerator and DirectAnswerNode
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from .base import BaseNode, NodeValidationError
from ...models.conversation_state import TechnicalConversationState, ResponseType, IntentType, ConversationStage
from ...models.streaming import StreamTextChunk


class AnswerType(str, Enum):
    """Types of direct answers"""
    EXPLANATION = "explanation"       # Explain concepts or technologies
    HOW_TO = "how_to"                # Step-by-step instructions
    TROUBLESHOOTING = "troubleshooting"  # Problem solving guidance
    BEST_PRACTICE = "best_practice"  # Recommendations and best practices
    COMPARISON = "comparison"        # Compare options or approaches
    REFERENCE = "reference"          # Reference information or specs


class AnswerComplexity(str, Enum):
    """Complexity levels for direct answers"""
    SIMPLE = "simple"           # Basic, straightforward answer
    MODERATE = "moderate"       # Some detail, multiple aspects
    DETAILED = "detailed"       # Comprehensive with examples
    EXPERT = "expert"          # In-depth technical analysis


class CodeExample(BaseModel):
    """Represents a code example"""
    
    language: str = Field(description="Programming language or type")
    code: str = Field(description="The code content")
    description: str = Field(description="What this code does")
    filename: Optional[str] = Field(None, description="Suggested filename")


class DirectAnswer(BaseModel):
    """Structured output for direct answers"""
    
    # Core answer
    answer_type: AnswerType = Field(description="Type of answer being provided")
    complexity_level: AnswerComplexity = Field(description="Complexity level of the answer")
    answer_title: str = Field(description="Brief title for the answer")
    answer_content: str = Field(description="Main answer content")
    
    # Supporting information
    key_points: List[str] = Field(default=[], description="Key takeaways")
    code_examples: List[CodeExample] = Field(default=[], description="Code examples")
    commands: List[str] = Field(default=[], description="Commands to execute")
    configuration_snippets: List[str] = Field(default=[], description="Configuration examples")
    
    # Context and guidance
    prerequisites: List[str] = Field(default=[], description="Prerequisites or assumptions")
    best_practices: List[str] = Field(default=[], description="Best practices related to this topic")
    common_pitfalls: List[str] = Field(default=[], description="Common mistakes to avoid")
    troubleshooting_tips: List[str] = Field(default=[], description="Troubleshooting guidance")
    
    # Additional resources
    related_concepts: List[str] = Field(default=[], description="Related concepts to explore")
    documentation_references: List[str] = Field(default=[], description="Relevant documentation")
    next_steps: List[str] = Field(default=[], description="Suggested next steps")
    
    # Metadata
    confidence_level: float = Field(description="Confidence in this answer", ge=0.0, le=1.0)
    answer_completeness: float = Field(description="How complete this answer is", ge=0.0, le=1.0)
    requires_followup: bool = Field(description="Whether this answer might need follow-up")
    followup_suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")


class DirectAnswerNode(BaseNode):
    """
    Node that provides direct answers to simple technical queries.
    
    Designed for:
    - Simple technical questions that don't need full implementation planning
    - Conceptual explanations and how-to guidance
    - Troubleshooting help and problem-solving
    - Best practices and recommendations
    - Quick reference information
    
    Should be used when:
    - Intent is clarification or asking for explanation
    - Query is specific and self-contained
    - User needs immediate guidance rather than comprehensive planning
    - Information gathering is not needed
    """
    
    def __init__(self, **kwargs):
        super().__init__(node_name="DirectAnswer", **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=DirectAnswer)
        
        # Load domain-specific answer patterns
        self.answer_patterns = self._build_answer_patterns()
        self.quick_references = self._load_quick_references()
    
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process state through direct answer generation.
        
        This node should be used for simple queries that don't require
        full implementation planning or extensive information gathering.
        """
        # Check if this is appropriate for direct answer
        if not self._should_provide_direct_answer(state):
            # Return state unchanged if not appropriate for direct answer
            return state
        
        # Check cache first
        cache_key = self._get_cache_key(
            state,
            [
                state.get('current_input', ''),
                state.get('intent', ''),
                str(len(state.get('conversation_history', [])))
            ]
        )
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return self._apply_cached_result(state, cached_result)
        
        # Generate direct answer
        direct_answer = await self._generate_direct_answer(state)
        
        # Update state with direct answer
        updated_state = self._update_state_with_answer(state, direct_answer)
        
        # Cache the result
        self._set_cache(cache_key, direct_answer)
        
        return updated_state
    
    async def stream_process(self, state: TechnicalConversationState) -> AsyncGenerator[str, None]:
        """
        Stream direct answer generation in real-time.
        
        Args:
            state: Current conversation state
            
        Yields:
            str: Text chunks of the direct answer response
        """
        # Check if this is appropriate for direct answer
        if not self._should_provide_direct_answer(state):
            yield "Let me analyze your request further to provide a comprehensive answer."
            return
        
        # Build prompts for streaming
        system_prompt = self._build_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        # Stream the LLM response
        try:
            async for chunk in self.stream_llm(
                system_prompt=system_prompt,
                user_message=user_message
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Error during streaming direct answer generation: {e}")
            yield f"Error generating response: {str(e)}"
    
    def _should_provide_direct_answer(self, state: TechnicalConversationState) -> bool:
        """Determine if this query should get a direct answer"""
        
        # Check intent - direct answers are appropriate for certain intents
        intent = state.get('intent')
        direct_answer_intents = {
            IntentType.CLARIFICATION,
            IntentType.ASKING_FOLLOWUP
        }
        
        if intent in direct_answer_intents:
            return True
        
        # Check if this is a simple, self-contained query
        current_input = state.get('current_input', '').lower()
        
        # Patterns that indicate direct answer is appropriate
        simple_patterns = [
            'what is', 'what are', 'how do i', 'how to', 'can you explain',
            'difference between', 'best practice', 'recommend', 'should i use',
            'error', 'problem', 'issue', 'troubleshoot', 'debug',
            'example', 'show me', 'syntax', 'command', 'configure'
        ]
        
        # Complex patterns that indicate implementation planning needed
        complex_patterns = [
            'build', 'create', 'develop', 'implement', 'design', 'architect',
            'solution', 'system', 'infrastructure', 'deploy', 'setup entire',
            'project', 'application', 'platform'
        ]
        
        has_simple_pattern = any(pattern in current_input for pattern in simple_patterns)
        has_complex_pattern = any(pattern in current_input for pattern in complex_patterns)
        
        # If it has simple patterns and no complex patterns, it's a direct answer candidate
        if has_simple_pattern and not has_complex_pattern:
            return True
        
        # Check query length - very short queries are often direct answer candidates
        if len(current_input.split()) <= 8 and has_simple_pattern:
            return True
        
        # Check if sufficiency assessment indicates we can't proceed with implementation
        # but the user is asking for clarification or explanation
        sufficiency_info = state.get('accumulated_info', {}).get('sufficiency_assessment', {})
        if (not sufficiency_info.get('can_proceed', False) and 
            intent == IntentType.CLARIFICATION):
            return True
        
        return False
    
    async def _generate_direct_answer(self, state: TechnicalConversationState) -> DirectAnswer:
        """Generate direct answer for the query"""
        
        system_prompt = self._build_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        direct_answer = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_parser=self.output_parser
        )
        
        # Enhance answer with domain-specific information
        direct_answer = self._enhance_answer_with_domain_knowledge(direct_answer, state)
        
        return direct_answer
    
    def _build_system_prompt(self, domain: str) -> str:
        """Build system prompt for direct answer generation"""
        
        domain_context = self.get_domain_context(domain)
        answer_patterns = self.answer_patterns.get(domain, {})
        quick_refs = self.quick_references.get(domain, {})
        
        base_prompt = f"""You are an expert {domain} consultant providing direct, actionable answers to technical questions.

Your task is to provide clear, concise, and immediately useful answers without requiring extensive information gathering.

DOMAIN: {domain.upper()}
EXPERTISE: {', '.join(domain_context.get('focus_areas', []))}
TECHNOLOGIES: {', '.join(domain_context.get('technologies', []))}

DIRECT ANSWER PRINCIPLES:

1. IMMEDIATE VALUE:
   - Provide actionable information the user can use right away
   - Focus on answering the specific question asked
   - Include practical examples and code snippets where helpful
   - Give clear, step-by-step guidance for how-to questions

2. ANSWER TYPES & FORMATS:
   - EXPLANATION: Clear, conceptual explanations with examples
   - HOW_TO: Step-by-step instructions with commands/code
   - TROUBLESHOOTING: Problem identification and solutions
   - BEST_PRACTICE: Recommendations with reasoning
   - COMPARISON: Side-by-side analysis of options
   - REFERENCE: Quick reference information and specs

3. STRUCTURE AND CLARITY:
   - Start with the direct answer to the question
   - Provide supporting details and context
   - Include relevant examples (code, commands, configs)
   - Add best practices and common pitfalls
   - Suggest next steps or related concepts when helpful

4. TECHNICAL DEPTH:
   - Match the complexity to what the user is asking
   - Don't over-explain simple concepts
   - Provide sufficient depth for meaningful guidance
   - Include prerequisites when they're important
   - Reference authoritative sources when available"""

        # Add domain-specific guidance
        if domain == "cloud":
            base_prompt += """

CLOUD DOMAIN GUIDANCE:
- Provide platform-specific examples (Azure, AWS, GCP)
- Include cost considerations when relevant
- Address security and compliance aspects
- Mention managed services options
- Include monitoring and operational guidance
- Provide Infrastructure as Code examples when applicable

COMMON CLOUD ANSWER PATTERNS:
- Service comparisons with use cases
- Configuration examples with best practices
- Security implementation guidance
- Cost optimization recommendations
- Scalability and performance guidance
- Integration patterns and examples"""

        elif domain == "network":
            base_prompt += """

NETWORK DOMAIN GUIDANCE:
- Provide vendor-specific examples (Cisco, Palo Alto, Fortinet)
- Include security implications of configurations
- Address performance and bandwidth considerations
- Mention redundancy and high availability
- Include troubleshooting commands and techniques
- Provide configuration examples with explanations

COMMON NETWORK ANSWER PATTERNS:
- Configuration syntax and examples
- Troubleshooting methodologies
- Security policy implementations
- Performance tuning recommendations
- Protocol explanations with practical context
- Network design patterns and best practices"""

        elif domain == "devops":
            base_prompt += """

DEVOPS DOMAIN GUIDANCE:
- Provide CI/CD pipeline examples
- Include containerization and orchestration guidance
- Address automation and Infrastructure as Code
- Mention monitoring and observability practices
- Include security scanning and compliance
- Provide tool-specific examples and configurations

COMMON DEVOPS ANSWER PATTERNS:
- Pipeline configuration examples
- Container and Kubernetes guidance
- Infrastructure automation scripts
- Monitoring setup and best practices
- Security integration examples
- Tool comparisons and recommendations"""

        elif domain == "dev":
            base_prompt += """

DEVELOPMENT DOMAIN GUIDANCE:
- Provide code examples in relevant languages
- Include API design and integration patterns
- Address testing and quality assurance
- Mention security and performance considerations
- Include database and data modeling guidance
- Provide framework-specific examples and patterns

COMMON DEVELOPMENT ANSWER PATTERNS:
- Code examples with explanations
- API design and implementation guidance
- Database query and schema examples
- Testing strategies and examples
- Security implementation patterns
- Framework and library usage examples"""

        base_prompt += f"""

5. ANSWER QUALITY GUIDELINES:

COMPLETENESS:
- Address the core question directly and completely
- Provide enough context for understanding
- Include practical examples when they help
- Mention important prerequisites or assumptions
- Suggest relevant next steps or follow-ups

ACCURACY:
- Provide current, accurate technical information
- Reference established best practices
- Include version information when relevant
- Mention important limitations or caveats
- Cite authoritative sources when available

USABILITY:
- Use clear, accessible language
- Format code and commands properly
- Organize information logically
- Highlight key points and takeaways
- Make it easy to scan and find information

6. DOMAIN-SPECIFIC QUICK REFERENCES:"""

        # Add domain-specific quick references
        for category, items in quick_refs.items():
            base_prompt += f"\n\n{category.upper()}:"
            for item in items[:5]:  # Limit to 5 items per category
                base_prompt += f"\n- {item}"

        base_prompt += f"""

{self.output_parser.get_format_instructions()}"""
        
        return base_prompt
    
    def _build_user_message(self, state: TechnicalConversationState) -> str:
        """Build user message for direct answer generation"""
        
        message_parts = [
            f"QUESTION: {state['current_input']}",
            f"DOMAIN: {state['domain']}",
            f"INTENT: {state.get('intent', 'unknown')}"
        ]
        
        # Add conversation context if this is a follow-up
        if state.get('conversation_history'):
            recent_history = state['conversation_history'][-2:]  # Last 2 exchanges
            if recent_history:
                message_parts.append("CONVERSATION CONTEXT:")
                for exchange in recent_history:
                    if isinstance(exchange, dict):
                        if 'user' in exchange:
                            message_parts.append(f"User: {exchange['user'][:150]}")
                        if 'bot' in exchange:
                            message_parts.append(f"Assistant: {exchange['bot'][:150]}")
        
        # Add any relevant extracted information
        if state.get('extracted_info'):
            relevant_info = []
            for category, items in state['extracted_info'].items():
                if category == 'extraction_metadata':
                    continue
                
                if isinstance(items, list) and items:
                    high_conf_items = [
                        item for item in items
                        if isinstance(item, dict) and item.get('confidence', 0) > 0.8
                    ]
                    
                    if high_conf_items:
                        item_names = [item.get('name', str(item)) for item in high_conf_items[:3]]
                        relevant_info.append(f"{category}: {', '.join(item_names)}")
            
            if relevant_info:
                message_parts.append("RELEVANT CONTEXT:")
                message_parts.extend(relevant_info[:5])
        
        # Add user expertise level hint if available
        conversation_turns = len(state.get('conversation_history', []))
        if conversation_turns > 0:
            message_parts.append(f"CONVERSATION_LENGTH: {conversation_turns} exchanges (indicates user context)")
        
        # Add channel context for response tailoring
        channel_context = state.get('channel_context', {})
        if channel_context:
            is_technical = channel_context.get('is_technical', False)
            message_parts.append(f"AUDIENCE: {'Technical team' if is_technical else 'General audience'}")
        
        return "\n\n".join(message_parts)
    
    def _enhance_answer_with_domain_knowledge(
        self, 
        answer: DirectAnswer, 
        state: TechnicalConversationState
    ) -> DirectAnswer:
        """Enhance answer with domain-specific knowledge"""
        
        domain = state['domain']
        
        # Add domain-specific best practices if not comprehensive
        if len(answer.best_practices) < 2:
            domain_practices = self._get_domain_best_practices(domain, answer.answer_type)
            answer.best_practices.extend(domain_practices[:3])
        
        # Add common pitfalls if not present
        if len(answer.common_pitfalls) < 2:
            domain_pitfalls = self._get_domain_pitfalls(domain, answer.answer_type)
            answer.common_pitfalls.extend(domain_pitfalls[:2])
        
        # Add related concepts for learning
        if len(answer.related_concepts) < 3:
            domain_concepts = self._get_related_concepts(domain, state.get('current_input', ''))
            answer.related_concepts.extend(domain_concepts[:3])
        
        # Enhance troubleshooting tips for troubleshooting answers
        if (answer.answer_type == AnswerType.TROUBLESHOOTING and 
            len(answer.troubleshooting_tips) < 3):
            domain_troubleshooting = self._get_domain_troubleshooting_tips(domain)
            answer.troubleshooting_tips.extend(domain_troubleshooting[:3])
        
        return answer
    
    def _get_domain_best_practices(self, domain: str, answer_type: AnswerType) -> List[str]:
        """Get domain-specific best practices"""
        
        practices = {
            "cloud": {
                AnswerType.EXPLANATION: [
                    "Use managed services to reduce operational overhead",
                    "Implement proper IAM and security controls",
                    "Design for scalability and cost optimization"
                ],
                AnswerType.HOW_TO: [
                    "Use Infrastructure as Code for reproducible deployments",
                    "Implement proper monitoring and alerting",
                    "Follow security best practices from the start"
                ],
                AnswerType.TROUBLESHOOTING: [
                    "Check logs in CloudWatch/Azure Monitor first", 
                    "Verify IAM permissions and security groups",
                    "Use cloud provider troubleshooting tools"
                ]
            },
            "network": {
                AnswerType.EXPLANATION: [
                    "Always consider security implications",
                    "Design with redundancy in mind",
                    "Document all network changes"
                ],
                AnswerType.HOW_TO: [
                    "Test configurations in lab environment first",
                    "Implement changes during maintenance windows",
                    "Keep rollback procedures ready"
                ],
                AnswerType.TROUBLESHOOTING: [
                    "Start with basic connectivity tests",
                    "Check logs and interface status",
                    "Verify routing and DNS resolution"
                ]
            },
            "devops": {
                AnswerType.EXPLANATION: [
                    "Automate everything possible",
                    "Implement proper CI/CD practices",
                    "Use Infrastructure as Code"
                ],
                AnswerType.HOW_TO: [
                    "Version control all configurations",
                    "Implement proper testing in pipelines",
                    "Use containerization for consistency"
                ],
                AnswerType.TROUBLESHOOTING: [
                    "Check pipeline logs and artifacts",
                    "Verify environment configurations",
                    "Use monitoring and alerting data"
                ]
            },
            "dev": {
                AnswerType.EXPLANATION: [
                    "Follow secure coding practices",
                    "Write comprehensive tests",
                    "Use proper error handling"
                ],
                AnswerType.HOW_TO: [
                    "Use version control and code reviews",
                    "Implement proper logging and monitoring",
                    "Follow API design best practices"
                ],
                AnswerType.TROUBLESHOOTING: [
                    "Check application logs and error messages",
                    "Use debugging tools and profilers",
                    "Verify configuration and dependencies"
                ]
            }
        }
        
        return practices.get(domain, {}).get(answer_type, [])
    
    def _get_domain_pitfalls(self, domain: str, answer_type: AnswerType) -> List[str]:
        """Get common pitfalls for domain and answer type"""
        
        pitfalls = {
            "cloud": [
                "Not implementing proper cost controls",
                "Ignoring security best practices", 
                "Not planning for disaster recovery"
            ],
            "network": [
                "Not testing changes before implementation",
                "Forgetting to update documentation",
                "Not considering security implications"
            ],
            "devops": [
                "Not implementing proper testing in pipelines",
                "Hardcoding secrets in configurations",
                "Not monitoring pipeline performance"
            ],
            "dev": [
                "Not validating user input properly",
                "Not implementing proper error handling",
                "Not considering performance implications"
            ]
        }
        
        return pitfalls.get(domain, [])
    
    def _get_related_concepts(self, domain: str, query: str) -> List[str]:
        """Get related concepts based on domain and query"""
        
        query_lower = query.lower()
        
        concepts = {
            "cloud": {
                "azure": ["Azure Resource Manager", "Azure Active Directory", "Azure Monitor"],
                "aws": ["EC2", "S3", "CloudWatch", "IAM"],
                "security": ["IAM", "Key Vault", "Security Groups", "RBAC"],
                "database": ["Azure SQL", "CosmosDB", "RDS", "DynamoDB"]
            },
            "network": {
                "vpn": ["IPsec", "SSL VPN", "Site-to-site", "Point-to-point"],
                "firewall": ["ACLs", "Security policies", "NAT", "Routing"],
                "cisco": ["IOS", "OSPF", "BGP", "VLANs"],
                "palo alto": ["Security policies", "NAT", "GlobalProtect", "Panorama"]
            },
            "devops": {
                "docker": ["Containers", "Dockerfile", "Docker Compose", "Container registries"],
                "kubernetes": ["Pods", "Services", "Deployments", "Helm"],
                "ci/cd": ["Jenkins", "GitHub Actions", "Azure DevOps", "GitLab CI"],
                "monitoring": ["Prometheus", "Grafana", "ELK Stack", "Observability"]
            },
            "dev": {
                "api": ["REST", "GraphQL", "OpenAPI", "Authentication"],
                "database": ["SQL", "NoSQL", "ORMs", "Database design"],
                "frontend": ["React", "Angular", "Vue.js", "JavaScript"],
                "backend": ["Node.js", "Python", "Java", "Microservices"]
            }
        }
        
        domain_concepts = concepts.get(domain, {})
        related = []
        
        for key, values in domain_concepts.items():
            if key in query_lower:
                related.extend(values)
        
        return related[:3] if related else list(domain_concepts.keys())[:3]
    
    def _get_domain_troubleshooting_tips(self, domain: str) -> List[str]:
        """Get domain-specific troubleshooting tips"""
        
        tips = {
            "cloud": [
                "Check service health status dashboards",
                "Review resource quotas and limits",
                "Verify network security group rules"
            ],
            "network": [
                "Use ping and traceroute for basic connectivity",
                "Check interface status and utilization",
                "Review routing tables and protocols"
            ],
            "devops": [
                "Check build logs for specific error messages",
                "Verify environment variable configurations",
                "Test deployment scripts in isolated environment"
            ],
            "dev": [
                "Use browser developer tools for frontend issues",
                "Check server logs for API errors",
                "Use debugging tools and breakpoints"
            ]
        }
        
        return tips.get(domain, [])
    
    def _update_state_with_answer(
        self, 
        state: TechnicalConversationState, 
        answer: DirectAnswer
    ) -> TechnicalConversationState:
        """Update state with direct answer"""
        
        updated_state = state.copy()
        
        # Format the answer as a response
        response = self._format_direct_answer_response(answer)
        updated_state['response'] = response
        updated_state['response_type'] = ResponseType.CLARIFICATION  # Direct answers are clarifications
        
        # Store answer metadata
        accumulated_info = updated_state.get('accumulated_info', {})
        accumulated_info['direct_answer'] = {
            'answer_type': answer.answer_type.value,
            'complexity_level': answer.complexity_level.value,
            'answer_title': answer.answer_title,
            'confidence_level': answer.confidence_level,
            'answer_completeness': answer.answer_completeness,
            'requires_followup': answer.requires_followup,
            'code_examples_count': len(answer.code_examples),
            'commands_count': len(answer.commands)
        }
        updated_state['accumulated_info'] = accumulated_info
        
        # Update conversation stage
        updated_state['stage'] = ConversationStage.SUFFICIENT
        
        # Update token usage
        token_usage = updated_state.get('token_usage', {}).copy()
        token_usage['direct_answer'] = token_usage.get('direct_answer', 0) + 300
        updated_state['token_usage'] = token_usage
        
        return updated_state
    
    async def stream_process(self, state: TechnicalConversationState) -> AsyncGenerator[str, None]:
        """
        Stream direct answer generation with real-time response delivery.
        
        Args:
            state: Current conversation state
            
        Yields:
            str: Text chunks for progressive response generation
        """
        # Check if this is appropriate for direct answer
        if not self._should_provide_direct_answer(state):
            yield "I think this requires a more comprehensive solution. "
            yield "Let me gather more information to provide you with a detailed implementation plan."
            return
        
        # Build the prompt for streaming generation
        system_prompt = self._build_streaming_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        try:
            # Stream the direct answer generation
            async for chunk in self.stream_llm(
                system_prompt=system_prompt,
                user_message=user_message
            ):
                if chunk and chunk.strip():
                    yield chunk
                    # Small delay for better streaming experience
                    await asyncio.sleep(0.05)
                    
        except Exception as e:
            yield f"\nI encountered an error while generating the answer: {str(e)}\n"
            yield "Please try rephrasing your question or provide more context."
    
    def _build_streaming_system_prompt(self, domain: str) -> str:
        """Build system prompt optimized for streaming direct answers"""
        
        domain_context = self.get_domain_context(domain)
        
        base_prompt = f"""You are an expert {domain} consultant providing direct, actionable answers to technical questions.

Your task is to provide clear, concise, and immediately useful answers without requiring extensive information gathering.

DOMAIN: {domain.upper()}
EXPERTISE: {', '.join(domain_context.get('focus_areas', []))}
TECHNOLOGIES: {', '.join(domain_context.get('technologies', []))}

STREAMING RESPONSE FORMAT:
Generate your response as clear, well-structured content that includes:

1. Start with a direct answer to the question
2. Provide supporting details and context
3. Include practical examples (code, commands, configs) when helpful
4. Add best practices and common pitfalls to avoid
5. Suggest next steps or related concepts when relevant

Write in a conversational yet professional tone, as if explaining to a technical colleague.
Use markdown formatting for structure and readability.
Be comprehensive but concise - focus on actionable guidance."""

        # Add domain-specific guidance for streaming
        if domain == "cloud":
            base_prompt += """

CLOUD RESPONSE FOCUS:
- Provide platform-specific examples (Azure, AWS, GCP)
- Include cost and security considerations when relevant
- Mention managed services options
- Include monitoring and operational guidance"""

        elif domain == "network":
            base_prompt += """

NETWORK RESPONSE FOCUS:
- Provide vendor-specific examples (Cisco, Palo Alto, Fortinet)
- Include security implications of configurations
- Address performance and bandwidth considerations
- Include troubleshooting commands and techniques"""

        elif domain == "devops":
            base_prompt += """

DEVOPS RESPONSE FOCUS:
- Provide CI/CD pipeline examples
- Include containerization and orchestration guidance
- Address automation and Infrastructure as Code
- Include monitoring and observability practices"""

        elif domain == "dev":
            base_prompt += """

DEVELOPMENT RESPONSE FOCUS:
- Provide code examples in relevant languages
- Include API design and integration patterns
- Address testing and security considerations
- Include database and data modeling guidance"""

        return base_prompt

    def _format_direct_answer_response(self, answer: DirectAnswer) -> str:
        """Format direct answer as user-friendly response"""
        
        response_parts = [
            f"# {answer.answer_title}",
            "",
            answer.answer_content
        ]
        
        # Add key points if available
        if answer.key_points:
            response_parts.extend([
                "",
                "## Key Points:",
                "- " + "\n- ".join(answer.key_points)
            ])
        
        # Add code examples if available
        if answer.code_examples:
            response_parts.extend([
                "",
                "## Examples:"
            ])
            
            for example in answer.code_examples[:3]:  # Limit to 3 examples
                if example.description:
                    response_parts.append(f"\n**{example.description}:**")
                
                response_parts.append(f"```{example.language}\n{example.code}\n```")
        
        # Add commands if available
        if answer.commands:
            response_parts.extend([
                "",
                "## Commands:"
            ])
            
            for cmd in answer.commands[:5]:  # Limit to 5 commands
                response_parts.append(f"```bash\n{cmd}\n```")
        
        # Add configuration snippets
        if answer.configuration_snippets:
            response_parts.extend([
                "",
                "## Configuration:"
            ])
            
            for config in answer.configuration_snippets[:3]:  # Limit to 3 configs
                response_parts.append(f"```\n{config}\n```")
        
        # Add best practices
        if answer.best_practices:
            response_parts.extend([
                "",
                "## Best Practices:",
                "- " + "\n- ".join(answer.best_practices[:5])
            ])
        
        # Add common pitfalls
        if answer.common_pitfalls:
            response_parts.extend([
                "",
                "## Common Pitfalls to Avoid:",
                "- " + "\n- ".join(answer.common_pitfalls[:3])
            ])
        
        # Add troubleshooting tips for troubleshooting answers
        if (answer.answer_type == AnswerType.TROUBLESHOOTING and 
            answer.troubleshooting_tips):
            response_parts.extend([
                "",
                "## Troubleshooting Tips:",
                "- " + "\n- ".join(answer.troubleshooting_tips[:4])
            ])
        
        # Add next steps
        if answer.next_steps:
            response_parts.extend([
                "",
                "## Next Steps:",
                "- " + "\n- ".join(answer.next_steps[:4])
            ])
        
        # Add related concepts
        if answer.related_concepts:
            response_parts.extend([
                "",
                "## Related Concepts:",
                "- " + "\n- ".join(answer.related_concepts[:5])
            ])
        
        # Add follow-up suggestions if this might need follow-up
        if answer.requires_followup and answer.followup_suggestions:
            response_parts.extend([
                "",
                "## You Might Also Want to Ask:",
                "- " + "\n- ".join(answer.followup_suggestions[:3])
            ])
        
        return "\n".join(response_parts)
    
    def _build_answer_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build domain-specific answer patterns"""
        
        return {
            "cloud": {
                "explanation_patterns": ["service comparison", "architecture overview", "cost analysis"],
                "howto_patterns": ["deployment guide", "configuration setup", "migration steps"],
                "troubleshooting_patterns": ["connectivity issues", "permission errors", "performance problems"]
            },
            "network": {
                "explanation_patterns": ["protocol explanation", "topology overview", "security concepts"],
                "howto_patterns": ["device configuration", "network setup", "troubleshooting steps"],
                "troubleshooting_patterns": ["connectivity problems", "routing issues", "security blocks"]
            },
            "devops": {
                "explanation_patterns": ["pipeline overview", "tool comparison", "best practices"],
                "howto_patterns": ["pipeline setup", "deployment automation", "monitoring implementation"],
                "troubleshooting_patterns": ["build failures", "deployment issues", "performance problems"]
            },
            "dev": {
                "explanation_patterns": ["concept explanation", "framework comparison", "design patterns"],
                "howto_patterns": ["code examples", "API implementation", "database setup"],
                "troubleshooting_patterns": ["debugging techniques", "error resolution", "performance issues"]
            }
        }
    
    def _load_quick_references(self) -> Dict[str, Dict[str, List[str]]]:
        """Load quick reference information by domain"""
        
        return {
            "cloud": {
                "azure_services": [
                    "Virtual Machines (IaaS compute)",
                    "App Service (PaaS web apps)",
                    "Azure SQL Database (managed database)",
                    "Storage Account (blob, file, queue storage)",
                    "Virtual Network (networking and security)"
                ],
                "aws_services": [
                    "EC2 (elastic compute instances)",
                    "S3 (object storage service)",
                    "RDS (managed relational database)",
                    "Lambda (serverless functions)",
                    "VPC (virtual private cloud)"
                ]
            },
            "network": {
                "cisco_commands": [
                    "show ip route (display routing table)",
                    "show interfaces (interface status)",
                    "show version (device information)",
                    "configure terminal (enter config mode)",
                    "show running-config (current configuration)"
                ],
                "palo_alto_concepts": [
                    "Security Policy (traffic control rules)",
                    "NAT Policy (address translation)",
                    "GlobalProtect (VPN solution)",
                    "Zones (security boundaries)",
                    "Virtual Router (routing instance)"
                ]
            },
            "devops": {
                "docker_commands": [
                    "docker build (build image from Dockerfile)",
                    "docker run (run container from image)",
                    "docker ps (list running containers)",
                    "docker logs (view container logs)",
                    "docker exec (execute command in container)"
                ],
                "kubernetes_concepts": [
                    "Pod (smallest deployable unit)",
                    "Service (network abstraction for pods)",
                    "Deployment (manages replica sets)",
                    "ConfigMap (configuration data)",
                    "Namespace (resource isolation)"
                ]
            },
            "dev": {
                "http_status_codes": [
                    "200 OK (successful request)",
                    "400 Bad Request (client error)",
                    "401 Unauthorized (authentication required)",
                    "404 Not Found (resource not found)",
                    "500 Internal Server Error (server error)"
                ],
                "api_best_practices": [
                    "Use RESTful URL patterns",
                    "Implement proper HTTP status codes",
                    "Include API versioning strategy",
                    "Use consistent request/response formats",
                    "Implement proper authentication and authorization"
                ]
            }
        }
    
    def _apply_cached_result(
        self, 
        state: TechnicalConversationState, 
        cached_answer: DirectAnswer
    ) -> TechnicalConversationState:
        """Apply cached direct answer to state"""
        
        # Record cache hit
        cache_hits = state.get('cache_hits', []).copy()
        cache_hits.append(f"direct_answer:{state['updated_at']}")
        
        updated_state = self._update_state_with_answer(state, cached_answer)
        updated_state['cache_hits'] = cache_hits
        
        return updated_state
    
    def _validate_output_state(self, state: TechnicalConversationState) -> None:
        """Validate output state for direct answer node"""
        super()._validate_output_state(state)
        
        # Ensure direct answer was generated
        if state.get('response_type') == ResponseType.IMPLEMENTATION:
            if not state.get('response') or len(state.get('response', '')) < 50:
                raise NodeValidationError("Direct answer should be generated but response is insufficient")
            
            if 'accumulated_info' not in state or 'direct_answer' not in state['accumulated_info']:
                raise NodeValidationError("Direct answer metadata missing from state")