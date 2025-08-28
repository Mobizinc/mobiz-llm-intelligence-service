"""
Sufficiency Checker Node and Question Generator Implementation

This module implements the SufficiencyCheckerNode and QuestionGenerator that
determine if enough information has been gathered and generate intelligent
clarification questions when needed.

Part of Epic 2: Agent Nodes Implementation
Story 2.4: Implement SufficiencyCheckerNode with QuestionGenerator
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from .base import BaseNode
from ..models.conversation_state import TechnicalConversationState, ConversationStage, ResponseType


class SufficiencyLevel(str, Enum):
    """Levels of information sufficiency"""
    INSUFFICIENT = "insufficient"      # Major gaps, cannot proceed
    PARTIAL = "partial"               # Some gaps, can proceed with assumptions
    SUFFICIENT = "sufficient"         # Enough info for basic solution
    COMPREHENSIVE = "comprehensive"   # Detailed info for optimal solution


class QuestionType(str, Enum):
    """Types of clarification questions"""
    REQUIREMENT = "requirement"       # Clarify requirements
    TECHNICAL = "technical"          # Technical specifications
    CONSTRAINT = "constraint"        # Limitations and constraints
    PREFERENCE = "preference"        # User preferences and priorities
    ENVIRONMENT = "environment"      # Environmental context
    VALIDATION = "validation"        # Confirm understanding


class Question(BaseModel):
    """Represents a clarification question"""
    
    id: str = Field(description="Unique question identifier")
    text: str = Field(description="The question text")
    type: QuestionType = Field(description="Type of question")
    priority: int = Field(description="Priority (1=highest, 5=lowest)")
    
    # Context and targeting
    missing_info: str = Field(description="What information this question seeks")
    domain_category: str = Field(description="Domain-specific category")
    
    # Question generation metadata
    reasoning: str = Field(description="Why this question is important")
    expected_answer_type: str = Field(description="Expected type of answer")
    follow_up_questions: List[str] = Field(default=[], description="Potential follow-up questions")
    
    # Options for multiple choice questions
    suggested_options: List[str] = Field(default=[], description="Suggested answer options")
    allows_custom_answer: bool = Field(default=True, description="Whether custom answers are allowed")


class SufficiencyAssessment(BaseModel):
    """Structured output for sufficiency assessment"""
    
    sufficiency_level: SufficiencyLevel = Field(description="Overall sufficiency level")
    sufficiency_score: float = Field(
        description="Sufficiency score from 0-100%",
        ge=0.0,
        le=100.0
    )
    
    # Detailed analysis
    complete_areas: List[str] = Field(default=[], description="Areas with sufficient information")
    partial_areas: List[str] = Field(default=[], description="Areas with some information")
    missing_areas: List[str] = Field(default=[], description="Areas with no information")
    
    # Critical gaps
    critical_gaps: List[str] = Field(default=[], description="Critical information gaps")
    blocking_gaps: List[str] = Field(default=[], description="Gaps that block implementation")
    nice_to_have_gaps: List[str] = Field(default=[], description="Gaps that would improve solution")
    
    # Decision and next steps
    can_proceed: bool = Field(description="Whether implementation can proceed")
    proceed_with_assumptions: bool = Field(description="Whether to proceed with assumptions")
    assumptions_needed: List[str] = Field(default=[], description="Assumptions needed to proceed")
    
    # Confidence and metadata
    assessment_confidence: float = Field(
        description="Confidence in this assessment",
        ge=0.0,
        le=1.0
    )
    recommendation: str = Field(description="Recommendation for next steps")


class GeneratedQuestions(BaseModel):
    """Structured output for question generation"""
    
    questions: List[Question] = Field(description="Generated clarification questions")
    
    # Question strategy
    question_strategy: str = Field(description="Strategy used for question selection")
    max_questions_per_turn: int = Field(description="Maximum questions to ask at once")
    
    # Prioritization
    high_priority_questions: List[str] = Field(default=[], description="Question IDs that are high priority")
    optional_questions: List[str] = Field(default=[], description="Question IDs that are optional")
    
    # Multi-turn strategy
    current_turn_questions: List[str] = Field(default=[], description="Questions for current turn")
    future_turn_questions: List[str] = Field(default=[], description="Questions for future turns")
    
    # Generation metadata
    generation_confidence: float = Field(
        description="Confidence in question quality",
        ge=0.0,
        le=1.0
    )
    coverage_analysis: str = Field(description="Analysis of information coverage")


class SufficiencyCheckerNode(BaseNode):
    """
    Node that checks if sufficient information has been gathered.
    
    Supports:
    - Rule-based and ML-based sufficiency checking
    - Domain-specific requirement templates
    - Sufficiency score calculation (0-100%)
    - Critical gap identification
    - Decision on whether to proceed or gather more info
    """
    
    def __init__(self, sufficiency_threshold: float = 70.0, **kwargs):
        super().__init__(node_name="SufficiencyChecker", **kwargs)
        self.sufficiency_threshold = sufficiency_threshold
        self.output_parser = PydanticOutputParser(pydantic_object=SufficiencyAssessment)
        
        # Load domain-specific sufficiency rules
        self.sufficiency_rules = self._build_sufficiency_rules()
    
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process state through sufficiency assessment.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with sufficiency assessment results
        """
        # Check cache first
        cache_key = self._get_cache_key(
            state,
            [
                str(hash(str(state.get('requirements', [])))),
                str(hash(str(state.get('extracted_info', {})))),
                str(self.sufficiency_threshold)
            ]
        )
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return self._apply_cached_result(state, cached_result)
        
        # Assess sufficiency
        assessment = await self._assess_sufficiency(state)
        
        # Update state with assessment results
        updated_state = self._update_state_with_assessment(state, assessment)
        
        # Cache the result
        self._set_cache(cache_key, assessment)
        
        return updated_state
    
    async def _assess_sufficiency(self, state: TechnicalConversationState) -> SufficiencyAssessment:
        """Assess whether information is sufficient for implementation"""
        
        system_prompt = self._build_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        assessment = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_parser=self.output_parser
        )
        
        # Enhance assessment with rule-based checks
        assessment = self._enhance_with_rule_based_checks(assessment, state)
        
        return assessment
    
    def _build_system_prompt(self, domain: str) -> str:
        """Build system prompt for sufficiency assessment"""
        
        domain_context = self.get_domain_context(domain)
        sufficiency_rules = self.sufficiency_rules.get(domain, {})
        
        base_prompt = f"""You are an expert solution architect for {domain} implementations.

Your task is to assess whether enough information has been gathered to provide a high-quality technical solution.

DOMAIN: {domain.upper()}
SUFFICIENCY THRESHOLD: {self.sufficiency_threshold}%
COMMON TECHNOLOGIES: {', '.join(domain_context.get('technologies', []))}

SUFFICIENCY ASSESSMENT FRAMEWORK:

1. SUFFICIENCY LEVELS:
   - INSUFFICIENT (0-40%): Major gaps, cannot provide useful guidance
   - PARTIAL (40-70%): Some gaps, can provide basic guidance with assumptions
   - SUFFICIENT (70-85%): Good information, can provide solid recommendations
   - COMPREHENSIVE (85-100%): Excellent detail, can provide optimal solutions

2. CRITICAL ASSESSMENT AREAS:"""

        # Add domain-specific assessment areas
        if domain == "cloud":
            base_prompt += """
   
   CLOUD DOMAIN ASSESSMENT AREAS:
   - PLATFORM CHOICE: Which cloud provider(s) and services
   - ARCHITECTURE: Application architecture and components
   - SCALABILITY: Expected load and scaling requirements
   - SECURITY: Security requirements and compliance needs
   - COST: Budget constraints and optimization requirements
   - INTEGRATION: Existing systems and data migration needs
   - OPERATIONAL: Monitoring, backup, and operational requirements"""

        elif domain == "network":
            base_prompt += """
   
   NETWORK DOMAIN ASSESSMENT AREAS:
   - CONNECTIVITY: What needs to connect to what
   - TOPOLOGY: Network topology and design requirements
   - DEVICES: Specific network devices and vendors
   - BANDWIDTH: Performance and capacity requirements
   - SECURITY: Security policies and access controls
   - REDUNDANCY: High availability and failover requirements
   - COMPLIANCE: Regulatory and organizational requirements"""

        elif domain == "devops":
            base_prompt += """
   
   DEVOPS DOMAIN ASSESSMENT AREAS:
   - PLATFORM: Deployment platforms and environments
   - PIPELINE: CI/CD requirements and existing tooling
   - INFRASTRUCTURE: Infrastructure automation needs
   - CONTAINERIZATION: Container and orchestration requirements
   - MONITORING: Logging, metrics, and alerting needs
   - TESTING: Testing strategy and automation
   - SECURITY: Security scanning and compliance requirements"""

        elif domain == "dev":
            base_prompt += """
   
   DEVELOPMENT DOMAIN ASSESSMENT AREAS:
   - FUNCTIONALITY: What the application should do
   - TECHNOLOGY: Programming languages and frameworks
   - ARCHITECTURE: Application architecture and patterns
   - DATA: Data storage and management requirements
   - INTEGRATION: APIs and third-party integrations
   - UI/UX: User interface and experience requirements
   - SECURITY: Authentication, authorization, and data protection"""

        base_prompt += f"""

3. ASSESSMENT METHODOLOGY:

   COMPLETENESS CHECK:
   - Evaluate each critical area for information completeness
   - Score each area: Complete (100%), Partial (50%), Missing (0%)
   - Weight areas by importance for domain
   - Calculate overall sufficiency score

   CRITICAL GAP ANALYSIS:
   - Identify gaps that completely block implementation
   - Identify gaps that significantly impact solution quality
   - Identify nice-to-have information that would improve solution
   
   ASSUMPTION FEASIBILITY:
   - Evaluate whether reasonable assumptions can fill gaps
   - Consider industry best practices and common patterns
   - Assess risk of making assumptions vs. gathering more info

4. DECISION CRITERIA:

   CAN PROCEED (sufficiency >= {self.sufficiency_threshold}%):
   - All critical areas have at least partial information
   - No blocking gaps that prevent basic implementation
   - Can make reasonable assumptions for missing details
   
   NEED MORE INFO (sufficiency < {self.sufficiency_threshold}%):
   - Critical gaps in core requirements
   - Insufficient information for even basic guidance
   - Too many assumptions needed, high risk of wrong solution

5. DOMAIN-SPECIFIC RULES:"""

        # Add domain-specific rules
        domain_rules = sufficiency_rules.get('assessment_rules', [])
        for rule in domain_rules[:6]:
            base_prompt += f"\n   - {rule}"

        base_prompt += f"""

ASSESSMENT GUIDELINES:
- Be thorough but practical - perfect information is rarely available
- Consider the user's expertise level and context
- Weight critical functional requirements higher than nice-to-haves
- Account for industry standards and common patterns
- Consider implementation complexity and risk factors

{self.output_parser.get_format_instructions()}"""
        
        return base_prompt
    
    def _build_user_message(self, state: TechnicalConversationState) -> str:
        """Build user message for sufficiency assessment"""
        
        message_parts = [
            f"ORIGINAL REQUEST: {state['initial_query']}",
            f"DOMAIN: {state['domain']}",
            f"CONVERSATION STAGE: {state.get('stage', 'initial')}",
            f"INTENT: {state.get('intent', 'unknown')}"
        ]
        
        # Add requirements summary
        if state.get('requirements'):
            req_summary = []
            for req in state['requirements'][:8]:  # Limit to 8 requirements
                if isinstance(req, dict):
                    req_type = req.get('type', 'unknown')
                    criticality = req.get('criticality', 'unknown')
                    confidence = req.get('confidence', 0.0)
                    req_summary.append(f"- {req.get('name', 'Unknown')}: {req_type}, {criticality} ({confidence:.2f})")
            
            if req_summary:
                message_parts.append("IDENTIFIED REQUIREMENTS:")
                message_parts.extend(req_summary)
        
        # Add extracted information summary
        if state.get('extracted_info'):
            info_summary = []
            for category, items in state['extracted_info'].items():
                if category == 'extraction_metadata':
                    continue
                
                if isinstance(items, list) and items:
                    item_count = len(items)
                    high_confidence_count = sum(
                        1 for item in items 
                        if isinstance(item, dict) and item.get('confidence', 0) > 0.7
                    )
                    info_summary.append(f"- {category}: {item_count} items ({high_confidence_count} high confidence)")
                elif items and not isinstance(items, (list, dict)):
                    info_summary.append(f"- {category}: {str(items)[:50]}")
            
            if info_summary:
                message_parts.append("EXTRACTED INFORMATION:")
                message_parts.extend(info_summary[:10])
        
        # Add conversation context
        if state.get('questions_asked'):
            message_parts.append(f"QUESTIONS ASKED: {len(state['questions_asked'])} questions")
            if len(state['questions_asked']) > 0:
                recent_questions = state['questions_asked'][-3:]
                message_parts.append("RECENT QUESTIONS:")
                for i, question in enumerate(recent_questions, 1):
                    message_parts.append(f"{i}. {question[:100]}")
        
        # Add missing fields context
        if state.get('missing_fields'):
            message_parts.append(f"IDENTIFIED MISSING FIELDS: {', '.join(state['missing_fields'][:8])}")
        
        # Add conversation history length
        if state.get('conversation_history'):
            message_parts.append(f"CONVERSATION TURNS: {len(state['conversation_history'])}")
        
        return "\n\n".join(message_parts)
    
    def _enhance_with_rule_based_checks(
        self, 
        assessment: SufficiencyAssessment, 
        state: TechnicalConversationState
    ) -> SufficiencyAssessment:
        """Enhance assessment with domain-specific rule-based checks"""
        
        domain_rules = self.sufficiency_rules.get(state['domain'], {})
        
        # Apply minimum requirements check
        min_requirements = domain_rules.get('minimum_requirements', [])
        requirements_met = self._check_minimum_requirements(state, min_requirements)
        
        # Apply critical information check
        critical_info = domain_rules.get('critical_information', [])
        critical_info_present = self._check_critical_information(state, critical_info)
        
        # Adjust sufficiency score based on rule checks
        rule_based_score = (requirements_met + critical_info_present) / 2 * 100
        
        # Combine with LLM assessment (weighted average)
        combined_score = (assessment.sufficiency_score * 0.7) + (rule_based_score * 0.3)
        assessment.sufficiency_score = min(combined_score, 100.0)
        
        # Update sufficiency level based on new score
        if assessment.sufficiency_score >= 85:
            assessment.sufficiency_level = SufficiencyLevel.COMPREHENSIVE
        elif assessment.sufficiency_score >= 70:
            assessment.sufficiency_level = SufficiencyLevel.SUFFICIENT
        elif assessment.sufficiency_score >= 40:
            assessment.sufficiency_level = SufficiencyLevel.PARTIAL
        else:
            assessment.sufficiency_level = SufficiencyLevel.INSUFFICIENT
        
        # Update can_proceed decision
        assessment.can_proceed = assessment.sufficiency_score >= self.sufficiency_threshold
        assessment.proceed_with_assumptions = (
            assessment.sufficiency_score >= (self.sufficiency_threshold - 20) and
            len(assessment.blocking_gaps) == 0
        )
        
        return assessment
    
    def _check_minimum_requirements(self, state: TechnicalConversationState, min_requirements: List[str]) -> float:
        """Check if minimum requirements are met"""
        
        if not min_requirements:
            return 1.0
        
        met_count = 0
        requirements = state.get('requirements', [])
        extracted_info = state.get('extracted_info', {})
        
        for min_req in min_requirements:
            # Check in requirements
            req_met = any(
                min_req.lower() in req.get('name', '').lower() or 
                min_req.lower() in req.get('description', '').lower()
                for req in requirements
                if isinstance(req, dict)
            )
            
            # Check in extracted info
            if not req_met:
                req_met = any(
                    min_req.lower() in str(category).lower() and 
                    isinstance(items, list) and len(items) > 0
                    for category, items in extracted_info.items()
                )
            
            if req_met:
                met_count += 1
        
        return met_count / len(min_requirements)
    
    def _check_critical_information(self, state: TechnicalConversationState, critical_info: List[str]) -> float:
        """Check if critical information is present"""
        
        if not critical_info:
            return 1.0
        
        present_count = 0
        extracted_info = state.get('extracted_info', {})
        
        for critical_item in critical_info:
            # Check if critical information is present in extracted info
            info_present = False
            
            for category, items in extracted_info.items():
                if category == 'extraction_metadata':
                    continue
                    
                if isinstance(items, list) and items:
                    for item in items:
                        if isinstance(item, dict):
                            item_text = f"{item.get('name', '')} {item.get('value', '')}".lower()
                            if critical_item.lower() in item_text:
                                info_present = True
                                break
                    
                    if info_present:
                        break
                elif critical_item.lower() in str(items).lower():
                    info_present = True
                    break
            
            if info_present:
                present_count += 1
        
        return present_count / len(critical_info)
    
    def _update_state_with_assessment(
        self, 
        state: TechnicalConversationState, 
        assessment: SufficiencyAssessment
    ) -> TechnicalConversationState:
        """Update state with sufficiency assessment results"""
        
        updated_state = state.copy()
        
        # Update conversation stage based on assessment
        if assessment.can_proceed:
            updated_state['stage'] = ConversationStage.SUFFICIENT
        elif assessment.proceed_with_assumptions:
            updated_state['stage'] = ConversationStage.ANALYZING
        else:
            updated_state['stage'] = ConversationStage.GATHERING
        
        # Update missing fields with critical gaps
        missing_fields = list(assessment.critical_gaps) + list(assessment.blocking_gaps)
        updated_state['missing_fields'] = missing_fields[:10]
        
        # Store assessment results in accumulated info
        accumulated_info = updated_state.get('accumulated_info', {})
        accumulated_info['sufficiency_assessment'] = {
            'sufficiency_level': assessment.sufficiency_level.value,
            'sufficiency_score': assessment.sufficiency_score,
            'can_proceed': assessment.can_proceed,
            'proceed_with_assumptions': assessment.proceed_with_assumptions,
            'critical_gaps_count': len(assessment.critical_gaps),
            'blocking_gaps_count': len(assessment.blocking_gaps),
            'assessment_confidence': assessment.assessment_confidence,
            'recommendation': assessment.recommendation,
            'complete_areas': assessment.complete_areas,
            'missing_areas': assessment.missing_areas
        }
        updated_state['accumulated_info'] = accumulated_info
        
        # Set response type and next action
        if assessment.can_proceed:
            updated_state['response_type'] = ResponseType.IMPLEMENTATION
        else:
            updated_state['response_type'] = ResponseType.QUESTION
        
        # Update token usage
        token_usage = updated_state.get('token_usage', {}).copy()
        token_usage['sufficiency_checker'] = token_usage.get('sufficiency_checker', 0) + 250
        updated_state['token_usage'] = token_usage
        
        return updated_state
    
    def _build_sufficiency_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build domain-specific sufficiency rules"""
        
        return {
            "cloud": {
                "minimum_requirements": [
                    "cloud_provider", "service_type", "scalability", "security"
                ],
                "critical_information": [
                    "azure", "aws", "gcp", "application", "database", "network", "storage"
                ],
                "assessment_rules": [
                    "Must identify target cloud provider (Azure/AWS/GCP)",
                    "Must have basic architecture requirements (compute/storage/network)",
                    "Must identify security and compliance needs",
                    "Should have scalability and performance requirements",
                    "Should identify integration needs with existing systems",
                    "Cost constraints are important for solution design"
                ]
            },
            "network": {
                "minimum_requirements": [
                    "connectivity", "devices", "security", "topology"
                ],
                "critical_information": [
                    "firewall", "router", "switch", "vpn", "bandwidth", "sites", "users"
                ],
                "assessment_rules": [
                    "Must identify what needs to connect (sites, users, services)",
                    "Must specify network devices or vendors involved",
                    "Must have security requirements and policies",
                    "Should have bandwidth and performance requirements",
                    "Should identify redundancy and failover needs",
                    "Must consider compliance and regulatory requirements"
                ]
            },
            "devops": {
                "minimum_requirements": [
                    "platform", "deployment", "automation", "monitoring"
                ],
                "critical_information": [
                    "docker", "kubernetes", "jenkins", "github", "azure", "aws", "ci/cd"
                ],
                "assessment_rules": [
                    "Must identify target deployment platform",
                    "Must have CI/CD pipeline requirements",
                    "Must specify automation and IaC needs",
                    "Should have testing strategy requirements",
                    "Should identify monitoring and logging needs",
                    "Security scanning and compliance are important"
                ]
            },
            "dev": {
                "minimum_requirements": [
                    "functionality", "technology", "architecture", "data"
                ],
                "critical_information": [
                    "api", "database", "frontend", "backend", "framework", "language"
                ],
                "assessment_rules": [
                    "Must identify core functionality requirements",
                    "Must specify technology stack or preferences",
                    "Must have basic architecture requirements",
                    "Should identify data storage and management needs",
                    "Should have integration and API requirements",
                    "Performance and scalability needs are important"
                ]
            }
        }
    
    def _apply_cached_result(
        self, 
        state: TechnicalConversationState, 
        cached_assessment: SufficiencyAssessment
    ) -> TechnicalConversationState:
        """Apply cached assessment result to state"""
        
        # Record cache hit
        cache_hits = state.get('cache_hits', []).copy()
        cache_hits.append(f"sufficiency_checker:{state['updated_at']}")
        
        updated_state = self._update_state_with_assessment(state, cached_assessment)
        updated_state['cache_hits'] = cache_hits
        
        return updated_state
    
    def _validate_output_state(self, state: TechnicalConversationState) -> None:
        """Validate output state for sufficiency checker"""
        super()._validate_output_state(state)
        
        # Ensure sufficiency assessment is present
        if 'accumulated_info' not in state or 'sufficiency_assessment' not in state['accumulated_info']:
            raise NodeValidationError("Sufficiency assessment missing from state")
        
        assessment = state['accumulated_info']['sufficiency_assessment']
        
        # Validate key assessment fields
        required_fields = ['sufficiency_level', 'sufficiency_score', 'can_proceed']
        for field in required_fields:
            if field not in assessment:
                raise NodeValidationError(f"Required assessment field '{field}' missing")


class QuestionGenerator(BaseNode):
    """
    Node that generates intelligent clarification questions.
    
    Supports:
    - Intelligent question generation for missing info
    - Question prioritization and ordering
    - Context-aware question phrasing
    - Multi-turn question strategy
    - Domain-specific question templates
    """
    
    def __init__(self, max_questions_per_turn: int = 3, **kwargs):
        super().__init__(node_name="QuestionGenerator", **kwargs)
        self.max_questions_per_turn = max_questions_per_turn
        self.output_parser = PydanticOutputParser(pydantic_object=GeneratedQuestions)
        
        # Load question templates and strategies
        self.question_templates = self._build_question_templates()
    
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process state through question generation.
        
        This node should only be called when sufficiency check indicates
        more information is needed.
        """
        # Check if questions are needed
        sufficiency_info = state.get('accumulated_info', {}).get('sufficiency_assessment', {})
        if sufficiency_info.get('can_proceed', False):
            # No questions needed, return state unchanged
            return state
        
        # Check cache first
        cache_key = self._get_cache_key(
            state,
            [
                str(hash(str(state.get('missing_fields', [])))),
                str(len(state.get('questions_asked', []))),
                str(self.max_questions_per_turn)
            ]
        )
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return self._apply_cached_result(state, cached_result)
        
        # Generate questions
        generated_questions = await self._generate_questions(state)
        
        # Update state with generated questions
        updated_state = self._update_state_with_questions(state, generated_questions)
        
        # Cache the result
        self._set_cache(cache_key, generated_questions)
        
        return updated_state
    
    async def _generate_questions(self, state: TechnicalConversationState) -> GeneratedQuestions:
        """Generate clarification questions based on missing information"""
        
        system_prompt = self._build_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        generated_questions = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_parser=self.output_parser
        )
        
        # Post-process questions for optimization
        generated_questions = self._optimize_questions(generated_questions, state)
        
        return generated_questions
    
    def _build_system_prompt(self, domain: str) -> str:
        """Build system prompt for question generation"""
        
        domain_context = self.get_domain_context(domain)
        templates = self.question_templates.get(domain, {})
        
        base_prompt = f"""You are an expert technical consultant for {domain} solutions.

Your task is to generate intelligent clarification questions to gather missing information needed for providing high-quality technical guidance.

DOMAIN: {domain.upper()}
MAX QUESTIONS PER TURN: {self.max_questions_per_turn}
TECHNOLOGIES: {', '.join(domain_context.get('technologies', []))}

QUESTION GENERATION PRINCIPLES:

1. PRIORITIZATION:
   - Priority 1: Critical information that blocks any solution
   - Priority 2: Important information that significantly affects solution quality
   - Priority 3: Nice-to-have information that optimizes solution
   - Priority 4: Optional details for comprehensive solutions
   - Priority 5: Background information for context

2. QUESTION TYPES:
   - REQUIREMENT: Clarify functional and business requirements
   - TECHNICAL: Gather technical specifications and constraints
   - CONSTRAINT: Identify limitations (budget, time, resources, policies)
   - PREFERENCE: Understand user preferences and priorities
   - ENVIRONMENT: Gather environmental and operational context
   - VALIDATION: Confirm understanding and assumptions

3. QUESTION QUALITY CRITERIA:
   - Specific and actionable (not vague or general)
   - Single-focused (one concept per question)
   - Context-aware (reference previous conversation)
   - User-friendly language (avoid excessive jargon)
   - Provide options when helpful
   - Build on what's already known"""

        # Add domain-specific question strategies
        if domain == "cloud":
            base_prompt += """

CLOUD DOMAIN QUESTION STRATEGY:
- Start with platform/provider choice if not specified
- Clarify application architecture and components
- Gather performance and scalability requirements
- Identify security and compliance needs
- Understand cost constraints and optimization goals
- Ask about existing systems and migration needs
- Clarify operational requirements (monitoring, backup, etc.)

EXAMPLE CLOUD QUESTIONS:
- "Which cloud provider do you prefer: Azure, AWS, or GCP?"
- "What's your expected number of concurrent users?"
- "Do you have any compliance requirements like HIPAA or SOX?"
- "What's your monthly cloud budget range?"
- "Are you migrating from existing on-premises systems?"
"""

        elif domain == "network":
            base_prompt += """

NETWORK DOMAIN QUESTION STRATEGY:
- Clarify what needs to connect (sites, users, services)
- Identify network devices and vendors involved
- Gather bandwidth and performance requirements
- Understand security policies and access controls
- Ask about redundancy and high availability needs
- Clarify compliance and regulatory requirements
- Understand current network infrastructure

EXAMPLE NETWORK QUESTIONS:
- "How many sites need to be connected?"
- "What network devices do you currently have (Cisco, Palo Alto, etc.)?"
- "What bandwidth do you need between sites?"
- "Do you have specific security compliance requirements?"
- "Do you need redundant connections for failover?"
"""

        elif domain == "devops":
            base_prompt += """

DEVOPS DOMAIN QUESTION STRATEGY:
- Clarify target deployment platforms
- Understand current development workflow
- Identify automation and CI/CD requirements
- Gather testing and quality gate needs
- Understand monitoring and observability requirements
- Ask about security scanning and compliance
- Clarify infrastructure and scaling needs

EXAMPLE DEVOPS QUESTIONS:
- "Are you deploying to cloud (Azure/AWS) or on-premises?"
- "What's your current development workflow?"
- "Do you need automated testing as part of the pipeline?"
- "What monitoring and alerting do you need?"
- "Do you have any security scanning requirements?"
"""

        elif domain == "dev":
            base_prompt += """

DEVELOPMENT DOMAIN QUESTION STRATEGY:
- Clarify core functionality and user requirements
- Identify preferred technology stack
- Understand data storage and management needs
- Gather integration and API requirements
- Ask about user interface and experience needs
- Understand performance and scalability requirements
- Clarify security and authentication needs

EXAMPLE DEVELOPMENT QUESTIONS:
- "What are the main features users need?"
- "Do you have preferred programming languages or frameworks?"
- "What kind of data will you be storing?"
- "Do you need to integrate with any existing systems or APIs?"
- "What type of user interface do you need (web, mobile, desktop)?"
"""

        base_prompt += f"""

QUESTION GENERATION GUIDELINES:

AVOID THESE MISTAKES:
- Don't ask for information already provided
- Don't ask multiple concepts in one question
- Don't use excessive technical jargon without context
- Don't ask questions that can't reasonably be answered
- Don't repeat previously asked questions

GOOD QUESTION STRUCTURE:
- Start with context: "To help design your [solution]..."
- Ask specific question with examples/options when helpful
- Explain why the information is important
- Provide guidance for answering if needed

MULTI-TURN STRATEGY:
- Ask most critical questions first
- Build on answers to ask follow-up questions
- Group related questions together
- Save detailed technical questions for later turns
- Prioritize based on solution impact

{self.output_parser.get_format_instructions()}"""
        
        return base_prompt
    
    def _build_user_message(self, state: TechnicalConversationState) -> str:
        """Build user message for question generation"""
        
        message_parts = [
            f"ORIGINAL REQUEST: {state['initial_query']}",
            f"DOMAIN: {state['domain']}",
            f"CONVERSATION TURNS: {len(state.get('conversation_history', []))}"
        ]
        
        # Add sufficiency assessment context
        sufficiency_info = state.get('accumulated_info', {}).get('sufficiency_assessment', {})
        if sufficiency_info:
            message_parts.append("SUFFICIENCY ASSESSMENT:")
            message_parts.append(f"- Sufficiency Score: {sufficiency_info.get('sufficiency_score', 0):.1f}%")
            message_parts.append(f"- Level: {sufficiency_info.get('sufficiency_level', 'unknown')}")
            message_parts.append(f"- Can Proceed: {sufficiency_info.get('can_proceed', False)}")
            
            if sufficiency_info.get('critical_gaps'):
                message_parts.append(f"- Critical Gaps: {', '.join(sufficiency_info['critical_gaps'][:5])}")
            
            if sufficiency_info.get('missing_areas'):
                message_parts.append(f"- Missing Areas: {', '.join(sufficiency_info['missing_areas'][:5])}")
        
        # Add what we know so far
        if state.get('extracted_info'):
            known_info = []
            for category, items in state['extracted_info'].items():
                if category == 'extraction_metadata':
                    continue
                if isinstance(items, list) and items:
                    known_info.append(f"{category}: {len(items)} items")
            
            if known_info:
                message_parts.append("INFORMATION WE HAVE:")
                message_parts.extend(known_info[:8])
        
        # Add questions already asked to avoid repetition
        if state.get('questions_asked'):
            message_parts.append("QUESTIONS ALREADY ASKED:")
            recent_questions = state['questions_asked'][-5:]  # Last 5 questions
            for i, question in enumerate(recent_questions, 1):
                message_parts.append(f"{i}. {question[:80]}{'...' if len(question) > 80 else ''}")
        
        # Add missing fields that need clarification
        if state.get('missing_fields'):
            message_parts.append(f"PRIORITY MISSING FIELDS: {', '.join(state['missing_fields'][:8])}")
        
        # Add conversation context
        if state.get('conversation_history'):
            last_exchange = state['conversation_history'][-1]
            if isinstance(last_exchange, dict):
                if 'user' in last_exchange:
                    message_parts.append(f"LAST USER MESSAGE: {last_exchange['user'][:150]}")
        
        return "\n\n".join(message_parts)
    
    def _optimize_questions(
        self, 
        generated_questions: GeneratedQuestions, 
        state: TechnicalConversationState
    ) -> GeneratedQuestions:
        """Optimize generated questions for better user experience"""
        
        # Remove duplicate questions
        unique_questions = []
        seen_texts = set()
        
        for question in generated_questions.questions:
            question_key = question.text.lower().strip()
            if question_key not in seen_texts:
                unique_questions.append(question)
                seen_texts.add(question_key)
        
        # Sort by priority
        unique_questions.sort(key=lambda q: (q.priority, -len(q.suggested_options)))
        
        # Select questions for current turn (limit to max_questions_per_turn)
        current_turn_questions = unique_questions[:self.max_questions_per_turn]
        future_turn_questions = unique_questions[self.max_questions_per_turn:]
        
        # Update the generated questions object
        generated_questions.questions = unique_questions
        generated_questions.current_turn_questions = [q.id for q in current_turn_questions]
        generated_questions.future_turn_questions = [q.id for q in future_turn_questions]
        
        generated_questions.high_priority_questions = [
            q.id for q in unique_questions if q.priority <= 2
        ]
        
        generated_questions.optional_questions = [
            q.id for q in unique_questions if q.priority >= 4
        ]
        
        return generated_questions
    
    def _update_state_with_questions(
        self, 
        state: TechnicalConversationState, 
        generated_questions: GeneratedQuestions
    ) -> TechnicalConversationState:
        """Update state with generated questions"""
        
        updated_state = state.copy()
        
        # Get questions for current turn
        current_questions = [
            q for q in generated_questions.questions 
            if q.id in generated_questions.current_turn_questions
        ]
        
        # Format questions for response
        if current_questions:
            response_parts = [
                f"I need a bit more information to provide the best {state['domain']} guidance for your needs.",
                ""
            ]
            
            for i, question in enumerate(current_questions, 1):
                question_text = question.text
                
                # Add suggested options if available
                if question.suggested_options:
                    options_text = ", ".join(question.suggested_options)
                    if question.allows_custom_answer:
                        question_text += f" (Options: {options_text}, or specify something else)"
                    else:
                        question_text += f" (Options: {options_text})"
                
                response_parts.append(f"{i}. {question_text}")
            
            # Add context about why these questions matter
            if len(current_questions) > 1:
                response_parts.extend([
                    "",
                    "These details will help me provide more targeted and practical recommendations for your specific situation."
                ])
            
            updated_state['response'] = "\n".join(response_parts)
            updated_state['response_type'] = ResponseType.QUESTION
        
        # Add current questions to questions_asked history
        questions_asked = updated_state.get('questions_asked', []).copy()
        for question in current_questions:
            questions_asked.append(question.text)
        updated_state['questions_asked'] = questions_asked
        
        # Store question generation metadata
        accumulated_info = updated_state.get('accumulated_info', {})
        accumulated_info['question_generation'] = {
            'total_questions_generated': len(generated_questions.questions),
            'current_turn_questions': len(current_questions),
            'high_priority_count': len(generated_questions.high_priority_questions),
            'generation_confidence': generated_questions.generation_confidence,
            'question_strategy': generated_questions.question_strategy,
            'coverage_analysis': generated_questions.coverage_analysis
        }
        updated_state['accumulated_info'] = accumulated_info
        
        # Update token usage
        token_usage = updated_state.get('token_usage', {}).copy()
        token_usage['question_generator'] = token_usage.get('question_generator', 0) + 200
        updated_state['token_usage'] = token_usage
        
        return updated_state
    
    def _build_question_templates(self) -> Dict[str, Dict[str, Any]]:
        """Build domain-specific question templates"""
        
        return {
            "cloud": {
                "critical_questions": [
                    "Which cloud provider do you prefer or are you currently using?",
                    "What type of application or workload are you looking to deploy?",
                    "What are your performance and scalability requirements?",
                    "Do you have any specific security or compliance requirements?"
                ],
                "follow_up_patterns": [
                    "cost optimization", "existing infrastructure", "migration timeline",
                    "monitoring needs", "backup requirements"
                ]
            },
            "network": {
                "critical_questions": [
                    "What locations or sites need to be connected?",
                    "What network equipment do you currently have?",
                    "What are your bandwidth and performance requirements?",
                    "What security policies need to be implemented?"
                ],
                "follow_up_patterns": [
                    "redundancy requirements", "compliance standards", "current topology",
                    "user access patterns", "application requirements"
                ]
            },
            "devops": {
                "critical_questions": [
                    "What platforms are you deploying to?",
                    "What's your current development and deployment workflow?",
                    "What testing and quality assurance do you need?",
                    "What monitoring and observability requirements do you have?"
                ],
                "follow_up_patterns": [
                    "existing tooling", "team structure", "security requirements",
                    "infrastructure automation", "release frequency"
                ]
            },
            "dev": {
                "critical_questions": [
                    "What are the main features and functionality you need?",
                    "Do you have preferred programming languages or frameworks?",
                    "What kind of data will your application handle?",
                    "Do you need to integrate with existing systems?"
                ],
                "follow_up_patterns": [
                    "user interface requirements", "performance expectations", "security needs",
                    "scalability requirements", "deployment preferences"
                ]
            }
        }
    
    def _apply_cached_result(
        self, 
        state: TechnicalConversationState, 
        cached_questions: GeneratedQuestions
    ) -> TechnicalConversationState:
        """Apply cached questions result to state"""
        
        # Record cache hit
        cache_hits = state.get('cache_hits', []).copy()
        cache_hits.append(f"question_generator:{state['updated_at']}")
        
        updated_state = self._update_state_with_questions(state, cached_questions)
        updated_state['cache_hits'] = cache_hits
        
        return updated_state
    
    def _validate_output_state(self, state: TechnicalConversationState) -> None:
        """Validate output state for question generator"""
        super()._validate_output_state(state)
        
        # Ensure questions were added if generation was triggered
        if state.get('response_type') == ResponseType.QUESTION:
            if not state.get('response') or len(state.get('response', '')) < 10:
                raise NodeValidationError("Questions should be generated but response is empty")