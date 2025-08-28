"""
Implementation Generator Node Implementation

This module implements the ImplementationGeneratorNode that creates comprehensive
technical implementation plans with code examples, best practices, and step-by-step guidance.

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
from ..models.conversation_state import TechnicalConversationState, ResponseType, ConversationStage
from ..models.streaming import StreamTextChunk


class SolutionComplexity(str, Enum):
    """Complexity levels for solutions"""
    SIMPLE = "simple"           # Basic, straightforward implementation
    MODERATE = "moderate"       # Some complexity, multiple components
    COMPLEX = "complex"         # High complexity, many dependencies
    ENTERPRISE = "enterprise"   # Enterprise-scale, advanced patterns


class ImplementationStep(BaseModel):
    """Represents a single implementation step"""
    
    step_number: int = Field(description="Order of this step")
    title: str = Field(description="Brief title for the step")
    description: str = Field(description="Detailed description of what to do")
    
    # Code and examples
    code_examples: List[str] = Field(default=[], description="Code snippets for this step")
    configuration_examples: List[str] = Field(default=[], description="Configuration examples")
    commands: List[str] = Field(default=[], description="Commands to execute")
    
    # Guidance
    best_practices: List[str] = Field(default=[], description="Best practices for this step")
    common_pitfalls: List[str] = Field(default=[], description="Common mistakes to avoid")
    validation_steps: List[str] = Field(default=[], description="How to verify this step worked")
    
    # Dependencies and timing
    prerequisites: List[str] = Field(default=[], description="What must be done before this step")
    estimated_time: Optional[str] = Field(None, description="Estimated time to complete")
    complexity_level: SolutionComplexity = Field(description="Complexity of this step")
    
    # Additional resources
    documentation_links: List[str] = Field(default=[], description="Relevant documentation")
    troubleshooting_tips: List[str] = Field(default=[], description="Troubleshooting guidance")


class SolutionOption(BaseModel):
    """Represents an alternative solution approach"""
    
    option_name: str = Field(description="Name of this solution option")
    description: str = Field(description="Description of this approach")
    
    # Trade-offs analysis
    advantages: List[str] = Field(description="Benefits of this approach")
    disadvantages: List[str] = Field(description="Drawbacks of this approach")
    use_cases: List[str] = Field(description="When to use this approach")
    
    # Implementation details
    implementation_steps: List[ImplementationStep] = Field(description="Steps for this approach")
    technologies_required: List[str] = Field(description="Technologies needed")
    prerequisites: List[str] = Field(description="Prerequisites for this approach")
    
    # Effort and cost estimates
    estimated_effort: str = Field(description="Estimated implementation effort")
    complexity_level: SolutionComplexity = Field(description="Overall complexity")
    cost_implications: List[str] = Field(default=[], description="Cost considerations")
    
    # Quality attributes
    performance_impact: str = Field(description="Expected performance characteristics")
    scalability_impact: str = Field(description="Scalability considerations")
    security_considerations: List[str] = Field(default=[], description="Security aspects")
    maintenance_requirements: List[str] = Field(default=[], description="Ongoing maintenance needs")


class ImplementationPlan(BaseModel):
    """Structured output for implementation planning"""
    
    # Overview
    solution_title: str = Field(description="Title of the solution")
    solution_summary: str = Field(description="Executive summary of the solution")
    domain: str = Field(description="Technical domain")
    complexity_assessment: SolutionComplexity = Field(description="Overall complexity")
    
    # Solution options
    recommended_solution: SolutionOption = Field(description="Primary recommended approach")
    alternative_solutions: List[SolutionOption] = Field(default=[], description="Alternative approaches")
    
    # Architecture and design
    architecture_overview: str = Field(description="High-level architecture description")
    key_components: List[str] = Field(description="Main components/services")
    technology_stack: List[str] = Field(description="Recommended technology stack")
    integration_points: List[str] = Field(default=[], description="External integration points")
    
    # Quality and operations
    security_framework: List[str] = Field(default=[], description="Security measures and controls")
    monitoring_strategy: List[str] = Field(default=[], description="Monitoring and observability")
    backup_strategy: List[str] = Field(default=[], description="Backup and recovery approach")
    scaling_strategy: List[str] = Field(default=[], description="Scaling approach and considerations")
    
    # Project management
    implementation_phases: List[str] = Field(default=[], description="Recommended implementation phases")
    estimated_timeline: str = Field(description="Overall timeline estimate")
    resource_requirements: List[str] = Field(default=[], description="Required resources and skills")
    risk_factors: List[str] = Field(default=[], description="Key risks and mitigation strategies")
    
    # Success metrics
    success_criteria: List[str] = Field(default=[], description="How to measure success")
    testing_strategy: List[str] = Field(default=[], description="Testing approach and requirements")
    rollback_plan: List[str] = Field(default=[], description="Rollback and disaster recovery")
    
    # Follow-up
    next_steps: List[str] = Field(description="Immediate next steps to start implementation")
    additional_resources: List[str] = Field(default=[], description="Additional learning resources")
    expert_consultation_areas: List[str] = Field(default=[], description="Areas that may need expert help")
    
    # Metadata
    plan_confidence: float = Field(description="Confidence in this plan", ge=0.0, le=1.0)
    assumptions_made: List[str] = Field(default=[], description="Key assumptions in this plan")
    information_gaps: List[str] = Field(default=[], description="Information that would improve this plan")


class ImplementationGeneratorNode(BaseNode):
    """
    Node that generates comprehensive implementation plans.
    
    Supports:
    - Step-by-step implementation plans with code examples
    - Multiple solution options with trade-offs
    - Technology stack recommendations
    - Security and best practice considerations
    - Cost and performance estimates
    - Solution validation against requirements
    """
    
    def __init__(self, **kwargs):
        super().__init__(node_name="ImplementationGenerator", **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=ImplementationPlan)
        
        # Load domain-specific templates and patterns
        self.implementation_templates = self._build_implementation_templates()
        self.best_practices_db = self._load_best_practices_database()
    
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process state through implementation plan generation.
        
        Args:
            state: Current conversation state with sufficient information
            
        Returns:
            Updated state with comprehensive implementation plan
        """
        # Verify this should run (sufficient information should be available)
        sufficiency_info = state.get('accumulated_info', {}).get('sufficiency_assessment', {})
        if not sufficiency_info.get('can_proceed', False):
            # Return state unchanged if not ready for implementation
            return state
        
        # Check cache first
        cache_key = self._get_cache_key(
            state,
            [
                str(hash(str(state.get('requirements', [])))),
                str(hash(str(state.get('extracted_info', {})))),
                str(sufficiency_info.get('sufficiency_score', 0))
            ]
        )
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return self._apply_cached_result(state, cached_result)
        
        # Generate implementation plan
        implementation_plan = await self._generate_implementation_plan(state)
        
        # Update state with implementation plan
        updated_state = self._update_state_with_plan(state, implementation_plan)
        
        # Cache the result
        self._set_cache(cache_key, implementation_plan)
        
        return updated_state
    
    async def stream_process(self, state: TechnicalConversationState) -> AsyncGenerator[str, None]:
        """
        Stream implementation generation in real-time.
        
        Args:
            state: Current conversation state
            
        Yields:
            str: Text chunks of the implementation response
        """
        # Verify this should run (sufficient information should be available)
        sufficiency_info = state.get('accumulated_info', {}).get('sufficiency_assessment', {})
        if not sufficiency_info.get('can_proceed', False):
            yield "I need more information to provide a complete implementation. Please provide additional details."
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
            logger.error(f"Error during streaming implementation generation: {e}")
            yield f"Error generating implementation: {str(e)}"
    
    async def _generate_implementation_plan(self, state: TechnicalConversationState) -> ImplementationPlan:
        """Generate comprehensive implementation plan"""
        
        system_prompt = self._build_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        implementation_plan = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_parser=self.output_parser
        )
        
        # Enhance plan with domain-specific best practices
        implementation_plan = self._enhance_plan_with_best_practices(implementation_plan, state)
        
        return implementation_plan
    
    def _build_system_prompt(self, domain: str) -> str:
        """Build system prompt for implementation planning"""
        
        domain_context = self.get_domain_context(domain)
        templates = self.implementation_templates.get(domain, {})
        best_practices = self.best_practices_db.get(domain, [])
        
        base_prompt = f"""You are a senior solution architect and implementation specialist for {domain} solutions.

Your task is to create a comprehensive, actionable implementation plan based on the gathered requirements and technical details.

DOMAIN: {domain.upper()}
EXPERTISE AREAS: {', '.join(domain_context.get('focus_areas', []))}
COMMON TECHNOLOGIES: {', '.join(domain_context.get('technologies', []))}

IMPLEMENTATION PLANNING FRAMEWORK:

1. SOLUTION DESIGN PRINCIPLES:
   - Start with the simplest solution that meets requirements
   - Design for maintainability and future growth
   - Follow industry best practices and security standards  
   - Consider operational aspects from day one
   - Plan for monitoring, backup, and disaster recovery
   - Include proper testing and validation approaches

2. IMPLEMENTATION APPROACH:
   - Break complex solutions into manageable phases
   - Provide specific, actionable steps with examples
   - Include code snippets and configuration examples
   - Explain the reasoning behind technical decisions
   - Address potential challenges and solutions
   - Provide validation and testing guidance

3. MULTIPLE SOLUTION OPTIONS:
   - Present 2-3 viable approaches when applicable
   - Clearly explain trade-offs between options
   - Recommend the best option with rationale
   - Consider different complexity levels and use cases
   - Address cost implications and resource requirements"""

        # Add domain-specific guidance
        if domain == "cloud":
            base_prompt += """

CLOUD IMPLEMENTATION FOCUS:
- Start with well-architected framework principles
- Leverage managed services to reduce operational overhead
- Implement proper IAM and security controls from the start
- Design for scalability and cost optimization
- Include monitoring, logging, and alerting setup
- Plan for backup, disaster recovery, and business continuity
- Consider compliance requirements and data residency
- Use Infrastructure as Code for consistency and reproducibility

CLOUD SOLUTION PATTERNS:
- Microservices architecture with containerization
- Serverless computing for event-driven workloads
- Multi-tier web applications with load balancing
- Data lakes and analytics pipelines
- Hybrid cloud connectivity and migration strategies
- DevOps pipelines with automated deployment"""

        elif domain == "network":
            base_prompt += """

NETWORK IMPLEMENTATION FOCUS:
- Design for security, performance, and reliability
- Implement defense-in-depth security strategies
- Plan for redundancy and eliminate single points of failure
- Include proper network segmentation and access controls
- Design for monitoring and troubleshooting
- Consider bandwidth optimization and QoS requirements
- Plan for growth and future scalability needs
- Include comprehensive documentation and change management

NETWORK SOLUTION PATTERNS:
- Site-to-site VPN with redundant connections
- Zero-trust network architecture
- Network segmentation with microsegmentation
- Load balancing and traffic optimization
- Network automation and orchestration
- Security policy enforcement and monitoring"""

        elif domain == "devops":
            base_prompt += """

DEVOPS IMPLEMENTATION FOCUS:
- Implement CI/CD pipelines with automated testing
- Use Infrastructure as Code for all environments
- Design for observability with comprehensive monitoring
- Implement security scanning and compliance checks
- Plan for automated deployment and rollback capabilities
- Include proper secret management and access controls
- Design for scalability and high availability
- Implement proper backup and disaster recovery

DEVOPS SOLUTION PATTERNS:
- GitOps workflows with automated deployment
- Container orchestration with Kubernetes
- Microservices deployment with service mesh
- Infrastructure automation with Terraform/Ansible
- Monitoring and observability with modern tools
- Security-first CI/CD with scanning and compliance"""

        elif domain == "dev":
            base_prompt += """

DEVELOPMENT IMPLEMENTATION FOCUS:
- Follow secure coding practices and OWASP guidelines
- Implement comprehensive testing strategies
- Design for scalability and performance
- Include proper error handling and logging
- Plan for API design and documentation
- Implement authentication and authorization properly
- Consider data modeling and database optimization
- Include deployment and operational considerations

DEVELOPMENT SOLUTION PATTERNS:
- RESTful API development with proper versioning
- Microservices architecture with event-driven design
- Modern web applications with responsive design
- Database design with performance optimization
- Authentication and authorization patterns
- Caching strategies and performance optimization"""

        base_prompt += f"""

4. IMPLEMENTATION STEP STRUCTURE:
   Each implementation step should include:
   - Clear step title and description
   - Code examples with explanations
   - Configuration examples and templates
   - Commands to execute with expected outputs
   - Best practices and common pitfalls to avoid
   - Validation steps to confirm success
   - Troubleshooting guidance for issues
   - Time estimates and complexity assessment

5. QUALITY ATTRIBUTES:
   Address these non-functional requirements:
   - Security: Authentication, authorization, encryption, compliance
   - Performance: Response times, throughput, scalability
   - Reliability: Uptime, fault tolerance, disaster recovery
   - Maintainability: Code quality, documentation, monitoring
   - Usability: User experience, accessibility, documentation

6. SOLUTION VALIDATION:
   - Ensure all requirements are addressed
   - Validate against technical constraints
   - Check compatibility with existing systems
   - Verify security and compliance requirements
   - Confirm resource and budget constraints are met

DOMAIN BEST PRACTICES:"""

        for practice in best_practices[:8]:
            base_prompt += f"\n- {practice}"

        base_prompt += f"""

{self.output_parser.get_format_instructions()}"""
        
        return base_prompt
    
    def _build_user_message(self, state: TechnicalConversationState) -> str:
        """Build user message for implementation planning"""
        
        message_parts = [
            f"SOLUTION REQUEST: {state['initial_query']}",
            f"DOMAIN: {state['domain']}",
            f"CURRENT CONTEXT: {state.get('current_input', '')}",
        ]
        
        # Add requirements summary
        if state.get('requirements'):
            req_summary = []
            must_haves = []
            nice_to_haves = []
            
            for req in state['requirements'][:10]:  # Limit to 10 requirements
                if isinstance(req, dict):
                    req_desc = f"{req.get('name', 'Unknown')}: {req.get('description', '')[:100]}"
                    if req.get('criticality') == 'must_have':
                        must_haves.append(req_desc)
                    else:
                        nice_to_haves.append(req_desc)
            
            if must_haves:
                req_summary.append("MUST-HAVE REQUIREMENTS:")
                req_summary.extend([f"- {req}" for req in must_haves])
            
            if nice_to_haves:
                req_summary.append("ADDITIONAL REQUIREMENTS:")
                req_summary.extend([f"- {req}" for req in nice_to_haves[:5]])
            
            if req_summary:
                message_parts.append("\n".join(req_summary))
        
        # Add extracted technical information
        if state.get('extracted_info'):
            tech_info = []
            
            for category, items in state['extracted_info'].items():
                if category == 'extraction_metadata':
                    continue
                
                if isinstance(items, list) and items:
                    high_confidence_items = [
                        item for item in items 
                        if isinstance(item, dict) and item.get('confidence', 0) > 0.7
                    ]
                    
                    if high_confidence_items:
                        category_items = []
                        for item in high_confidence_items[:5]:  # Limit to 5 per category
                            name = item.get('name', str(item))
                            value = item.get('value', '')
                            if value and value != name:
                                category_items.append(f"{name}: {value}")
                            else:
                                category_items.append(name)
                        
                        if category_items:
                            tech_info.append(f"{category.upper().replace('_', ' ')}: {', '.join(category_items)}")
            
            if tech_info:
                message_parts.append("TECHNICAL SPECIFICATIONS:")
                message_parts.extend(tech_info[:8])
        
        # Add sufficiency assessment for context
        sufficiency_info = state.get('accumulated_info', {}).get('sufficiency_assessment', {})
        if sufficiency_info:
            message_parts.append("SUFFICIENCY ANALYSIS:")
            message_parts.append(f"- Information Completeness: {sufficiency_info.get('sufficiency_score', 0):.1f}%")
            message_parts.append(f"- Assessment: {sufficiency_info.get('sufficiency_level', 'unknown')}")
            
            if sufficiency_info.get('complete_areas'):
                message_parts.append(f"- Complete Areas: {', '.join(sufficiency_info['complete_areas'][:5])}")
            
            if sufficiency_info.get('assumptions_needed'):
                message_parts.append(f"- Assumptions Needed: {', '.join(sufficiency_info.get('assumptions_needed', [])[:3])}")
        
        # Add conversation context for personalization
        message_parts.append(f"CONVERSATION HISTORY: {len(state.get('conversation_history', []))} exchanges")
        
        # Add any specific constraints or preferences
        constraints = []
        if state.get('extracted_info', {}).get('constraints'):
            for constraint in state['extracted_info']['constraints'][:3]:
                if isinstance(constraint, dict):
                    constraints.append(constraint.get('name', str(constraint)))
        
        if constraints:
            message_parts.append(f"KEY CONSTRAINTS: {', '.join(constraints)}")
        
        return "\n\n".join(message_parts)
    
    def _enhance_plan_with_best_practices(
        self, 
        plan: ImplementationPlan, 
        state: TechnicalConversationState
    ) -> ImplementationPlan:
        """Enhance implementation plan with domain-specific best practices"""
        
        domain = state['domain']
        domain_practices = self.best_practices_db.get(domain, [])
        
        # Add security framework if not comprehensive
        if len(plan.security_framework) < 3:
            security_practices = [
                practice for practice in domain_practices 
                if 'security' in practice.lower() or 'auth' in practice.lower()
            ]
            plan.security_framework.extend(security_practices[:3])
        
        # Add monitoring strategy if missing
        if len(plan.monitoring_strategy) < 2:
            monitoring_practices = [
                practice for practice in domain_practices
                if 'monitor' in practice.lower() or 'log' in practice.lower()
            ]
            plan.monitoring_strategy.extend(monitoring_practices[:2])
        
        # Add scaling strategy based on domain
        if len(plan.scaling_strategy) < 2:
            scaling_patterns = self._get_domain_scaling_patterns(domain)
            plan.scaling_strategy.extend(scaling_patterns[:2])
        
        # Enhance testing strategy
        if len(plan.testing_strategy) < 3:
            testing_patterns = self._get_domain_testing_patterns(domain)
            plan.testing_strategy.extend(testing_patterns[:3])
        
        return plan
    
    def _get_domain_scaling_patterns(self, domain: str) -> List[str]:
        """Get domain-specific scaling patterns"""
        
        patterns = {
            "cloud": [
                "Implement auto-scaling groups with appropriate scaling policies",
                "Use load balancers to distribute traffic across multiple instances",
                "Implement caching strategies to reduce database load",
                "Consider using CDN for static content distribution"
            ],
            "network": [
                "Design with redundant paths and failover capabilities",
                "Implement load balancing across multiple network paths",
                "Plan for bandwidth scaling and capacity growth",
                "Use network segmentation to isolate traffic flows"
            ],
            "devops": [
                "Implement horizontal pod autoscaling in Kubernetes",
                "Use container orchestration for dynamic scaling",
                "Implement pipeline parallelization for faster builds",
                "Design infrastructure to scale with demand"
            ],
            "dev": [
                "Implement database read replicas for read scaling",
                "Use microservices architecture for independent scaling",
                "Implement caching at multiple layers",
                "Design APIs for horizontal scaling"
            ]
        }
        
        return patterns.get(domain, [])
    
    def _get_domain_testing_patterns(self, domain: str) -> List[str]:
        """Get domain-specific testing patterns"""
        
        patterns = {
            "cloud": [
                "Implement infrastructure testing with tools like Terratest",
                "Use chaos engineering to test resilience",
                "Implement comprehensive monitoring and alerting tests",
                "Test disaster recovery and backup procedures"
            ],
            "network": [
                "Implement network connectivity and performance testing",
                "Test failover scenarios and redundancy paths",
                "Validate security policies and access controls",
                "Test network capacity and bandwidth utilization"
            ],
            "devops": [
                "Implement automated testing in CI/CD pipelines",
                "Use infrastructure validation and compliance testing",
                "Test deployment and rollback procedures",
                "Implement security scanning and vulnerability testing"
            ],
            "dev": [
                "Implement unit testing with high code coverage",
                "Use integration testing for API and service interactions",
                "Implement end-to-end testing for critical user flows",
                "Use performance testing to validate scalability"
            ]
        }
        
        return patterns.get(domain, [])
    
    def _update_state_with_plan(
        self, 
        state: TechnicalConversationState, 
        plan: ImplementationPlan
    ) -> TechnicalConversationState:
        """Update state with implementation plan"""
        
        updated_state = state.copy()
        
        # Format the implementation plan as a response
        response = self._format_implementation_response(plan, state['domain'])
        updated_state['response'] = response
        updated_state['response_type'] = ResponseType.IMPLEMENTATION
        
        # Store the full implementation plan
        accumulated_info = updated_state.get('accumulated_info', {})
        accumulated_info['implementation_plan'] = {
            'solution_title': plan.solution_title,
            'complexity_assessment': plan.complexity_assessment.value,
            'technology_stack': plan.technology_stack,
            'implementation_phases': plan.implementation_phases,
            'estimated_timeline': plan.estimated_timeline,
            'plan_confidence': plan.plan_confidence,
            'assumptions_made': plan.assumptions_made,
            'next_steps': plan.next_steps,
            'alternative_solutions_count': len(plan.alternative_solutions),
            'total_implementation_steps': len(plan.recommended_solution.implementation_steps)
        }
        updated_state['accumulated_info'] = accumulated_info
        
        # Update conversation stage to completed
        updated_state['stage'] = ConversationStage.SUFFICIENT
        
        # Update token usage
        token_usage = updated_state.get('token_usage', {}).copy()
        token_usage['implementation_generator'] = token_usage.get('implementation_generator', 0) + 600
        updated_state['token_usage'] = token_usage
        
        return updated_state
    
    async def stream_process(self, state: TechnicalConversationState) -> AsyncGenerator[str, None]:
        """
        Stream implementation plan generation with real-time response delivery.
        
        Args:
            state: Current conversation state
            
        Yields:
            str: Text chunks for progressive response generation
        """
        # Verify this should run (sufficient information should be available)
        sufficiency_info = state.get('accumulated_info', {}).get('sufficiency_assessment', {})
        if not sufficiency_info.get('can_proceed', False):
            yield "I need more information to create a comprehensive implementation plan. "
            yield "Could you provide additional details about your requirements?"
            return
        
        # Build the prompt for streaming generation
        system_prompt = self._build_streaming_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        try:
            # Stream the implementation plan generation
            async for chunk in self.stream_llm(
                system_prompt=system_prompt,
                user_message=user_message
            ):
                if chunk and chunk.strip():
                    yield chunk
                    # Small delay for better streaming experience
                    await asyncio.sleep(0.05)
                    
        except Exception as e:
            yield f"\nI encountered an error while generating the implementation plan: {str(e)}\n"
            yield "Please try again or provide more specific requirements."
    
    def _build_streaming_system_prompt(self, domain: str) -> str:
        """Build system prompt optimized for streaming implementation responses"""
        
        domain_context = self.get_domain_context(domain)
        
        base_prompt = f"""You are a senior solution architect and implementation specialist for {domain} solutions.

Your task is to create a comprehensive, actionable implementation plan based on the gathered requirements and technical details.

DOMAIN: {domain.upper()}
EXPERTISE AREAS: {', '.join(domain_context.get('focus_areas', []))}
COMMON TECHNOLOGIES: {', '.join(domain_context.get('technologies', []))}

STREAMING RESPONSE FORMAT:
Generate your response as a well-structured, markdown-formatted implementation plan that includes:

1. Start with a clear solution title and summary
2. Provide architecture overview and technology stack
3. Break down into numbered implementation steps with:
   - Clear step titles and descriptions
   - Code examples and configurations when relevant
   - Validation steps and best practices
4. Include security considerations and monitoring strategy
5. Provide clear next steps

Write in a conversational yet professional tone, as if explaining to a technical colleague.
Use markdown formatting for structure and readability.
Be comprehensive but concise - focus on actionable guidance."""

        # Add domain-specific guidance for streaming
        if domain == "cloud":
            base_prompt += """

CLOUD IMPLEMENTATION FOCUS:
- Lead with well-architected framework principles
- Emphasize managed services and operational efficiency
- Include specific Azure/AWS/GCP service recommendations
- Address security, scalability, and cost optimization
- Provide Infrastructure as Code examples"""

        elif domain == "network":
            base_prompt += """

NETWORK IMPLEMENTATION FOCUS:
- Start with network topology and security design
- Provide specific device configurations and protocols
- Include redundancy and failover strategies
- Address performance and capacity planning
- Include monitoring and troubleshooting guidance"""

        elif domain == "devops":
            base_prompt += """

DEVOPS IMPLEMENTATION FOCUS:
- Begin with CI/CD pipeline architecture
- Include Infrastructure as Code and automation
- Address security scanning and compliance
- Provide container orchestration strategies
- Include monitoring and observability setup"""

        elif domain == "dev":
            base_prompt += """

DEVELOPMENT IMPLEMENTATION FOCUS:
- Start with application architecture and design patterns
- Include code structure and development practices
- Address API design and data management
- Provide testing and security implementation
- Include deployment and operational considerations"""

        return base_prompt

    def _format_implementation_response(self, plan: ImplementationPlan, domain: str) -> str:
        """Format implementation plan as user-friendly response"""
        
        response_parts = [
            f"# {plan.solution_title}",
            "",
            f"**Summary:** {plan.solution_summary}",
            "",
            f"**Complexity:** {plan.complexity_assessment.value.title()} | **Timeline:** {plan.estimated_timeline}",
            ""
        ]
        
        # Add architecture overview
        if plan.architecture_overview:
            response_parts.extend([
                "## 🏗️ Architecture Overview",
                plan.architecture_overview,
                ""
            ])
        
        # Add technology stack
        if plan.technology_stack:
            response_parts.extend([
                "## 🛠️ Technology Stack",
                "- " + "\n- ".join(plan.technology_stack),
                ""
            ])
        
        # Add recommended implementation steps
        if plan.recommended_solution.implementation_steps:
            response_parts.extend([
                "## 📋 Implementation Steps",
                ""
            ])
            
            for step in plan.recommended_solution.implementation_steps[:8]:  # Limit to 8 steps for readability
                response_parts.append(f"### Step {step.step_number}: {step.title}")
                response_parts.append(step.description)
                
                # Add code examples if available
                if step.code_examples:
                    response_parts.append("\n**Example:**")
                    for code in step.code_examples[:2]:  # Limit to 2 examples per step
                        response_parts.append(f"```\n{code}\n```")
                
                # Add commands if available
                if step.commands:
                    response_parts.append("\n**Commands:**")
                    for cmd in step.commands[:3]:  # Limit to 3 commands per step
                        response_parts.append(f"```bash\n{cmd}\n```")
                
                # Add validation steps
                if step.validation_steps:
                    response_parts.append("\n**Validation:**")
                    response_parts.append("- " + "\n- ".join(step.validation_steps[:2]))
                
                response_parts.append("")
        
        # Add security considerations
        if plan.security_framework:
            response_parts.extend([
                "## 🔒 Security Framework",
                "- " + "\n- ".join(plan.security_framework[:5]),
                ""
            ])
        
        # Add alternative solutions if available
        if plan.alternative_solutions:
            response_parts.extend([
                "## 🔄 Alternative Approaches",
                ""
            ])
            
            for alt_solution in plan.alternative_solutions[:2]:  # Limit to 2 alternatives
                response_parts.extend([
                    f"### {alt_solution.option_name}",
                    alt_solution.description,
                    "",
                    "**Advantages:**",
                    "- " + "\n- ".join(alt_solution.advantages[:3]),
                    "",
                    "**Use When:**",
                    "- " + "\n- ".join(alt_solution.use_cases[:2]),
                    ""
                ])
        
        # Add next steps
        if plan.next_steps:
            response_parts.extend([
                "## 🚀 Next Steps",
                ""
            ])
            
            for i, step in enumerate(plan.next_steps[:5], 1):
                response_parts.append(f"{i}. {step}")
            
            response_parts.append("")
        
        # Add monitoring and operations
        if plan.monitoring_strategy:
            response_parts.extend([
                "## 📊 Monitoring & Operations",
                "- " + "\n- ".join(plan.monitoring_strategy[:4]),
                ""
            ])
        
        # Add assumptions and caveats
        if plan.assumptions_made:
            response_parts.extend([
                "## ⚠️ Key Assumptions",
                "- " + "\n- ".join(plan.assumptions_made[:3]),
                ""
            ])
        
        # Add expert consultation areas if needed
        if plan.expert_consultation_areas:
            response_parts.extend([
                "## 👨‍💻 Consider Expert Help For:",
                "- " + "\n- ".join(plan.expert_consultation_areas[:3]),
                ""
            ])
        
        # Add follow-up note
        response_parts.extend([
            "---",
            f"*This {domain} implementation plan is based on the requirements and constraints you've provided. " +
            f"Feel free to ask for clarification on any step or explore the alternative approaches mentioned above.*"
        ])
        
        return "\n".join(response_parts)
    
    def _build_implementation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Build domain-specific implementation templates"""
        
        return {
            "cloud": {
                "common_architectures": [
                    "Three-tier web application",
                    "Microservices with containers", 
                    "Serverless event-driven",
                    "Data pipeline with analytics",
                    "Hybrid cloud connectivity"
                ],
                "typical_phases": [
                    "Infrastructure setup and security",
                    "Core services deployment",
                    "Integration and testing",
                    "Monitoring and optimization",
                    "Go-live and operations"
                ]
            },
            "network": {
                "common_architectures": [
                    "Site-to-site VPN mesh",
                    "Hub-and-spoke topology",
                    "Zero-trust network",
                    "Network segmentation",
                    "Load balanced services"
                ],
                "typical_phases": [
                    "Network design and planning",
                    "Infrastructure deployment",
                    "Security policy implementation",
                    "Testing and validation",
                    "Go-live and monitoring"
                ]
            },
            "devops": {
                "common_architectures": [
                    "GitOps CI/CD pipeline",
                    "Container orchestration",
                    "Infrastructure as Code",
                    "Monitoring and observability",
                    "Security-first DevOps"
                ],
                "typical_phases": [
                    "Pipeline setup and automation",
                    "Infrastructure automation",
                    "Security integration",
                    "Monitoring implementation",
                    "Team onboarding and optimization"
                ]
            },
            "dev": {
                "common_architectures": [
                    "RESTful API backend",
                    "Microservices architecture",
                    "Modern web application",
                    "Mobile application",
                    "Data processing system"
                ],
                "typical_phases": [
                    "Architecture and design",
                    "Core development",
                    "Integration and testing",
                    "Deployment and operations",
                    "Maintenance and enhancement"
                ]
            }
        }
    
    def _load_best_practices_database(self) -> Dict[str, List[str]]:
        """Load comprehensive domain-specific best practices"""
        
        return {
            "cloud": [
                "Use Infrastructure as Code for all cloud resources",
                "Implement comprehensive monitoring and alerting across all services",
                "Follow the principle of least privilege for all access controls",
                "Design for failure with automated recovery and redundancy",
                "Implement proper backup and disaster recovery strategies",
                "Use managed services to reduce operational overhead",
                "Implement cost monitoring and optimization from day one",
                "Follow security best practices including encryption in transit and at rest",
                "Design for scalability with auto-scaling and load balancing",
                "Implement proper logging and audit trails for compliance"
            ],
            "network": [
                "Implement defense-in-depth security strategies",
                "Design with redundancy to eliminate single points of failure", 
                "Use network segmentation to contain security breaches",
                "Implement proper access controls and authentication",
                "Monitor network performance and security continuously",
                "Plan for capacity growth and future scalability",
                "Document all network configurations and changes",
                "Implement proper change management processes",
                "Use standard protocols and avoid proprietary solutions",
                "Design for maintainability with clear documentation"
            ],
            "devops": [
                "Implement CI/CD pipelines for all code deployments",
                "Use Infrastructure as Code for reproducible environments",
                "Implement comprehensive testing at all levels",
                "Use containerization for consistency across environments",
                "Implement monitoring, logging, and alerting for all services",
                "Practice security scanning and vulnerability management",
                "Implement proper secret management and access controls",
                "Design for rollback and disaster recovery",
                "Use GitOps practices for configuration management",
                "Implement automated compliance and policy enforcement"
            ],
            "dev": [
                "Follow secure coding practices and OWASP guidelines",
                "Implement comprehensive testing strategies including unit, integration, and e2e tests",
                "Use version control and code review processes",
                "Design for scalability and performance from the start",
                "Implement proper error handling and logging throughout",
                "Follow API design best practices with proper versioning",
                "Implement authentication and authorization correctly",
                "Design for maintainability with clean code practices",
                "Use dependency management and security scanning",
                "Implement proper database design and optimization"
            ]
        }
    
    def _apply_cached_result(
        self, 
        state: TechnicalConversationState, 
        cached_plan: ImplementationPlan
    ) -> TechnicalConversationState:
        """Apply cached implementation plan to state"""
        
        # Record cache hit
        cache_hits = state.get('cache_hits', []).copy()
        cache_hits.append(f"implementation_generator:{state['updated_at']}")
        
        updated_state = self._update_state_with_plan(state, cached_plan)
        updated_state['cache_hits'] = cache_hits
        
        return updated_state
    
    def _validate_output_state(self, state: TechnicalConversationState) -> None:
        """Validate output state for implementation generator"""
        super()._validate_output_state(state)
        
        # Ensure implementation plan was created
        if state.get('response_type') == ResponseType.IMPLEMENTATION:
            if not state.get('response') or len(state.get('response', '')) < 100:
                raise NodeValidationError("Implementation plan should be generated but response is insufficient")
            
            if 'accumulated_info' not in state or 'implementation_plan' not in state['accumulated_info']:
                raise NodeValidationError("Implementation plan metadata missing from state")