"""
Requirement Analyzer Node Implementation

This module implements the RequirementAnalyzerNode that analyzes extracted
information to identify requirements, dependencies, and gaps.

Part of Epic 2: Agent Nodes Implementation  
Story 2.3: Create RequirementAnalyzerNode
"""

import json
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from .base import BaseNode
from ..models.conversation_state import TechnicalConversationState, ConversationStage


class RequirementType(str, Enum):
    """Types of requirements"""
    FUNCTIONAL = "functional"      # What the system should do
    TECHNICAL = "technical"        # How the system should work
    CONSTRAINT = "constraint"      # Limitations and restrictions
    QUALITY = "quality"           # Performance, security, reliability
    COMPLIANCE = "compliance"      # Regulatory requirements


class CriticalityLevel(str, Enum):
    """Criticality levels for requirements"""
    MUST_HAVE = "must_have"       # Critical, system won't work without it
    NICE_TO_HAVE = "nice_to_have" # Important but not critical
    OPTIONAL = "optional"         # Would be good but not necessary


class Requirement(BaseModel):
    """Represents a single requirement"""
    
    id: str = Field(description="Unique requirement identifier")
    name: str = Field(description="Short name for the requirement")
    description: str = Field(description="Detailed requirement description")
    type: RequirementType = Field(description="Type of requirement")
    criticality: CriticalityLevel = Field(description="How critical this requirement is")
    confidence: float = Field(description="Confidence in this requirement", ge=0.0, le=1.0)
    
    # Source and context
    source: str = Field(description="Where this requirement came from")
    context: Optional[str] = Field(None, description="Additional context")
    
    # Dependencies
    depends_on: List[str] = Field(default=[], description="Requirements this depends on")
    blocks: List[str] = Field(default=[], description="Requirements this blocks")
    
    # Validation
    is_testable: bool = Field(description="Whether this requirement can be tested")
    acceptance_criteria: List[str] = Field(default=[], description="How to verify this requirement")
    
    # Domain-specific metadata
    domain_tags: List[str] = Field(default=[], description="Domain-specific tags")
    estimated_effort: Optional[str] = Field(None, description="Estimated implementation effort")


class RequirementConflict(BaseModel):
    """Represents a conflict between requirements"""
    
    requirement_ids: List[str] = Field(description="IDs of conflicting requirements")
    conflict_type: str = Field(description="Type of conflict")
    description: str = Field(description="Description of the conflict")
    severity: str = Field(description="Severity: low, medium, high")
    resolution_suggestions: List[str] = Field(default=[], description="Suggested resolutions")


class RequirementAnalysis(BaseModel):
    """Structured output for requirement analysis"""
    
    requirements: List[Requirement] = Field(description="Analyzed requirements")
    conflicts: List[RequirementConflict] = Field(default=[], description="Identified conflicts")
    
    # Dependency information
    dependency_graph: Dict[str, List[str]] = Field(
        default={}, 
        description="Requirement dependency relationships"
    )
    critical_path: List[str] = Field(
        default=[], 
        description="Critical path of must-have requirements"
    )
    
    # Gap analysis
    missing_requirements: List[str] = Field(
        default=[], 
        description="Categories of requirements that appear to be missing"
    )
    implicit_requirements: List[str] = Field(
        default=[], 
        description="Requirements implied but not explicitly stated"
    )
    
    # Analysis metadata
    analysis_confidence: float = Field(
        description="Overall confidence in the analysis", 
        ge=0.0, 
        le=1.0
    )
    completeness_score: float = Field(
        description="How complete the requirement set appears to be",
        ge=0.0,
        le=1.0
    )
    recommendation: str = Field(description="Next steps recommendation")


class RequirementAnalyzerNode(BaseNode):
    """
    Node that analyzes extracted information to identify and structure requirements.
    
    Supports:
    - Requirement categorization (functional, technical, constraints)
    - Dependency graph generation for requirements
    - Criticality scoring (must-have, nice-to-have, optional)
    - Gap analysis between stated and implied requirements
    - Requirement validation against best practices
    - Conflict detection between requirements
    """
    
    def __init__(self, **kwargs):
        super().__init__(node_name="RequirementAnalyzer", **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=RequirementAnalysis)
        
        # Load domain-specific requirement templates
        self.domain_templates = self._build_domain_templates()
        self.best_practices = self._load_best_practices()
    
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process state through requirement analysis.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with requirement analysis results
        """
        # Check cache first
        cache_key = self._get_cache_key(
            state, 
            [str(hash(str(state.get('extracted_info', {}))))]
        )
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return self._apply_cached_result(state, cached_result)
        
        # Analyze requirements
        requirement_analysis = await self._analyze_requirements(state)
        
        # Update state with analysis results
        updated_state = self._update_state_with_analysis(state, requirement_analysis)
        
        # Cache the result
        self._set_cache(cache_key, requirement_analysis)
        
        return updated_state
    
    async def _analyze_requirements(self, state: TechnicalConversationState) -> RequirementAnalysis:
        """Analyze extracted information to identify requirements"""
        
        system_prompt = self._build_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        requirement_analysis = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_parser=self.output_parser
        )
        
        # Post-process to add derived insights
        requirement_analysis = self._enhance_analysis(requirement_analysis, state)
        
        return requirement_analysis
    
    def _build_system_prompt(self, domain: str) -> str:
        """Build system prompt for requirement analysis"""
        
        domain_context = self.get_domain_context(domain)
        domain_template = self.domain_templates.get(domain, {})
        best_practices = self.best_practices.get(domain, [])
        
        base_prompt = f"""You are an expert solution architect specializing in {domain} requirements analysis.

Your task is to analyze extracted technical information and identify structured requirements with dependencies and criticality.

DOMAIN: {domain.upper()}
TECHNOLOGIES: {', '.join(domain_context.get('technologies', []))}
COMMON PATTERNS: {', '.join(domain_context.get('common_patterns', []))}

REQUIREMENT ANALYSIS FRAMEWORK:

1. REQUIREMENT TYPES:
   - FUNCTIONAL: What the system must do (features, capabilities)
   - TECHNICAL: How the system should work (architecture, performance)
   - CONSTRAINT: Limitations (budget, time, existing systems)
   - QUALITY: Non-functional attributes (security, reliability, usability)
   - COMPLIANCE: Regulatory/policy requirements (HIPAA, SOX, etc.)

2. CRITICALITY LEVELS:
   - MUST_HAVE: System cannot function without this
   - NICE_TO_HAVE: Important for success but not blocking
   - OPTIONAL: Would be beneficial but not necessary

3. DEPENDENCY ANALYSIS:
   - Identify which requirements depend on others
   - Map blocking relationships
   - Consider implementation order
   - Note circular dependencies (conflicts)

4. REQUIREMENT VALIDATION:
   - Ensure requirements are testable and measurable
   - Identify acceptance criteria
   - Check for completeness and consistency
   - Flag potential conflicts"""

        # Add domain-specific analysis rules
        if domain == "cloud":
            base_prompt += """

CLOUD DOMAIN ANALYSIS:
- Scalability requirements (auto-scaling, load handling)
- Security requirements (access control, encryption, compliance)
- Cost constraints and optimization goals
- Performance requirements (latency, throughput, availability)
- Integration requirements with existing systems
- Disaster recovery and backup requirements
- Monitoring and observability needs

COMMON CLOUD REQUIREMENT PATTERNS:
- High availability (99.9%+ uptime)
- Auto-scaling based on demand
- Multi-region deployment for disaster recovery
- Security controls (IAM, network security, encryption)
- Cost monitoring and optimization
- Infrastructure as Code (IaC) for deployment
- Compliance with industry standards"""

        elif domain == "network":
            base_prompt += """

NETWORK DOMAIN ANALYSIS:
- Connectivity requirements (bandwidth, latency, redundancy)
- Security requirements (firewall rules, access policies, VPN)
- Performance requirements (throughput, packet loss, jitter)
- Compliance requirements (regulatory, industry standards)
- Scalability requirements (future growth, capacity planning)
- Monitoring and management requirements
- Integration with existing network infrastructure

COMMON NETWORK REQUIREMENT PATTERNS:
- Site-to-site connectivity with failover
- Secure remote access for users
- Network segmentation and access control
- Bandwidth guarantees and QoS policies
- 24/7 network monitoring and alerting
- Compliance with security frameworks
- Disaster recovery and business continuity"""

        elif domain == "devops":
            base_prompt += """

DEVOPS DOMAIN ANALYSIS:
- CI/CD pipeline requirements (automation, testing, deployment)
- Infrastructure requirements (containerization, orchestration)
- Monitoring requirements (logging, metrics, alerting)
- Security requirements (vulnerability scanning, secrets management)
- Scalability requirements (auto-scaling, load balancing)
- Compliance requirements (audit trails, change management)
- Integration requirements with existing tools

COMMON DEVOPS REQUIREMENT PATTERNS:
- Automated build, test, and deployment pipelines
- Infrastructure as Code (IaC) for reproducibility
- Container orchestration for scalability
- Centralized logging and monitoring
- Security scanning in CI/CD pipeline
- Rollback capabilities for failed deployments
- Environment parity (dev/staging/prod consistency)"""

        elif domain == "dev":
            base_prompt += """

DEVELOPMENT DOMAIN ANALYSIS:
- Functional requirements (features, user stories, APIs)
- Technical requirements (frameworks, databases, integrations)
- Performance requirements (response time, concurrent users)
- Security requirements (authentication, authorization, data protection)
- Scalability requirements (horizontal/vertical scaling)
- Quality requirements (testing, code coverage, maintainability)
- Integration requirements (third-party services, existing systems)

COMMON DEVELOPMENT REQUIREMENT PATTERNS:
- User authentication and authorization
- Data persistence and backup
- API design and documentation
- Error handling and logging
- Input validation and security
- Performance optimization and caching
- Testing strategy (unit, integration, e2e)"""

        base_prompt += f"""

ANALYSIS GUIDELINES:

REQUIREMENT EXTRACTION:
- Extract both explicit and implicit requirements
- Consider industry best practices for the domain
- Identify missing requirements based on common patterns
- Ensure requirements are specific and measurable

DEPENDENCY MAPPING:
- Map technical dependencies (A requires B to function)
- Identify ordering dependencies (A must be done before B)
- Note resource dependencies (shared infrastructure, teams)
- Flag circular dependencies as conflicts

GAP ANALYSIS:
- Compare against domain best practices
- Identify missing non-functional requirements
- Consider integration and operational requirements
- Note assumptions that should be validated

CONFLICT DETECTION:
- Resource conflicts (competing for same resources)
- Technical conflicts (incompatible technologies)
- Constraint conflicts (conflicting limitations)
- Priority conflicts (competing must-haves)

BEST PRACTICES FOR {domain.upper()}:
{chr(10).join(f"- {practice}" for practice in best_practices[:8])}

{self.output_parser.get_format_instructions()}"""
        
        return base_prompt
    
    def _build_user_message(self, state: TechnicalConversationState) -> str:
        """Build user message for requirement analysis"""
        
        message_parts = [
            f"ORIGINAL REQUEST: {state['initial_query']}",
            f"CURRENT INPUT: {state['current_input']}",
            f"DOMAIN: {state['domain']}",
            f"INTENT: {state.get('intent', 'unknown')}",
            f"CONVERSATION STAGE: {state.get('stage', 'initial')}"
        ]
        
        # Add extracted information
        if state.get('extracted_info'):
            extracted_sections = []
            
            for category, items in state['extracted_info'].items():
                if category == 'extraction_metadata':
                    continue
                    
                if isinstance(items, list) and items:
                    item_summaries = []
                    for item in items[:5]:  # Limit to 5 items per category
                        if isinstance(item, dict):
                            name = item.get('name', str(item))
                            value = item.get('value', '')
                            confidence = item.get('confidence', 0.5)
                            item_summaries.append(f"{name} ({confidence:.2f})")
                        else:
                            item_summaries.append(str(item))
                    
                    if item_summaries:
                        extracted_sections.append(f"{category.upper()}: {', '.join(item_summaries)}")
            
            if extracted_sections:
                message_parts.append("EXTRACTED INFORMATION:")
                message_parts.extend(extracted_sections)
        
        # Add conversation history for context
        if state.get('conversation_history'):
            recent_history = state['conversation_history'][-2:]
            history_text = []
            for exchange in recent_history:
                if isinstance(exchange, dict):
                    if 'user' in exchange:
                        history_text.append(f"User: {exchange['user'][:100]}")
                    if 'bot' in exchange:
                        history_text.append(f"Bot: {exchange['bot'][:100]}")
            
            if history_text:
                message_parts.append("CONVERSATION CONTEXT:")
                message_parts.extend(history_text)
        
        # Add any previously identified requirements
        if state.get('requirements'):
            existing_reqs = []
            for req in state['requirements'][:3]:  # Limit to 3 existing requirements
                if isinstance(req, dict):
                    req_summary = f"{req.get('name', 'Unknown')}: {req.get('type', 'unknown')} ({req.get('criticality', 'unknown')})"
                    existing_reqs.append(req_summary)
            
            if existing_reqs:
                message_parts.append("EXISTING REQUIREMENTS:")
                message_parts.extend(existing_reqs)
        
        # Add missing fields for context
        if state.get('missing_fields'):
            message_parts.append(f"MISSING INFORMATION: {', '.join(state['missing_fields'][:5])}")
        
        return "\n\n".join(message_parts)
    
    def _enhance_analysis(
        self, 
        analysis: RequirementAnalysis, 
        state: TechnicalConversationState
    ) -> RequirementAnalysis:
        """Enhance analysis with derived insights"""
        
        # Build dependency graph
        dependency_graph = {}
        for req in analysis.requirements:
            dependency_graph[req.id] = req.depends_on
        
        analysis.dependency_graph = dependency_graph
        
        # Identify critical path (must-have requirements in dependency order)
        critical_path = self._calculate_critical_path(analysis.requirements)
        analysis.critical_path = critical_path
        
        # Calculate completeness score based on domain expectations
        completeness_score = self._calculate_completeness_score(analysis, state['domain'])
        analysis.completeness_score = completeness_score
        
        # Add implicit requirements based on domain patterns
        implicit_requirements = self._identify_implicit_requirements(analysis, state['domain'])
        analysis.implicit_requirements.extend(implicit_requirements)
        
        return analysis
    
    def _calculate_critical_path(self, requirements: List[Requirement]) -> List[str]:
        """Calculate critical path of must-have requirements"""
        
        must_have_reqs = [req for req in requirements if req.criticality == CriticalityLevel.MUST_HAVE]
        
        # Simple topological sort for critical path
        # Build dependency graph for must-have requirements only
        graph = {req.id: [] for req in must_have_reqs}
        in_degree = {req.id: 0 for req in must_have_reqs}
        
        for req in must_have_reqs:
            for dep in req.depends_on:
                if dep in graph:
                    graph[dep].append(req.id)
                    in_degree[req.id] += 1
        
        # Topological sort
        queue = [req_id for req_id, degree in in_degree.items() if degree == 0]
        critical_path = []
        
        while queue:
            req_id = queue.pop(0)
            critical_path.append(req_id)
            
            for dependent in graph[req_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return critical_path
    
    def _calculate_completeness_score(self, analysis: RequirementAnalysis, domain: str) -> float:
        """Calculate how complete the requirement set appears to be"""
        
        domain_expectations = self.domain_templates.get(domain, {}).get('expected_categories', [])
        
        if not domain_expectations:
            return 0.8  # Default score if no domain expectations
        
        # Check coverage of expected requirement categories
        covered_categories = set()
        for req in analysis.requirements:
            covered_categories.add(req.type.value)
            for tag in req.domain_tags:
                covered_categories.add(tag.lower())
        
        coverage_ratio = len(covered_categories.intersection(set(domain_expectations))) / len(domain_expectations)
        
        # Adjust based on number of requirements and confidence
        req_count_factor = min(len(analysis.requirements) / 5, 1.0)  # Expect at least 5 requirements
        avg_confidence = sum(req.confidence for req in analysis.requirements) / len(analysis.requirements) if analysis.requirements else 0.5
        
        completeness_score = (coverage_ratio * 0.5) + (req_count_factor * 0.3) + (avg_confidence * 0.2)
        
        return min(completeness_score, 1.0)
    
    def _identify_implicit_requirements(self, analysis: RequirementAnalysis, domain: str) -> List[str]:
        """Identify requirements that are implied but not explicitly stated"""
        
        implicit_reqs = []
        
        # Get existing requirement types
        existing_types = {req.type for req in analysis.requirements}
        existing_tags = set()
        for req in analysis.requirements:
            existing_tags.update(req.domain_tags)
        
        # Domain-specific implicit requirements
        domain_implications = {
            "cloud": {
                "web_application": ["monitoring", "backup", "security"],
                "database": ["backup", "disaster_recovery", "performance"],
                "high_availability": ["load_balancing", "redundancy", "health_checks"]
            },
            "network": {
                "vpn": ["encryption", "authentication", "firewall_rules"],
                "site_to_site": ["routing", "redundancy", "monitoring"],
                "remote_access": ["authentication", "security_policies", "endpoint_protection"]
            },
            "devops": {
                "ci_cd": ["testing", "security_scanning", "rollback"],
                "containerization": ["orchestration", "monitoring", "security"],
                "cloud_deployment": ["infrastructure_as_code", "monitoring", "backup"]
            },
            "dev": {
                "web_api": ["authentication", "rate_limiting", "documentation"],
                "user_interface": ["accessibility", "responsive_design", "testing"],
                "database": ["backup", "migration", "performance_optimization"]
            }
        }
        
        implications = domain_implications.get(domain, {})
        
        for req in analysis.requirements:
            for tag in req.domain_tags:
                tag_lower = tag.lower()
                if tag_lower in implications:
                    for implied_req in implications[tag_lower]:
                        if implied_req not in existing_tags and implied_req not in implicit_reqs:
                            implicit_reqs.append(f"Implicit: {implied_req} (due to {tag})")
        
        return implicit_reqs[:5]  # Limit to 5 implicit requirements
    
    def _update_state_with_analysis(
        self, 
        state: TechnicalConversationState, 
        analysis: RequirementAnalysis
    ) -> TechnicalConversationState:
        """Update state with requirement analysis results"""
        
        updated_state = state.copy()
        
        # Convert requirements to dict format for state storage
        requirements_dict = [req.dict() for req in analysis.requirements]
        updated_state['requirements'] = requirements_dict
        
        # Update missing fields based on analysis
        missing_fields = list(analysis.missing_requirements)
        
        # Add critical missing requirements from low completeness score
        if analysis.completeness_score < 0.6:
            domain_critical = self._get_critical_missing_fields(state['domain'], analysis)
            missing_fields.extend(domain_critical[:3])
        
        updated_state['missing_fields'] = missing_fields[:10]
        
        # Update accumulated info with analysis metadata
        accumulated_info = updated_state.get('accumulated_info', {})
        accumulated_info['requirement_analysis'] = {
            'total_requirements': len(analysis.requirements),
            'must_have_count': len([r for r in analysis.requirements if r.criticality == CriticalityLevel.MUST_HAVE]),
            'conflicts_count': len(analysis.conflicts),
            'completeness_score': analysis.completeness_score,
            'critical_path_length': len(analysis.critical_path),
            'analysis_confidence': analysis.analysis_confidence,
            'recommendation': analysis.recommendation
        }
        updated_state['accumulated_info'] = accumulated_info
        
        # Update conversation stage based on analysis
        if analysis.completeness_score >= 0.8 and len(analysis.missing_requirements) <= 2:
            updated_state['stage'] = ConversationStage.SUFFICIENT
        elif len(analysis.missing_requirements) > 0:
            updated_state['stage'] = ConversationStage.GATHERING
        else:
            updated_state['stage'] = ConversationStage.ANALYZING
        
        # Update token usage
        token_usage = updated_state.get('token_usage', {}).copy()
        token_usage['requirement_analyzer'] = token_usage.get('requirement_analyzer', 0) + 400
        updated_state['token_usage'] = token_usage
        
        return updated_state
    
    def _get_critical_missing_fields(self, domain: str, analysis: RequirementAnalysis) -> List[str]:
        """Get critical missing fields based on domain and current analysis"""
        
        critical_fields = {
            "cloud": ["security_requirements", "scalability_requirements", "cost_constraints"],
            "network": ["bandwidth_requirements", "security_policies", "redundancy_requirements"],
            "devops": ["deployment_strategy", "monitoring_requirements", "testing_requirements"],
            "dev": ["performance_requirements", "security_requirements", "testing_strategy"]
        }
        
        domain_fields = critical_fields.get(domain, [])
        
        # Check which are actually missing
        existing_tags = set()
        for req in analysis.requirements:
            existing_tags.update(req.domain_tags)
        
        missing_critical = []
        for field in domain_fields:
            field_covered = any(
                field.replace('_', ' ') in req.description.lower() or 
                field.split('_')[0] in ' '.join(req.domain_tags).lower()
                for req in analysis.requirements
            )
            
            if not field_covered:
                missing_critical.append(field.replace('_', ' '))
        
        return missing_critical
    
    def _build_domain_templates(self) -> Dict[str, Dict[str, Any]]:
        """Build domain-specific requirement templates"""
        
        return {
            "cloud": {
                "expected_categories": ["functional", "technical", "quality", "compliance", "constraint"],
                "common_requirements": ["scalability", "security", "monitoring", "backup", "cost_optimization"],
                "critical_dependencies": ["infrastructure", "security", "compliance"]
            },
            "network": {
                "expected_categories": ["functional", "technical", "quality", "compliance"],
                "common_requirements": ["connectivity", "security", "performance", "redundancy", "monitoring"],
                "critical_dependencies": ["infrastructure", "security", "compliance"]
            },
            "devops": {
                "expected_categories": ["functional", "technical", "quality", "constraint"],
                "common_requirements": ["automation", "testing", "deployment", "monitoring", "security"],
                "critical_dependencies": ["infrastructure", "tooling", "security"]
            },
            "dev": {
                "expected_categories": ["functional", "technical", "quality", "constraint"],
                "common_requirements": ["functionality", "performance", "security", "usability", "maintainability"],
                "critical_dependencies": ["architecture", "data", "integration"]
            }
        }
    
    def _load_best_practices(self) -> Dict[str, List[str]]:
        """Load domain-specific best practices"""
        
        return {
            "cloud": [
                "Design for failure and implement auto-recovery",
                "Use Infrastructure as Code for consistent deployments",
                "Implement comprehensive monitoring and alerting",
                "Follow security best practices (least privilege, encryption)",
                "Design for cost optimization and resource efficiency",
                "Plan for disaster recovery and business continuity",
                "Use managed services where possible to reduce operational overhead",
                "Implement proper backup and retention policies"
            ],
            "network": [
                "Implement defense in depth security strategy",
                "Design for redundancy and eliminate single points of failure",
                "Use network segmentation to contain security breaches",
                "Implement proper access controls and authentication",
                "Monitor network performance and security continuously",
                "Plan for capacity growth and future scalability",
                "Document network topology and configuration changes",
                "Implement proper change management processes"
            ],
            "devops": [
                "Implement CI/CD pipelines for automated deployments",
                "Use Infrastructure as Code for reproducible environments",
                "Implement comprehensive testing at all levels",
                "Use containerization for consistency across environments",
                "Implement monitoring, logging, and alerting",
                "Practice security scanning and vulnerability management",
                "Implement proper secret management",
                "Design for rollback and disaster recovery"
            ],
            "dev": [
                "Follow secure coding practices and input validation",
                "Implement comprehensive testing strategy",
                "Use version control and code review processes",
                "Design for scalability and performance",
                "Implement proper error handling and logging",
                "Follow API design best practices",
                "Implement proper authentication and authorization",
                "Design for maintainability and documentation"
            ]
        }
    
    def _apply_cached_result(
        self, 
        state: TechnicalConversationState, 
        cached_analysis: RequirementAnalysis
    ) -> TechnicalConversationState:
        """Apply cached analysis result to state"""
        
        # Record cache hit
        cache_hits = state.get('cache_hits', []).copy()
        cache_hits.append(f"requirement_analyzer:{state['updated_at']}")
        
        updated_state = self._update_state_with_analysis(state, cached_analysis)
        updated_state['cache_hits'] = cache_hits
        
        return updated_state
    
    def _validate_output_state(self, state: TechnicalConversationState) -> None:
        """Validate output state for requirement analyzer"""
        super()._validate_output_state(state)
        
        # Ensure requirements are present
        if 'requirements' not in state:
            raise NodeValidationError("Requirements missing from state")
        
        if not isinstance(state['requirements'], list):
            raise NodeValidationError("Requirements must be a list")
        
        # Validate requirement analysis metadata
        if 'accumulated_info' not in state or 'requirement_analysis' not in state['accumulated_info']:
            raise NodeValidationError("Requirement analysis metadata missing")