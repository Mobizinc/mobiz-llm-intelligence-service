"""
Information Extractor Node Implementation

This module implements the InformationExtractorNode that extracts structured
technical information from user messages using domain-specific schemas.

Part of Epic 2: Agent Nodes Implementation
Story 2.2: Build InformationExtractorNode
"""

import json
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from .base import BaseNode
from ...models.conversation_state import TechnicalConversationState, DomainType


class ExtractedEntity(BaseModel):
    """Represents a single extracted entity"""
    
    name: str = Field(description="Name/identifier of the entity")
    type: str = Field(description="Type of entity (technology, framework, service, etc.)")
    value: Union[str, List[str], Dict[str, Any]] = Field(description="Extracted value")
    confidence: float = Field(description="Confidence in extraction", ge=0.0, le=1.0)
    context: Optional[str] = Field(None, description="Context where entity was found")
    source: str = Field(description="Source of extraction (current_message, history, implicit)")


class ExtractedInformation(BaseModel):
    """Structured output for information extraction"""
    
    # Core technical entities
    technologies: List[ExtractedEntity] = Field(default=[], description="Technologies mentioned (Azure, Docker, etc.)")
    frameworks: List[ExtractedEntity] = Field(default=[], description="Frameworks mentioned (React, Django, etc.)")
    services: List[ExtractedEntity] = Field(default=[], description="Services mentioned (APIs, databases, etc.)")
    platforms: List[ExtractedEntity] = Field(default=[], description="Platforms mentioned (AWS, Kubernetes, etc.)")
    
    # Requirements and constraints
    functional_requirements: List[ExtractedEntity] = Field(default=[], description="What the system should do")
    technical_requirements: List[ExtractedEntity] = Field(default=[], description="Technical specifications")
    constraints: List[ExtractedEntity] = Field(default=[], description="Limitations or restrictions")
    
    # Environmental context
    environment: List[ExtractedEntity] = Field(default=[], description="Environment details (prod, dev, etc.)")
    scale_metrics: List[ExtractedEntity] = Field(default=[], description="Scale requirements (users, load, etc.)")
    compliance_requirements: List[ExtractedEntity] = Field(default=[], description="Compliance needs (HIPAA, PCI, etc.)")
    
    # Infrastructure details  
    network_details: List[ExtractedEntity] = Field(default=[], description="Network configuration details")
    security_requirements: List[ExtractedEntity] = Field(default=[], description="Security specifications")
    performance_requirements: List[ExtractedEntity] = Field(default=[], description="Performance needs")
    
    # Additional metadata
    extraction_confidence: float = Field(description="Overall confidence in extraction", ge=0.0, le=1.0)
    incomplete_areas: List[str] = Field(default=[], description="Areas where more information is needed")
    implicit_assumptions: List[str] = Field(default=[], description="Assumptions made during extraction")


class InformationExtractorNode(BaseNode):
    """
    Node that extracts structured technical information from conversations.
    
    Supports:
    - Entity extraction for technologies, frameworks, services, constraints
    - Structured information schema with validation
    - Context-aware extraction using conversation history
    - Handling of implicit vs explicit information
    - Confidence scoring for each extracted entity
    - Incremental extraction across messages
    """
    
    def __init__(self, **kwargs):
        super().__init__(node_name="InformationExtractor", **kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=ExtractedInformation)
        
        # Domain-specific extraction schemas
        self.domain_schemas = self._build_domain_schemas()
    
    async def process(self, state: TechnicalConversationState) -> TechnicalConversationState:
        """
        Process state through information extraction.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with extracted information merged
        """
        # Check cache first
        cache_key = self._get_cache_key(state, [state.get('domain', 'general')])
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return self._apply_cached_result(state, cached_result)
        
        # Extract information from current message
        extracted_info = await self._extract_information(state)
        
        # Merge with existing extracted information
        updated_state = self._merge_extracted_information(state, extracted_info)
        
        # Cache the result
        self._set_cache(cache_key, extracted_info)
        
        return updated_state
    
    async def _extract_information(self, state: TechnicalConversationState) -> ExtractedInformation:
        """Extract information from the current message and context"""
        
        system_prompt = self._build_system_prompt(state['domain'])
        user_message = self._build_user_message(state)
        
        extracted_info = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            response_parser=self.output_parser
        )
        
        return extracted_info
    
    def _build_system_prompt(self, domain: str) -> str:
        """Build system prompt for information extraction"""
        
        domain_context = self.get_domain_context(domain)
        domain_schema = self.domain_schemas.get(domain, {})
        
        base_prompt = f"""You are an expert information extractor for {domain} technical conversations.

Your task is to extract structured technical information from user messages, including both explicit and implicit details.

DOMAIN: {domain.upper()}
COMMON TECHNOLOGIES: {', '.join(domain_context.get('technologies', []))}
FOCUS AREAS: {', '.join(domain_context.get('focus_areas', []))}

EXTRACTION CATEGORIES:

1. TECHNOLOGIES: Specific tools, platforms, software mentioned
   Examples: Azure, Docker, Kubernetes, Python, React
   
2. FRAMEWORKS: Development frameworks and libraries
   Examples: Django, Express.js, .NET, Spring Boot
   
3. SERVICES: APIs, databases, external services
   Examples: REST API, PostgreSQL, Redis, Auth0
   
4. PLATFORMS: Cloud platforms, operating systems
   Examples: AWS, Linux, Windows Server, Heroku

5. FUNCTIONAL REQUIREMENTS: What the system should do
   Examples: "user authentication", "data processing", "real-time updates"
   
6. TECHNICAL REQUIREMENTS: Technical specifications
   Examples: "99.9% uptime", "handle 1000 concurrent users", "sub-second response time"
   
7. CONSTRAINTS: Limitations or restrictions  
   Examples: "budget under $1000/month", "must use existing Active Directory"
   
8. ENVIRONMENT: Deployment environment details
   Examples: "production", "development", "on-premise", "hybrid cloud"
   
9. SCALE METRICS: Scale and performance requirements
   Examples: "10,000 users", "1TB storage", "10 req/sec"
   
10. COMPLIANCE: Regulatory or compliance requirements
    Examples: "HIPAA compliant", "SOX compliance", "GDPR requirements"

DOMAIN-SPECIFIC EXTRACTION RULES:"""

        # Add domain-specific rules
        if domain == "cloud":
            base_prompt += """

CLOUD DOMAIN RULES:
- Extract cloud providers (Azure, AWS, GCP) with high confidence
- Identify service types (VM, Storage, Database, Networking)
- Note regions and availability zones
- Capture cost constraints and optimization requirements
- Extract compliance and security requirements
- Identify hybrid/multi-cloud patterns"""

        elif domain == "network":
            base_prompt += """

NETWORK DOMAIN RULES:
- Extract network devices (firewalls, routers, switches) with vendor details
- Identify connection types (VPN, leased line, internet)
- Capture bandwidth and performance requirements
- Note security policies and access controls
- Extract network topology information (site-to-site, hub-spoke)
- Identify protocols and standards (BGP, OSPF, IPsec)"""

        elif domain == "devops":
            base_prompt += """

DEVOPS DOMAIN RULES:
- Extract CI/CD tools and pipeline requirements
- Identify containerization and orchestration needs
- Capture deployment strategies and environments
- Note monitoring and logging requirements
- Extract automation and IaC requirements
- Identify testing and quality gates"""

        elif domain == "dev":
            base_prompt += """

DEVELOPMENT DOMAIN RULES:
- Extract programming languages and versions
- Identify frameworks and libraries
- Capture API requirements and integrations
- Note database and data storage needs
- Extract UI/UX requirements
- Identify testing and security requirements"""

        base_prompt += f"""

EXTRACTION GUIDELINES:

EXPLICIT vs IMPLICIT:
- Explicit: Directly mentioned technologies/requirements
- Implicit: Inferred from context (e.g., "web app" implies HTTP, HTML, etc.)
- Mark source as "explicit" or "implicit" for each entity

CONFIDENCE SCORING:
- 0.9-1.0: Explicitly mentioned with clear context
- 0.7-0.8: Clearly implied or mentioned without full context  
- 0.5-0.6: Inferred from typical patterns
- 0.3-0.4: Weak inference, might be wrong
- 0.0-0.2: Very uncertain, avoid extracting

CONTEXT CONSIDERATION:
- Use conversation history to understand references
- Consider domain context for disambiguation
- Note when information builds on previous messages

VALIDATION RULES:
- Only extract entities with confidence >= 0.3
- Provide context for extracted entities
- Note incomplete areas where more info is needed
- Make reasonable technical assumptions explicit

{self.output_parser.get_format_instructions()}"""
        
        return base_prompt
    
    def _build_user_message(self, state: TechnicalConversationState) -> str:
        """Build user message for information extraction"""
        
        message_parts = [
            f"CURRENT MESSAGE: {state['current_input']}",
            f"DOMAIN: {state['domain']}",
            f"INTENT: {state.get('intent', 'unknown')}"
        ]
        
        # Add conversation history for context
        if state.get('conversation_history'):
            recent_history = state['conversation_history'][-2:]  # Last 2 exchanges
            history_text = []
            for exchange in recent_history:
                if isinstance(exchange, dict):
                    if 'user' in exchange and 'bot' in exchange:
                        history_text.append(f"User: {exchange['user']}")
                        history_text.append(f"Bot: {exchange['bot']}")
            
            if history_text:
                message_parts.append(f"CONVERSATION CONTEXT:\n" + "\n".join(history_text))
        
        # Add existing extracted information for context
        if state.get('extracted_info'):
            existing_info = []
            for key, value in state['extracted_info'].items():
                if value and key != 'extraction_metadata':
                    if isinstance(value, list) and len(value) > 0:
                        existing_info.append(f"{key}: {len(value)} items")
                    elif not isinstance(value, (list, dict)):
                        existing_info.append(f"{key}: {str(value)[:50]}")
            
            if existing_info:
                message_parts.append(f"PREVIOUSLY EXTRACTED:\n" + "\n".join(existing_info[:8]))
        
        # Add any specific requirements or missing fields
        if state.get('missing_fields'):
            message_parts.append(f"MISSING INFO NEEDED: {', '.join(state['missing_fields'][:5])}")
        
        return "\n\n".join(message_parts)
    
    def _merge_extracted_information(
        self, 
        state: TechnicalConversationState, 
        new_info: ExtractedInformation
    ) -> TechnicalConversationState:
        """Merge new extracted information with existing state"""
        
        updated_state = state.copy()
        existing_info = updated_state.get('extracted_info', {})
        
        # Convert new extraction to dict format for state storage
        new_info_dict = self._convert_extraction_to_dict(new_info)
        
        # Merge with existing information
        merged_info = self._deep_merge_extractions(existing_info, new_info_dict)
        
        # Update state
        updated_state['extracted_info'] = merged_info
        
        # Update accumulated info (historical view)
        accumulated_info = updated_state.get('accumulated_info', {})
        accumulated_info.update(merged_info)
        updated_state['accumulated_info'] = accumulated_info
        
        # Update missing fields based on extraction results
        updated_state = self._update_missing_fields(updated_state, new_info)
        
        # Update token usage
        token_usage = updated_state.get('token_usage', {}).copy()
        token_usage['information_extractor'] = token_usage.get('information_extractor', 0) + 300
        updated_state['token_usage'] = token_usage
        
        return updated_state
    
    def _convert_extraction_to_dict(self, extraction: ExtractedInformation) -> Dict[str, Any]:
        """Convert ExtractedInformation to dict format for state storage"""
        
        result = {}
        
        # Convert entity lists to simplified format
        for field_name, entities in extraction.dict().items():
            if field_name in ['extraction_confidence', 'incomplete_areas', 'implicit_assumptions']:
                result[field_name] = entities
            elif isinstance(entities, list) and len(entities) > 0:
                # Convert to simplified entity format
                result[field_name] = []
                for entity in entities:
                    if isinstance(entity, dict):
                        simple_entity = {
                            'name': entity.get('name'),
                            'value': entity.get('value'),
                            'confidence': entity.get('confidence', 0.5),
                            'source': entity.get('source', 'current_message')
                        }
                        result[field_name].append(simple_entity)
        
        # Add extraction metadata
        result['extraction_metadata'] = {
            'last_extraction_timestamp': state.get('updated_at', ''),
            'extraction_confidence': extraction.extraction_confidence,
            'incomplete_areas': extraction.incomplete_areas,
            'implicit_assumptions': extraction.implicit_assumptions
        }
        
        return result
    
    def _deep_merge_extractions(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge extraction results, avoiding duplicates"""
        
        merged = existing.copy()
        
        for key, new_value in new.items():
            if key not in merged:
                merged[key] = new_value
            elif isinstance(new_value, list) and isinstance(merged[key], list):
                # Merge lists, avoiding duplicates by entity name
                existing_names = {
                    item.get('name', str(item)).lower() 
                    for item in merged[key] 
                    if isinstance(item, dict)
                }
                
                for new_item in new_value:
                    if isinstance(new_item, dict):
                        item_name = new_item.get('name', str(new_item)).lower()
                        if item_name not in existing_names:
                            merged[key].append(new_item)
                            existing_names.add(item_name)
                        else:
                            # Update existing item if new one has higher confidence
                            for i, existing_item in enumerate(merged[key]):
                                if isinstance(existing_item, dict) and existing_item.get('name', '').lower() == item_name:
                                    if new_item.get('confidence', 0) > existing_item.get('confidence', 0):
                                        merged[key][i] = new_item
                                    break
            elif key == 'extraction_metadata':
                # Always use new metadata
                merged[key] = new_value
            else:
                # For non-list values, prefer new value if it has higher confidence or is more recent
                merged[key] = new_value
        
        return merged
    
    def _update_missing_fields(
        self, 
        state: TechnicalConversationState, 
        extraction: ExtractedInformation
    ) -> TechnicalConversationState:
        """Update missing fields based on extraction results"""
        
        updated_state = state.copy()
        
        # Use incomplete areas from extraction to update missing fields
        missing_fields = list(extraction.incomplete_areas)
        
        # Add domain-specific missing fields based on what wasn't extracted
        domain_requirements = self._get_domain_requirements(state['domain'])
        
        for requirement in domain_requirements:
            field_category = requirement['category']
            if not any(
                len(state['extracted_info'].get(field_category, [])) > 0 
                for field_category in [requirement['category']]
            ):
                if requirement['field'] not in missing_fields:
                    missing_fields.append(requirement['field'])
        
        # Limit missing fields to most important ones
        updated_state['missing_fields'] = missing_fields[:10]
        
        return updated_state
    
    def _get_domain_requirements(self, domain: str) -> List[Dict[str, str]]:
        """Get domain-specific requirements for completeness checking"""
        
        domain_requirements = {
            "cloud": [
                {"category": "platforms", "field": "cloud_provider"},
                {"category": "services", "field": "required_services"},
                {"category": "scale_metrics", "field": "scale_requirements"},
                {"category": "environment", "field": "deployment_environment"}
            ],
            "network": [
                {"category": "network_details", "field": "network_topology"},
                {"category": "technologies", "field": "network_devices"},
                {"category": "security_requirements", "field": "security_policies"},
                {"category": "performance_requirements", "field": "bandwidth_requirements"}
            ],
            "devops": [
                {"category": "platforms", "field": "deployment_platform"},
                {"category": "technologies", "field": "ci_cd_tools"},
                {"category": "environment", "field": "target_environments"},
                {"category": "technical_requirements", "field": "automation_requirements"}
            ],
            "dev": [
                {"category": "technologies", "field": "programming_languages"},
                {"category": "frameworks", "field": "development_frameworks"},
                {"category": "services", "field": "data_storage"},
                {"category": "technical_requirements", "field": "api_requirements"}
            ]
        }
        
        return domain_requirements.get(domain, [])
    
    def _build_domain_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Build domain-specific extraction schemas"""
        
        return {
            "cloud": {
                "priority_entities": ["platforms", "services", "scale_metrics", "environment"],
                "common_patterns": ["IaaS", "PaaS", "SaaS", "hybrid", "multi-cloud"],
                "key_vendors": ["Microsoft", "Amazon", "Google", "Azure", "AWS", "GCP"]
            },
            "network": {
                "priority_entities": ["technologies", "network_details", "security_requirements"],
                "common_patterns": ["VPN", "firewall", "load balancer", "site-to-site"],
                "key_vendors": ["Cisco", "Palo Alto", "Fortinet", "Meraki", "Juniper"]
            },
            "devops": {
                "priority_entities": ["technologies", "platforms", "technical_requirements"],
                "common_patterns": ["CI/CD", "containerization", "orchestration", "IaC"],
                "key_vendors": ["Docker", "Kubernetes", "Jenkins", "GitHub", "Azure DevOps"]
            },
            "dev": {
                "priority_entities": ["technologies", "frameworks", "services"],
                "common_patterns": ["API", "database", "frontend", "backend", "full-stack"],
                "key_vendors": ["Microsoft", "Google", "Facebook", "Oracle", "MongoDB"]
            }
        }
    
    def _apply_cached_result(
        self, 
        state: TechnicalConversationState, 
        cached_info: ExtractedInformation
    ) -> TechnicalConversationState:
        """Apply cached extraction result to state"""
        
        # Record cache hit
        cache_hits = state.get('cache_hits', []).copy()
        cache_hits.append(f"information_extractor:{state['updated_at']}")
        
        updated_state = self._merge_extracted_information(state, cached_info)
        updated_state['cache_hits'] = cache_hits
        
        return updated_state
    
    def _validate_output_state(self, state: TechnicalConversationState) -> None:
        """Validate output state for information extractor"""
        super()._validate_output_state(state)
        
        # Ensure extracted info is present
        if 'extracted_info' not in state:
            raise NodeValidationError("Extracted information missing from state")
        
        # Validate extraction metadata
        if 'extraction_metadata' not in state['extracted_info']:
            raise NodeValidationError("Extraction metadata missing")
        
        metadata = state['extracted_info']['extraction_metadata']
        if 'extraction_confidence' not in metadata:
            raise NodeValidationError("Extraction confidence missing from metadata")