# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands and Development Workflow

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run full service with FastAPI
uvicorn src.api.rest.main:app --reload --port 8000

# Run independent service (minimal dependencies)
python independent_test.py

# Run specific test service
python test_main.py
```

### Docker Development
```bash
# Build Docker image
docker build -t mobiz-llm-intelligence .

# Run with Docker Compose (includes Redis)
docker-compose up -d

# View logs
docker-compose logs -f mobiz-llm-intelligence
```

### Testing
```bash
# Run tests (basic integration tests available)
python test_main.py
python independent_test.py

# Test health endpoint
curl http://localhost:8000/health
curl http://localhost:8001/health  # Independent service
```

### Deployment
```bash
# Kubernetes deployment
kubectl apply -f kubernetes/

# Health check endpoints
# Main service: http://localhost:8000/health
# Independent service: http://localhost:8001/health
```

## Architecture Overview

This is a **Universal LLM Intelligence Service** built with Python, FastAPI, and LangGraph for enterprise platform analysis and anomaly detection.

### Core Components

**LangGraph Orchestration Layer** (`src/core/`)
- `conversation_graph.py` - Main LangGraph workflow orchestrator with conditional routing
- `routing_engine.py` - Intelligent routing decisions based on intent, confidence, and completeness
- `flow_controller.py` - Circuit breaker, rate limiting, and retry logic
- `supervisor.py` - Multi-agent coordination for complex workflows
- `parallel_executor.py` - Parallel execution of independent analysis tasks

**Agent Nodes** (`src/nodes/`)
- All nodes inherit from `BaseNode` with error handling and telemetry
- `intent_classifier.py` - Classify user intent with confidence scoring
- `information_extractor.py` - Extract structured data from conversations
- `requirement_analyzer.py` - Analyze completeness of requirements
- `sufficiency_checker.py` - Determine if information is sufficient for action
- `implementation_generator.py` - Generate implementations and remediations
- `direct_answer.py` - Provide direct answers for simple queries

**State Management** (`src/models/`)
- `universal_state.py` - Universal intelligence state for cross-platform analysis
- `conversation_state.py` - Technical conversation state for LangGraph workflows
- `independent_state.py` - Lightweight state for independent service mode

**Services** (`src/services/`)
- `llm_manager.py` - OpenAI GPT integration with structured outputs
- `mcp_manager.py` - MCP Protocol integration for extensible tools
- `ai_service.py` - AI processing coordination
- `security_service.py` - Authentication and authorization

### Service Modes

**Full Service Mode** (`src/api/rest/main.py`)
- Complete LangGraph orchestration
- All dependencies (Azure, Redis, OpenTelemetry)
- Enterprise features (checkpointing, telemetry)
- Port: 8000

**Independent Mode** (`src/api/independent_main.py`, `independent_test.py`)
- Minimal dependencies (FastAPI, OpenAI)
- Simplified workflows without LangGraph
- Perfect for development and testing
- Port: 8001

### Platform Support

The service analyzes these platforms:
- **Power Platform** - Canvas Apps, Power Automate, Dataverse
- **ServiceNow** - Workflows, CMDB, Service Catalog  
- **Salesforce** - Cases, Opportunities, Approvals
- **Generic** - Custom enterprise systems via adapters

### Key Design Patterns

**LangGraph State Machine**
- All workflows use `TechnicalConversationState` or `UniversalIntelligenceState`
- Conditional routing based on confidence scores and completeness
- Checkpointing for conversation persistence
- Error recovery and circuit breakers

**Domain-Driven Routing**
- Domain-specific routing rules in `RoutingEngine`
- Confidence thresholds vary by domain complexity
- Multi-agent coordination for complex scenarios

**Microservice Architecture**
- Complete isolation with container-ready design
- Health checks and monitoring built-in
- Azure-native with Table Storage and Key Vault support

## Important Development Notes

### LangGraph Workflow Development
- When adding new nodes, inherit from `BaseNode` in `src/nodes/base/base.py`
- Update routing logic in `RoutingEngine` for new node types
- Add nodes to workflow in `ConversationGraph._build_workflow()`
- All nodes must handle `TechnicalConversationState`

### State Management
- Use `ConversationStateManager` for state operations
- State is automatically checkpointed in Azure Table Storage
- Support both streaming and batch processing modes

### Error Handling
- All services use circuit breakers via `FlowController`
- Structured logging with correlation IDs
- Graceful degradation for service failures
- Health checks report component status

### API Development
- Main analysis endpoint: `POST /v1/analyze`
- All requests require platform type and analysis type
- Response includes anomalies, patterns, and remediation options
- Streaming support available for long-running analyses

### Configuration
- Environment variables for API keys and Azure services
- Domain-specific configuration in routing engine
- MCP server definitions for tool integration
- Rate limiting and timeout configuration per domain

### Testing Strategy
- Use `test_main.py` for basic integration testing
- Use `independent_test.py` for isolated testing without dependencies  
- Health endpoints provide service status validation
- Mock implementations available for development

This service represents production-tested components from KITT (Technical Bots) adapted for universal intelligence microservice architecture.