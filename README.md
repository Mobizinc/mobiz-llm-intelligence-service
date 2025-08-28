# Mobiz LLM Intelligence Service

Universal LLM Intelligence Microservice powered by LangGraph for anomaly detection, pattern learning, and automated remediation across enterprise platforms.

## 🎯 Overview

This microservice provides intelligent analysis capabilities for:
- Power Platform (Canvas Apps, Power Automate, Dataverse)
- ServiceNow (Workflows, CMDB, Service Catalog)
- Salesforce (Cases, Opportunities, Approvals)
- Generic enterprise systems via adapters

## 🏗️ Architecture

Built on production-tested components from KITT (Technical Bots) with LangGraph orchestration:

- **LangGraph State Machine**: Intelligent workflow orchestration
- **GPT-5 Integration**: Latest OpenAI models with structured outputs
- **MCP Protocol Support**: Extensible tool integration
- **Microservice Design**: Complete isolation and scalability
- **Azure Native**: Table Storage, Key Vault, Container Apps ready

## 🚀 Features

### Core Capabilities
- **Intent Classification**: Understand user queries with confidence scoring
- **Information Extraction**: Extract structured data from conversations
- **Anomaly Detection**: Identify patterns and bottlenecks across platforms
- **Remediation Generation**: Create executable scripts and fixes
- **Cross-Platform Correlation**: Find relationships across different systems
- **Pattern Learning**: Learn from investigations for future automation

### API Interfaces
- **REST API**: Standard HTTP endpoints for all operations
- **WebSocket**: Real-time streaming for long-running analyses
- **gRPC**: High-performance service-to-service communication (future)
- **Client Libraries**: Python SDK with TypeScript planned

## 📋 Quick Start

### Prerequisites
- Python 3.12+
- OpenAI API key
- Azure services (Table Storage, Key Vault)

### Local Development
```bash
# Clone repository
git clone https://github.com/Mobizinc/mobiz-llm-intelligence-service
cd mobiz-llm-intelligence-service

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run service
uvicorn src.api.rest.main:app --reload
```

### Docker Deployment
```bash
# Build container
docker build -t mobiz-llm-intelligence .

# Run container
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key mobiz-llm-intelligence
```

## 📊 Usage Examples

### Analyze Power Platform App
```python
from mobiz_llm_client import MobizLLMClient

client = MobizLLMClient("http://localhost:8000", api_key="your-key")

# Analyze Canvas App for anomalies
result = await client.analyze_power_platform({
    "app_id": "12345",
    "app_name": "Customer Portal",
    "context_graph": { /* app structure */ }
})

print(f"Found {len(result.anomalies)} issues")
```

### Generate Remediation Script
```python
# Generate fix for identified issue
remediation = await client.generate_remediation({
    "issue_type": "stuck_approval",
    "affected_entities": ["workorder_123", "workorder_456"],
    "root_cause": "missing_approver"
})

print(remediation.script)  # PowerShell/Python script
```

## 🛠️ Development

### Project Structure
```
src/
├── core/                 # LangGraph orchestration
├── nodes/               # Analysis nodes
├── models/              # Data models
├── services/            # Business logic
├── api/                # API layers
└── clients/            # Client libraries
```

### Adding New Platforms
1. Create adapter in `src/services/adapters/`
2. Define platform-specific nodes in `src/nodes/`
3. Update routing configuration
4. Add MCP server definition if needed

## 🔧 Configuration

See `config/` directory for:
- MCP server definitions
- Platform adapter settings
- Routing rules
- Domain-specific configurations

## 📈 Monitoring

- **Health Checks**: `/health` endpoint
- **Metrics**: Prometheus-compatible metrics
- **Tracing**: LangSmith integration
- **Logging**: Structured JSON logging

## 🚀 Deployment

### Kubernetes
```bash
kubectl apply -f kubernetes/
```

### Azure Container Apps
```bash
az containerapp create --resource-group your-rg \
  --name mobiz-llm-intelligence \
  --image mobizinc/llm-intelligence:latest
```

## 📄 License

Private - Mobizinc Internal Use

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.