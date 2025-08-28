# LangGraph Studio Setup Guide

## 🚀 Quick Start

### 1. Set up Virtual Environment & Install Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install all dependencies including LangGraph CLI
pip install -r requirements.txt
pip install -U "langgraph-cli[inmem]"
```

### 2. Configure Environment
```bash
# Copy and update environment file
cp .env.example .env

# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Start LangGraph Studio (Easy Way)
```bash
# Use the helper script (recommended)
./langgraph_dev.sh

# Or with options:
./langgraph_dev.sh --debug    # Enable debugger on port 5678
./langgraph_dev.sh --tunnel   # Use secure tunnel for Safari
./langgraph_dev.sh --help     # See all options
```

### 4. Manual Start (Alternative)
```bash
# Activate virtual environment
source venv/bin/activate

# Set environment variables
export LANGSMITH_TRACING=false
export LOG_LEVEL=DEBUG

# Start LangGraph server
langgraph dev

# Or with debugging:
langgraph dev --debug-port 5678
```

### 5. Access LangGraph Studio
Open your browser and go to:
```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## 📊 Available Graphs

Your project includes 5 pre-configured graphs that will appear in LangGraph Studio:

| Graph Name | Description | Use Case |
|------------|-------------|----------|
| **conversation** | Full LangGraph workflow with routing | Production conversations with complete orchestration |
| **independent** | Lightweight analysis workflow | Quick analysis without full dependencies |
| **development** | Testing new nodes and edges | Safe experimentation during development |
| **power_platform** | Power Platform specialized | Focused on Power Platform analysis |
| **minimal_test** | Quick debugging workflow | Fast testing and validation |

## 🔧 Key Features of This Setup

### ✅ No Constant Exports Required!
- **Dynamic Graph Loading**: Graphs are created on-demand via factory functions
- **Hot Reload**: Changes to graph code are automatically picked up
- **Easy Node Addition**: Just modify your graph classes - no configuration updates needed

### ✅ Development-Friendly
- **Multiple Environments**: Test different scenarios without affecting production graphs
- **Debug Support**: VSCode debugging integration with breakpoints
- **Virtual Environment**: Complete isolation of dependencies

### ✅ Visual Development
- **Real-time Visualization**: See your graph structure and execution flow
- **State Inspection**: Monitor `TechnicalConversationState` changes step by step  
- **Conditional Routing**: Visualize routing decisions from your `RoutingEngine`

## 🛠️ Development Workflow

### Adding New Nodes
1. Create your new node in the appropriate location (e.g., `src/nodes/`)
2. Add it to your graph in `conversation_graph.py` or `independent_graph.py`
3. The factory functions will automatically pick up the changes
4. LangGraph Studio will reload and show your new node

### Testing New Edges/Routing
1. Use the **development** graph for safe experimentation:
   - Modify `create_development_graph()` in `src/core/graph_factory.py`
   - Test different routing scenarios
   - Once validated, move changes to your main graphs

### Debugging Graph Execution
1. Start LangGraph Studio with debug mode:
   ```bash
   ./langgraph_dev.sh --debug
   ```
2. In VSCode, use "Attach to LangGraph Studio" debug configuration
3. Set breakpoints in your node code
4. Execute graphs in Studio and debug step-by-step

## 📋 Testing Your Setup

Before starting development, run the validation script:

```bash
# Test all graph factory functions
./test_graph_factory.py

# Should show all tests passing:
# 📊 Test Results: 6/6 passed
# 🎉 All tests passed! Your graph factory setup is ready for LangGraph Studio.
```

## 🔍 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the project root and virtual environment is activated
cd /path/to/mobiz-llm-intelligence-service
source venv/bin/activate
export PYTHONPATH=$(pwd)
```

**2. Missing Dependencies**
```bash
# Update all dependencies
pip install -r requirements.txt
pip install -U "langgraph-cli[inmem]"
```

**3. Browser Issues**
```bash
# Safari blocks localhost - use tunnel mode
./langgraph_dev.sh --tunnel

# Or use Chrome/Firefox with direct URL:
# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

**4. OpenAI API Issues**
- Ensure your `.env` file has a valid `OPENAI_API_KEY`
- Check your OpenAI account has sufficient credits
- Verify the key starts with `sk-`

### Advanced Debugging

**Enable Detailed Logging:**
```bash
export LOG_LEVEL=DEBUG
export LANGSMITH_TRACING=false
./langgraph_dev.sh --debug
```

**Check Graph Factory:**
```bash
# Test individual factory functions
python src/core/graph_factory.py
```

**Validate Configuration:**
```bash
# Check langgraph.json syntax
python -m json.tool langgraph.json
```

## 🎯 Development Tips

### 1. Use the Right Graph for the Job
- **minimal_test**: Quick validation of simple logic
- **development**: Testing complex routing and new features  
- **conversation**: Full production workflow testing
- **independent**: Testing without external dependencies

### 2. Leverage Hot Reload
- Make changes to your Python files
- LangGraph Studio automatically picks up changes
- No need to restart the server or export graphs

### 3. VSCode Integration
- Use the provided debug configurations
- Set breakpoints in your node code
- Use "Attach to LangGraph Studio" when running with `--debug`

### 4. State Inspection
- Use LangGraph Studio's state viewer to see how `TechnicalConversationState` evolves
- Monitor your routing decisions from `RoutingEngine`
- Check parallel execution paths

### 5. Safe Experimentation
- Always test new features in the **development** graph first
- Use different scenarios: `basic`, `conditional`, etc.
- Move to production graphs only after validation

## 🚀 Next Steps

1. **Start Development**: Run `./langgraph_dev.sh` and explore your graphs
2. **Add New Features**: Create nodes, test in development graph, then integrate
3. **Monitor Performance**: Use Studio's performance metrics to optimize
4. **Scale Up**: Deploy to production when ready

## 📚 Additional Resources

- [LangGraph Documentation](https://docs.langchain.com/langgraph)
- [LangGraph Studio Guide](https://docs.langchain.com/langgraph-platform/quick-start-studio)
- [Project CLAUDE.md](./CLAUDE.md) - Architecture and development guidance

---

**Happy developing! 🎉**

Your LangGraph Studio setup is now ready for visual development, debugging, and rapid iteration without the hassle of constant graph exports.