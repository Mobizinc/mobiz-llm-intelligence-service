#!/bin/bash

# LangGraph Studio Development Helper Script
# ==========================================
# This script sets up the development environment and starts LangGraph Studio
# with all the necessary configurations for the Mobiz LLM Intelligence Service.

set -e  # Exit on any error

echo "🚀 Starting LangGraph Studio for Mobiz LLM Intelligence Service..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python -m venv venv
    echo "✅ Virtual environment created."
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if LangGraph CLI is installed
if ! command -v langgraph &> /dev/null; then
    echo "📦 Installing LangGraph CLI..."
    pip install -U "langgraph-cli[inmem]"
    echo "✅ LangGraph CLI installed."
fi

# Install/update dependencies if requirements.txt is newer than last install
if [ requirements.txt -nt venv/pyvenv.cfg ] || [ ! -f "venv/.requirements_installed" ]; then
    echo "📦 Installing/updating dependencies..."
    pip install -r requirements.txt
    touch venv/.requirements_installed
    echo "✅ Dependencies updated."
fi

# Set development environment variables
export LANGSMITH_TRACING=false
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "📝 Please update .env with your API keys before running graphs."
    else
        echo "❌ .env.example not found. You'll need to create .env manually."
    fi
fi

# Validate langgraph.json exists
if [ ! -f "langgraph.json" ]; then
    echo "❌ langgraph.json not found. This file is required for LangGraph Studio."
    exit 1
fi

# Check if OpenAI API key is set (basic validation)
if ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo "⚠️  OpenAI API key may not be set in .env file."
    echo "    Please ensure OPENAI_API_KEY is configured for full functionality."
fi

# Parse command line arguments
DEBUG_MODE=false
TUNNEL_MODE=false
PORT=2024

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --tunnel)
            TUNNEL_MODE=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            echo "LangGraph Development Helper"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug     Enable debug mode with debugpy on port 5678"
            echo "  --tunnel    Use secure tunnel (required for Safari)"
            echo "  --port      Set custom port (default: 2024)"
            echo "  --help      Show this help message"
            echo ""
            echo "Available graphs:"
            echo "  - conversation: Main conversation graph with full orchestration"
            echo "  - independent: Lightweight analysis graph"
            echo "  - development: Testing graph for new nodes/edges"
            echo "  - power_platform: Specialized Power Platform graph"
            echo "  - minimal_test: Quick testing graph"
            echo ""
            echo "After starting, access Studio at:"
            echo "  https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:$PORT"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Build the langgraph command
LANGGRAPH_CMD="langgraph dev"

if [ "$DEBUG_MODE" = true ]; then
    echo "🐛 Debug mode enabled. Debugger will be available on port 5678."
    LANGGRAPH_CMD="$LANGGRAPH_CMD --debug-port 5678"
fi

if [ "$TUNNEL_MODE" = true ]; then
    echo "🌐 Tunnel mode enabled for Safari compatibility."
    LANGGRAPH_CMD="$LANGGRAPH_CMD --tunnel"
fi

if [ "$PORT" != "2024" ]; then
    echo "🔧 Using custom port: $PORT"
    LANGGRAPH_CMD="$LANGGRAPH_CMD --port $PORT"
fi

echo ""
echo "🎯 Configuration:"
echo "   - Virtual Environment: ✅ Activated"
echo "   - Debug Mode: $([ "$DEBUG_MODE" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "   - Tunnel Mode: $([ "$TUNNEL_MODE" = true ] && echo "✅ Enabled" || echo "❌ Disabled")"
echo "   - Port: $PORT"
echo "   - Environment: Development"
echo "   - LangSmith Tracing: Disabled"
echo ""

echo "📊 Available graphs in this project:"
echo "   🔄 conversation     - Full LangGraph workflow with routing"
echo "   ⚡ independent      - Lightweight analysis workflow"
echo "   🧪 development     - Testing new nodes and edges"
echo "   🏢 power_platform  - Power Platform specialized"
echo "   🔬 minimal_test    - Quick debugging workflow"
echo ""

echo "🌐 Studio will be available at:"
if [ "$TUNNEL_MODE" = true ]; then
    echo "   (Secure tunnel URL will be displayed when ready)"
else
    echo "   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:$PORT"
fi
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🔄 Shutting down LangGraph Studio..."
    echo "✅ Cleanup complete. Virtual environment is still active."
}

# Set up cleanup trap
trap cleanup EXIT

# Start LangGraph Studio
echo "🚀 Starting LangGraph Studio server..."
echo "📝 Press Ctrl+C to stop the server"
echo ""

# Execute the langgraph command
exec $LANGGRAPH_CMD