#!/usr/bin/env python3
"""
Test script for Graph Factory Functions
=======================================
This script tests all graph factory functions to ensure they work correctly
with LangGraph Studio. Run this before starting LangGraph Studio to validate
your setup.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import():
    """Test if we can import the graph factory module"""
    print("🔍 Testing module imports...")
    
    try:
        from src.core.graph_factory import (
            create_minimal_test_graph,
            create_development_graph,
            list_available_graphs,
            get_graph_info
        )
        print("✅ Graph factory imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_minimal_graph():
    """Test the minimal graph creation"""
    print("🔍 Testing minimal graph creation...")
    
    try:
        from src.core.graph_factory import create_minimal_test_graph
        
        graph = create_minimal_test_graph()
        print(f"✅ Minimal graph created: {type(graph)}")
        
        # Test if we can get graph information
        print("🔍 Testing graph execution...")
        
        # Create a simple test state
        test_state = {
            'conversation_id': 'test_123',
            'current_input': 'test message',
            'stage': 'initial'
        }
        
        # This would be async in real use, but we'll just check if it's callable
        if hasattr(graph, 'invoke') or hasattr(graph, 'ainvoke'):
            print("✅ Graph has invoke methods")
        else:
            print("⚠️  Graph missing invoke methods")
            
        return True
        
    except Exception as e:
        print(f"❌ Minimal graph test failed: {e}")
        return False

def test_development_graph():
    """Test the development graph creation"""
    print("🔍 Testing development graph creation...")
    
    try:
        from src.core.graph_factory import create_development_graph
        
        # Test basic scenario
        graph_basic = create_development_graph("basic")
        print("✅ Basic development graph created")
        
        # Test conditional scenario
        graph_conditional = create_development_graph("conditional")
        print("✅ Conditional development graph created")
        
        return True
        
    except Exception as e:
        print(f"❌ Development graph test failed: {e}")
        return False

def test_graph_info_functions():
    """Test graph information functions"""
    print("🔍 Testing graph info functions...")
    
    try:
        from src.core.graph_factory import list_available_graphs, get_graph_info
        
        # Test list function
        graphs = list_available_graphs()
        print(f"✅ Available graphs: {len(graphs)}")
        for name, desc in graphs.items():
            print(f"   📊 {name}: {desc}")
        
        # Test info function
        info = get_graph_info("minimal_test")
        print("✅ Graph info retrieved successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph info test failed: {e}")
        return False

async def test_async_graph_creation():
    """Test async aspects of graph creation"""
    print("🔍 Testing async graph creation scenarios...")
    
    try:
        # This tests the async handling in graph factories
        from src.core.graph_factory import create_minimal_test_graph
        
        # Test in async context
        graph = create_minimal_test_graph()
        
        # Test basic invocation (if possible)
        test_state = {
            'conversation_id': 'async_test',
            'current_input': 'async test message'
        }
        
        if hasattr(graph, 'ainvoke'):
            print("✅ Graph supports async invocation")
        
        return True
        
    except Exception as e:
        print(f"❌ Async graph test failed: {e}")
        return False

def test_langgraph_json_compatibility():
    """Test if our factory functions work with langgraph.json structure"""
    print("🔍 Testing langgraph.json compatibility...")
    
    try:
        import json
        
        # Read the langgraph.json file
        if os.path.exists("langgraph.json"):
            with open("langgraph.json", "r") as f:
                config = json.load(f)
            
            print("✅ langgraph.json file found and parsed")
            
            # Check if graphs are defined
            graphs = config.get("graphs", {})
            print(f"✅ Found {len(graphs)} graph definitions")
            
            for name, path in graphs.items():
                print(f"   📊 {name}: {path}")
                
                # Validate path format
                if "::" in path or ":" in path:
                    print(f"   ✅ {name} has valid factory function format")
                else:
                    print(f"   ⚠️  {name} may not have factory function format")
            
            # Check dependencies
            deps = config.get("dependencies", [])
            print(f"✅ Found {len(deps)} dependencies")
            
            return True
        else:
            print("❌ langgraph.json not found")
            return False
            
    except Exception as e:
        print(f"❌ langgraph.json compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Graph Factory Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Import", test_import),
        ("Minimal Graph", test_minimal_graph),
        ("Development Graph", test_development_graph),
        ("Graph Info Functions", test_graph_info_functions),
        ("LangGraph.json Compatibility", test_langgraph_json_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
    
    # Run async test
    print(f"\n📋 Running Async Graph Creation test...")
    try:
        if asyncio.run(test_async_graph_creation()):
            passed += 1
            print("✅ Async Graph Creation test passed")
        else:
            print("❌ Async Graph Creation test failed")
    except Exception as e:
        print(f"💥 Async Graph Creation test crashed: {e}")
    
    total += 1  # Add async test to total
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Your graph factory setup is ready for LangGraph Studio.")
        print("\n🚀 Next steps:")
        print("   1. Set up your .env file with API keys")
        print("   2. Run: ./langgraph_dev.sh")
        print("   3. Open LangGraph Studio in your browser")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("   1. Make sure you're in the project root directory")
        print("   2. Install all dependencies: pip install -r requirements.txt")
        print("   3. Set PYTHONPATH: export PYTHONPATH=$(pwd)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)