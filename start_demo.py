#!/usr/bin/env python3
"""
Quick Start Demo Script for Grounding & Template Policy

Starts MCP Bridge and runs the grounding/template policy tests.
"""

import subprocess
import time
import sys
import os
import signal
import requests
from threading import Thread

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ SpaCy model 'en_core_web_sm' is available")
        except OSError:
            print("❌ SpaCy model 'en_core_web_sm' not found")
            print("   Please install with: python -m spacy download en_core_web_sm")
            return False
    except ImportError:
        print("❌ SpaCy not installed")
        print("   Please install with: pip install spacy")
        return False
    
    try:
        import motor
        print("✅ Motor (MongoDB driver) is available")
    except ImportError:
        print("❌ Motor not installed")
        print("   Please install with: pip install motor")
        return False
    
    return True

def start_mcp_bridge():
    """Start the MCP Bridge in background."""
    print("🚀 Starting MCP Bridge...")
    
    # Start MCP Bridge process
    process = subprocess.Popen(
        [sys.executable, "mcp_bridge.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for startup
    print("   Waiting for MCP Bridge to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8002/health", timeout=2)
            if response.status_code == 200:
                print("✅ MCP Bridge is running on http://localhost:8002")
                return process
        except:
            pass
        time.sleep(1)
        print(f"   Waiting... ({i+1}/30)")
    
    print("❌ Failed to start MCP Bridge")
    process.terminate()
    return None

def run_tests():
    """Run the grounding policy tests."""
    print("\n🧪 Running grounding & template policy tests...")
    
    result = subprocess.run([sys.executable, "test_grounding_policy.py"], 
                          capture_output=False, text=True)
    
    return result.returncode == 0

def demo_api_calls():
    """Demonstrate API calls with template policy."""
    print("\n🌐 Demonstrating Template Policy API Calls")
    print("=" * 50)
    
    # Test regular endpoint
    print("\n1. Testing regular /handle_task endpoint:")
    try:
        response = requests.post("http://localhost:8002/handle_task", 
                               json={
                                   "agent": "edumentor_agent",
                                   "input": "Explain machine learning concepts",
                                   "input_type": "text",
                                   "tags": ["education"],
                                   "source_texts": [
                                       "Machine learning uses algorithms to find patterns in data.",
                                       "Common ML techniques include supervised and unsupervised learning."
                                   ]
                               }, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            agent_output = result.get("agent_output", {})
            print(f"   ✅ Success! Template: {agent_output.get('template_id', 'N/A')}")
            print(f"   📊 Grounded: {agent_output.get('grounded', 'N/A')}")
            print(f"   📈 Score: {agent_output.get('grounding_score', 'N/A')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test template-specific endpoint
    print("\n2. Testing /handle_task_with_template endpoint:")
    try:
        response = requests.post("http://localhost:8002/handle_task_with_template",
                               json={
                                   "agent": "edumentor_agent", 
                                   "input": "Summarize renewable energy benefits",
                                   "input_type": "text",
                                   "tags": ["environment"],
                                   "source_texts": [
                                       "Solar power reduces carbon emissions significantly.",
                                       "Wind energy creates sustainable jobs and reduces electricity costs."
                                   ],
                                   "force_template_id": "extractive_heavy"
                               }, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            trace = result.get("composition_trace", {})
            print(f"   ✅ Success! Template: {trace.get('template_id', 'N/A')}")
            print(f"   📊 Grounded: {trace.get('grounded', 'N/A')}")
            print(f"   🔄 Fallback used: {trace.get('fallback_used', 'N/A')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test performance monitoring
    print("\n3. Testing template performance monitoring:")
    try:
        response = requests.get("http://localhost:8002/template-performance", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            template_stats = data.get("template_selection", {}).get("template_performance", {})
            print("   ✅ Performance metrics retrieved:")
            for template_id, stats in template_stats.items():
                if stats.get("count", 0) > 0:
                    print(f"      📊 {template_id}: reward={stats['avg_reward']:.3f}, count={stats['count']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

def main():
    """Main demo function."""
    print("🎯 BHIV Grounding & Template Policy Demo")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return False
    
    # Start MCP Bridge
    bridge_process = start_mcp_bridge()
    if not bridge_process:
        return False
    
    try:
        # Run tests
        test_success = run_tests()
        
        if test_success:
            # Demo API calls
            demo_api_calls()
            
            print("\n🎉 Demo completed successfully!")
            print("\n📋 Key Features Demonstrated:")
            print("   ✅ Grounding verification with tuned thresholds")
            print("   ✅ Epsilon-greedy template selection (ε=0.2)")
            print("   ✅ Automatic fallback to extractive templates")
            print("   ✅ Template performance tracking")
            print("   ✅ RL integration with reward bonuses")
            
            print("\n🔗 Available Endpoints:")
            print("   📡 API Docs: http://localhost:8002/docs")
            print("   ❤️  Health: http://localhost:8002/health")
            print("   📊 Performance: http://localhost:8002/template-performance")
            
        else:
            print("\n❌ Tests failed. Check the output above for details.")
        
    finally:
        # Cleanup
        print("\n🛑 Stopping MCP Bridge...")
        bridge_process.terminate()
        bridge_process.wait()
        print("✅ Cleanup completed")
    
    return test_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)