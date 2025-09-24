"""
Test script for BHIV Core Integration

This script tests the core integration components including:
1. Standard agent interface
2. Core orchestration layer
3. Task execution functionality
"""

import sys
import os
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.text_agent import TextAgent
from agents.audio_agent import AudioAgent
from orchestration.core_orchestrator import execute_task, execute_sequence

def test_standard_interface():
    """Test the standard agent interface implementation."""
    print("Testing standard agent interface...")
    
    # Test TextAgent
    text_agent = TextAgent()
    text_input = {
        "input": "Artificial intelligence is a wonderful field that is developing rapidly.",
        "input_type": "text",
        "task_id": "test-text-001"
    }
    
    try:
        text_result = text_agent.run(text_input)
        print(f"TextAgent result: {text_result.get('status')}")
        print(f"Summary: {text_result.get('result', 'N/A')[:100]}...")
        assert text_result.get('status') == 200, "TextAgent should return status 200"
        print("✓ TextAgent interface test passed")
    except Exception as e:
        print(f"✗ TextAgent interface test failed: {e}")
    
    # Test AudioAgent (will fail without audio file, but testing interface)
    audio_agent = AudioAgent()
    audio_input = {
        "input": "nonexistent_audio.wav",
        "input_type": "audio",
        "task_id": "test-audio-001"
    }
    
    try:
        audio_result = audio_agent.run(audio_input)
        print(f"AudioAgent result: {audio_result.get('status')}")
        # This will likely fail due to missing file, but we're testing the interface
        print("✓ AudioAgent interface test completed (file not found is expected)")
    except Exception as e:
        print(f"AudioAgent interface test encountered expected error: {e}")

def test_core_orchestration():
    """Test the core orchestration layer."""
    print("\nTesting core orchestration layer...")
    
    # Test single task execution
    task_payload = {
        "input": "Explain the theory of relativity in simple terms.",
        "agent": "edumentor_agent",
        "input_type": "text",
        "tags": ["physics", "education"],
        "task_id": "test-orch-001"
    }
    
    try:
        result = execute_task(task_payload)
        print(f"Orchestration result: {result.get('status')}")
        if result.get('status') == 'success':
            agent_output = result.get('agent_output', {})
            print(f"Agent output status: {agent_output.get('status')}")
            print(f"Result: {agent_output.get('result', 'N/A')[:100]}...")
            print("✓ Core orchestration test passed")
        else:
            print(f"Orchestration failed: {result.get('error')}")
    except Exception as e:
        print(f"✗ Core orchestration test failed: {e}")
    
    # Test sequence execution
    task_sequence = [
        {
            "input": "What is machine learning?",
            "agent": "edumentor_agent",
            "input_type": "text",
            "tags": ["ai", "education"],
            "task_id": "test-seq-001"
        },
        {
            "input": "List applications of machine learning.",
            "agent": "edumentor_agent",
            "input_type": "text",
            "tags": ["ai", "applications"],
            "task_id": "test-seq-002"
        }
    ]
    
    try:
        results = execute_sequence(task_sequence)
        print(f"Sequence execution completed with {len(results)} results")
        for i, result in enumerate(results):
            status = result.get('status')
            print(f"Task {i+1} status: {status}")
        print("✓ Sequence execution test completed")
    except Exception as e:
        print(f"✗ Sequence execution test failed: {e}")

def test_agent_registry_compatibility():
    """Test compatibility with agent registry."""
    print("\nTesting agent registry compatibility...")
    
    try:
        from agents.agent_registry import agent_registry
        
        # List available agents
        agents = agent_registry.list_agents()
        print(f"Available agents: {list(agents.keys())}")
        
        # Test agent lookup
        task_context = {
            "task": "process",
            "keywords": ["text", "summarize"],
            "input_type": "text",
            "tags": ["education"]
        }
        
        agent_name = agent_registry.find_agent(task_context)
        print(f"Selected agent for text task: {agent_name}")
        
        agent_config = agent_registry.get_agent_config(agent_name)
        print(f"Agent config: {agent_config}")
        
        print("✓ Agent registry compatibility test passed")
    except Exception as e:
        print(f"✗ Agent registry compatibility test failed: {e}")

def main():
    """Main test function."""
    print("BHIV Core Integration Test Suite")
    print("=" * 40)
    
    test_standard_interface()
    test_core_orchestration()
    test_agent_registry_compatibility()
    
    print("\n" + "=" * 40)
    print("Test suite completed.")

if __name__ == "__main__":
    main()