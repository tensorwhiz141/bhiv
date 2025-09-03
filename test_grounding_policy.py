#!/usr/bin/env python3
"""
Test Script for Grounding Verification and Template Selection Policy

Demonstrates the new grounding verification and epsilon-greedy template selection
with automatic fallback to more extractive templates when grounding fails.
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any, List
from utils.logger import get_logger
from utils.response_composer import compose_response_with_grounding
from utils.grounding_verifier import verify_content_grounding
from reinforcement.template_selector import template_selector
from config.template_config import RESPONSE_TEMPLATES

logger = get_logger(__name__)

def test_grounding_verification():
    """Test grounding verification functionality."""
    print("\n🔍 Testing Grounding Verification")
    print("=" * 50)
    
    # Test cases with different grounding qualities
    test_cases = [
        {
            "name": "Well-grounded response",
            "generated_text": "According to Smith (2020), artificial intelligence has shown significant progress in natural language processing. The research demonstrates that modern AI systems can achieve human-level performance in text understanding tasks.",
            "source_texts": [
                "Smith et al. (2020) conducted extensive research on AI performance in NLP tasks, showing that contemporary systems match human capabilities in text comprehension.",
                "Recent advances in artificial intelligence, particularly in natural language processing, have been documented by leading researchers."
            ]
        },
        {
            "name": "Poorly-grounded response", 
            "generated_text": "AI will definitely replace all human jobs by 2025. This is inevitable and has been proven by numerous studies.",
            "source_texts": [
                "Some experts suggest AI may impact certain job sectors over the next decade.",
                "Research indicates mixed effects of automation on employment markets."
            ]
        },
        {
            "name": "No source response",
            "generated_text": "This is a response with no backing sources.",
            "source_texts": []
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print("-" * 30)
        
        metrics = verify_content_grounding(
            case["generated_text"],
            case["source_texts"]
        )
        
        print(f"   Source overlap: {metrics.source_overlap:.3f}")
        print(f"   Citation density: {metrics.citation_density:.3f}")
        print(f"   Factual consistency: {metrics.factual_consistency:.3f}")
        print(f"   Overall score: {metrics.overall_score:.3f}")
        print(f"   Grounded: {'✅' if metrics.is_grounded else '❌'}")

def test_template_selection():
    """Test template selection policy."""
    print("\n📝 Testing Template Selection Policy")
    print("=" * 50)
    
    # Simulate multiple task contexts
    test_contexts = [
        {"task_id": "test_1", "input_type": "text", "tags": ["academic"]},
        {"task_id": "test_2", "input_type": "pdf", "tags": ["research"]}, 
        {"task_id": "test_3", "input_type": "text", "tags": ["summary"]},
        {"task_id": "test_4", "input_type": "text", "tags": ["analysis"]},
        {"task_id": "test_5", "input_type": "text", "tags": ["explanation"]}
    ]
    
    print("\nTemplate selections (epsilon-greedy):")
    for i, context in enumerate(test_contexts, 1):
        template = template_selector.select_template(context)
        print(f"   {i}. Task {context['task_id']}: {template.template_id} ({template.name})")
    
    # Test performance update
    print("\nUpdating template performance...")
    template_selector.update_performance("test_1", "generative_standard", 0.8)
    template_selector.update_performance("test_2", "balanced_hybrid", 0.6) 
    template_selector.update_performance("test_3", "extractive_heavy", 0.9)
    
    # Show performance summary
    summary = template_selector.get_performance_summary()
    print("\nTemplate Performance Summary:")
    for template_id, perf in summary["template_performance"].items():
        if perf["count"] > 0:
            print(f"   {template_id}: avg_reward={perf['avg_reward']:.3f}, count={perf['count']}")

def test_grounding_fallback():
    """Test grounding failure fallback mechanism."""
    print("\n🔄 Testing Grounding Fallback Mechanism")
    print("=" * 50)
    
    # Simulate a grounding failure scenario
    from utils.grounding_verifier import GroundingMetrics
    
    failed_metrics = GroundingMetrics(
        source_overlap=0.2,
        citation_density=0.05,
        factual_consistency=0.3,
        overall_score=0.25,
        is_grounded=False,
        details={"test": "simulated_failure"}
    )
    
    fallback_template = template_selector.handle_grounding_failure(
        "test_fallback", 
        "generative_standard", 
        failed_metrics
    )
    
    print(f"Original template: generative_standard")
    print(f"Fallback template: {fallback_template.template_id}")
    print(f"Extractive ratio: {fallback_template.extractive_ratio}")
    print(f"Min citations: {fallback_template.min_citations}")

async def test_full_composition():
    """Test full response composition with grounding and template selection."""
    print("\n🎯 Testing Full Response Composition")
    print("=" * 50)
    
    # Test case with source texts for grounding
    task_id = "composition_test"
    input_data = "Explain the benefits of renewable energy"
    context = {
        "task_id": task_id,
        "agent": "test_agent", 
        "input_type": "text",
        "tags": ["explanation", "environment"]
    }
    source_texts = [
        "Renewable energy sources like solar and wind power provide clean electricity without greenhouse gas emissions.",
        "Studies show renewable energy can reduce electricity costs over time and create sustainable jobs.",
        "The transition to renewable energy is essential for combating climate change and reducing fossil fuel dependency."
    ]
    
    # Compose response
    enhanced_result, trace = compose_response_with_grounding(
        task_id=task_id,
        input_data=input_data,
        context=context,
        source_texts=source_texts
    )
    
    print(f"Task ID: {trace.task_id}")
    print(f"Template used: {trace.template_id}")
    print(f"Grounded: {'✅' if trace.grounded else '❌'}")
    print(f"Grounding score: {trace.grounding_score:.3f}")
    print(f"Fallback used: {'Yes' if trace.fallback_used else 'No'}")
    print(f"Composition time: {trace.composition_time:.3f}s")
    
    if enhanced_result.get("result"):
        print(f"\nGenerated response preview:")
        print(f"   {enhanced_result['result'][:100]}...")

async def test_api_integration():
    """Test API integration with template policy."""
    print("\n🌐 Testing API Integration")
    print("=" * 50)
    
    # Test if MCP Bridge is running
    try:
        health_response = requests.get("http://localhost:8002/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ MCP Bridge is not running. Start it with: python mcp_bridge.py")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to MCP Bridge. Start it with: python mcp_bridge.py") 
        return
    
    print("✅ MCP Bridge is running")
    
    # Test template-enhanced endpoint
    test_payload = {
        "agent": "edumentor_agent",
        "input": "Summarize the key concepts of machine learning",
        "input_type": "text",
        "tags": ["education", "AI"],
        "source_texts": [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "Key ML concepts include supervised learning, unsupervised learning, and reinforcement learning.",
            "Popular algorithms include decision trees, neural networks, and support vector machines."
        ],
        "force_template_id": "balanced_hybrid"
    }
    
    try:
        response = requests.post(
            "http://localhost:8002/handle_task_with_template",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Template-enhanced API call successful")
            
            agent_output = result.get("agent_output", {})
            trace = result.get("composition_trace", {})
            
            print(f"   Template ID: {agent_output.get('template_id', 'N/A')}")
            print(f"   Grounded: {agent_output.get('grounded', 'N/A')}")
            print(f"   Grounding score: {agent_output.get('grounding_score', 'N/A'):.3f}")
            
        else:
            print(f"❌ API call failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {str(e)}")
    
    # Test template performance endpoint
    try:
        perf_response = requests.get("http://localhost:8002/template-performance", timeout=10)
        if perf_response.status_code == 200:
            perf_data = perf_response.json()
            print("\n📊 Template Performance Metrics:")
            
            template_stats = perf_data.get("template_selection", {}).get("template_performance", {})
            for template_id, stats in template_stats.items():
                if stats.get("count", 0) > 0:
                    print(f"   {template_id}: avg_reward={stats['avg_reward']:.3f}, "
                          f"count={stats['count']}, grounding_rate={stats.get('grounding_success_rate', 0):.3f}")
        else:
            print(f"❌ Performance endpoint failed: {perf_response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Performance request failed: {str(e)}")

def print_reward_threshold_questions():
    """Print questions for Shashank about reward thresholds."""
    print("\n❓ Questions for Shashank (Reward Thresholds)")
    print("=" * 50)
    print("1. What reward thresholds should trigger policy updates?")
    print("   - Current fallback threshold: 0.6")
    print("   - Suggested: Update policy when avg reward < 0.5 over 10 tasks")
    print("   - Question: What's the minimum reward threshold for good performance?")
    print()
    print("2. How frequently should we retrain the template selection policy?")
    print("   - Current: On-demand based on performance")
    print("   - Question: Should we retrain every N tasks or based on performance degradation?")
    print()
    print("3. What's the acceptable grounding failure rate?")
    print("   - Current: Fallback triggers at grounding score < 0.5")
    print("   - Question: What grounding success rate indicates healthy template selection?")

def print_feedback_questions():
    """Print questions for Nipun/Nisarg about feedback data."""
    print("\n❓ Questions for Nipun/Nisarg (Feedback Requirements)")
    print("=" * 50)
    print("When feedback arrives, minimal data needed to update policy:")
    print()
    print("Required data structure:")
    print("   {")
    print('     "task_id": "string",')
    print('     "template_id": "string", ')
    print('     "grounded": "boolean",')
    print('     "user_satisfaction": "float (0.0-1.0)",')
    print('     "feedback_type": "string (positive/negative/neutral)"')
    print("   }")
    print()
    print("Questions:")
    print("1. Do you need historical template performance for feedback integration?")
    print("2. Should feedback override RL-learned preferences or complement them?")
    print("3. What's the feedback data format from your system?")
    print("4. How should we handle conflicting feedback vs. automatic grounding scores?")

async def main():
    """Run all tests and display questions."""
    print("🚀 BHIV Grounding & Policy Hook Testing")
    print("=" * 60)
    
    # Run functional tests
    test_grounding_verification()
    test_template_selection()
    test_grounding_fallback()
    await test_full_composition()
    await test_api_integration()
    
    # Display questions for team
    print_reward_threshold_questions()
    print_feedback_questions()
    
    print("\n✅ All tests completed!")
    print("\n📋 Implementation Summary:")
    print("   ✅ Grounding verification with source overlap, citation density, factual consistency")
    print("   ✅ Epsilon-greedy template selection with UCB exploitation")
    print("   ✅ Automatic fallback to extractive templates on grounding failure") 
    print("   ✅ Template_id and grounded flags in response traces")
    print("   ✅ RL logging of template selection as actions")
    print("   ✅ Enhanced reward function with template and grounding bonuses")
    print("   ✅ API endpoints for template management and performance monitoring")

if __name__ == "__main__":
    asyncio.run(main())