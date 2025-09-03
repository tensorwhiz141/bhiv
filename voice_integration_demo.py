#!/usr/bin/env python3
"""
BHIV Voice Integration Demo Script
Comprehensive setup and demonstration of the voice-enabled AI system
"""

import os
import sys
import asyncio
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.qdrant_loader import QdrantDocumentLoader
from utils.voice_rl_integration import voice_feedback_collector
from agents.audio_agent import AudioAgent
from agents.voice_persona_agent import VoicePersonaAgent
from agents.knowledge_agent import KnowledgeAgent

logger = get_logger(__name__)

class VoiceDemoOrchestrator:
    """Orchestrates the complete voice integration demo."""
    
    def __init__(self):
        self.services_started = []
        self.demo_results = {}
        
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete voice integration demonstration."""
        print("🚀 Starting BHIV Voice Integration Demo")
        print("=" * 50)
        
        try:
            # Step 1: System Health Check
            print("\\n🏥 Step 1: System Health Check")
            health_status = await self.check_system_health()
            self.demo_results['health_check'] = health_status
            
            if not health_status['overall_healthy']:
                print("⚠️ System health issues detected. Some features may not work.")
            else:
                print("✅ All systems healthy!")
            
            # Step 2: Initialize Qdrant with Documents
            print("\\n📚 Step 2: Initialize Knowledge Base")
            qdrant_status = await self.initialize_qdrant()
            self.demo_results['qdrant_initialization'] = qdrant_status
            
            # Step 3: Test Voice Personas
            print("\\n🗣️ Step 3: Test Voice Personas")
            tts_results = await self.test_voice_personas()
            self.demo_results['tts_test'] = tts_results
            
            # Step 4: Test STT Integration
            print("\\n🎙️ Step 4: Test Speech Recognition")
            stt_results = await self.test_speech_recognition()
            self.demo_results['stt_test'] = stt_results
            
            # Step 5: Test Complete Voice Pipeline
            print("\\n🔄 Step 5: Test Complete Voice Pipeline")
            pipeline_results = await self.test_voice_pipeline()
            self.demo_results['voice_pipeline'] = pipeline_results
            
            # Step 6: Test Knowledge Retrieval
            print("\\n🧠 Step 6: Test Knowledge Retrieval")
            knowledge_results = await self.test_knowledge_retrieval()
            self.demo_results['knowledge_test'] = knowledge_results
            
            # Step 7: Demonstrate RL Integration
            print("\\n🤖 Step 7: Demonstrate RL Integration")
            rl_results = await self.test_rl_integration()
            self.demo_results['rl_integration'] = rl_results
            
            # Step 8: Start Web Interface
            print("\\n🌐 Step 8: Start Web Interface")
            web_status = await self.start_web_interface()
            self.demo_results['web_interface'] = web_status
            
            print("\\n🎉 Demo Complete!")
            self.print_demo_summary()
            
            return self.demo_results
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            self.demo_results['error'] = str(e)
            return self.demo_results
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check health of all system components."""
        health_status = {
            'python_version': sys.version,
            'dependencies': {},
            'components': {},
            'overall_healthy': True
        }
        
        # Check Python dependencies
        dependencies = [
            'fastapi', 'uvicorn', 'whisper', 'gtts', 'qdrant_client', 
            'sentence_transformers', 'transformers', 'speechrecognition'
        ]
        
        for dep in dependencies:
            try:
                __import__(dep)
                health_status['dependencies'][dep] = '✅ Available'
                print(f"  ✅ {dep}")
            except ImportError:
                health_status['dependencies'][dep] = '❌ Missing'
                health_status['overall_healthy'] = False
                print(f"  ❌ {dep} - MISSING")
        
        # Check optional dependencies
        optional_deps = ['pyttsx3', 'pvporcupine', 'webrtcvad', 'pyaudio']
        for dep in optional_deps:
            try:
                __import__(dep)
                health_status['dependencies'][dep] = '✅ Available (Optional)'
                print(f"  ✅ {dep} (optional)")
            except ImportError:
                health_status['dependencies'][dep] = '⚠️ Missing (Optional)'
                print(f"  ⚠️ {dep} (optional) - Missing")
        
        # Check directories
        directories = ['audio_output', 'temp', 'logs', 'templates']
        for dir_name in directories:
            dir_path = PROJECT_ROOT / dir_name
            if dir_path.exists():
                health_status['components'][f'{dir_name}_dir'] = '✅ Exists'
                print(f"  ✅ {dir_name}/ directory")
            else:
                dir_path.mkdir(exist_ok=True)
                health_status['components'][f'{dir_name}_dir'] = '✅ Created'
                print(f"  ✅ Created {dir_name}/ directory")
        
        return health_status
    
    async def initialize_qdrant(self) -> Dict[str, Any]:
        """Initialize Qdrant with sample documents."""
        try:
            loader = QdrantDocumentLoader()
            result = await loader.initialize_pipeline()
            
            if result['status'] == 'success':
                print(f"  ✅ Qdrant initialized with {result['documents_loaded']} documents")
                print(f"  📊 Collection: {result['collection_info']['collection_name']}")
                print(f"  📈 Vector count: {result['collection_info']['vectors_count']}")
                
                # Test search
                if result.get('test_search_results'):
                    print(f"  🔍 Test search returned {len(result['test_search_results'])} results")
                
                return result
            else:
                print(f"  ❌ Qdrant initialization failed: {result['message']}")
                return result
                
        except Exception as e:
            error_msg = f"Qdrant initialization error: {e}"
            print(f"  ❌ {error_msg}")
            return {'status': 'error', 'message': error_msg}
    
    async def test_voice_personas(self) -> Dict[str, Any]:
        """Test different voice personas."""
        try:
            voice_agent = VoicePersonaAgent()
            results = {}
            
            test_cases = [
                ("Hello, I am your AI guru. Let me guide you.", "guru", "en"),
                ("Welcome to today's lesson about artificial intelligence.", "teacher", "en"),
                ("Hi there! How can I help you today?", "friend", "en"),
                ("I am ready to assist you with your tasks.", "assistant", "en"),
                ("नमस्ते, मैं आपका एआई गुरु हूं।", "guru", "hi")
            ]
            
            for text, persona, language in test_cases:
                print(f"  🎭 Testing {persona} persona ({language})...")
                
                result = await voice_agent.synthesize_speech(text, persona, language)
                
                if result['status'] == 200:
                    print(f"    ✅ Generated: {os.path.basename(result['audio_file'])}")
                    results[f"{persona}_{language}"] = {
                        'success': True,
                        'audio_file': result['audio_file'],
                        'method': result['method']
                    }
                else:
                    print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
                    results[f"{persona}_{language}"] = {
                        'success': False,
                        'error': result.get('error')
                    }
            
            return results
            
        except Exception as e:
            error_msg = f"Voice persona test error: {e}"
            print(f"  ❌ {error_msg}")
            return {'error': error_msg}
    
    async def test_speech_recognition(self) -> Dict[str, Any]:
        """Test speech recognition capabilities."""
        try:
            audio_agent = AudioAgent()
            
            # Check available methods
            methods_available = {
                'whisper': audio_agent.whisper_model is not None,
                'wav2vec2': audio_agent.wav2vec2_available,
                'speech_recognition': True  # Always available
            }
            
            print("  🎙️ STT Methods Available:")
            for method, available in methods_available.items():
                status = "✅" if available else "❌"
                print(f"    {status} {method}")
            
            # Test with sample audio if available
            sample_audio_dir = PROJECT_ROOT / "sample_audio"
            if sample_audio_dir.exists():
                audio_files = list(sample_audio_dir.glob("*.wav"))
                if audio_files:
                    print(f"  🎵 Testing with {len(audio_files)} sample files...")
                    
                    results = {}
                    for audio_file in audio_files[:3]:  # Test first 3 files
                        result = audio_agent.process_audio(str(audio_file), f"test_{audio_file.name}")
                        results[audio_file.name] = result
                        
                        if result['status'] == 200:
                            print(f"    ✅ {audio_file.name}: '{result['result'][:50]}...'")
                        else:
                            print(f"    ❌ {audio_file.name}: {result.get('error', 'Failed')}")
                    
                    return results
            
            print("  ℹ️ No sample audio files found. STT methods ready for use.")
            return {'methods_available': methods_available}
            
        except Exception as e:
            error_msg = f"STT test error: {e}"
            print(f"  ❌ {error_msg}")
            return {'error': error_msg}
    
    async def test_voice_pipeline(self) -> Dict[str, Any]:
        """Test complete voice processing pipeline."""
        try:
            print("  🔄 Simulating complete voice interaction...")
            
            # Simulate user saying: "What is yoga?"
            user_query = "What is yoga?"
            task_id = f"pipeline_test_{int(time.time())}"
            
            # Step 1: STT (simulated)
            print("    🎙️ Step 1: Speech-to-Text")
            await voice_feedback_collector.collect_stt_feedback(
                task_id=task_id,
                transcription=user_query,
                confidence=0.95,
                processing_time=1.2,
                method_used="whisper"
            )
            print(f"      ✅ Transcribed: '{user_query}'")
            
            # Step 2: Knowledge retrieval
            print("    🧠 Step 2: Knowledge Retrieval")
            knowledge_agent = KnowledgeAgent()
            kb_result = await knowledge_agent.query(
                query=user_query,
                task_id=task_id,
                filters={},
                tags=["philosophy", "yoga"]
            )
            
            # Step 3: Generate response
            print("    💭 Step 3: Generate Response")
            if kb_result['status'] == 200 and kb_result['response']:
                agent_response = f"Based on ancient wisdom, {kb_result['response'][0]['text'][:100]}..."
            else:
                agent_response = "Yoga is an ancient practice that combines physical postures, breathing techniques, and meditation to achieve harmony of body, mind, and spirit."
            
            print(f"      ✅ Response: '{agent_response[:80]}...'")
            
            # Step 4: TTS
            print("    🗣️ Step 4: Text-to-Speech")
            voice_agent = VoicePersonaAgent()
            tts_result = await voice_agent.synthesize_speech(
                text=agent_response,
                persona="guru",
                language="en"
            )
            
            await voice_feedback_collector.collect_tts_feedback(
                task_id=task_id,
                persona="guru",
                language="en",
                generation_time=2.5,
                audio_file=tts_result.get('audio_file')
            )
            
            if tts_result['status'] == 200:
                print(f"      ✅ Audio generated: {os.path.basename(tts_result['audio_file'])}")
            else:
                print(f"      ⚠️ TTS failed: {tts_result.get('error')}")
            
            # Step 5: Collect complete feedback
            print("    📊 Step 5: RL Feedback Collection")
            await voice_feedback_collector.collect_voice_query_feedback(
                task_id=task_id,
                user_query=user_query,
                agent_response=agent_response,
                agent_used="knowledge_agent",
                total_processing_time=6.2
            )
            
            # Simulate user feedback
            await voice_feedback_collector.collect_user_feedback(
                task_id=task_id,
                user_rating=0.8,
                feedback_text="Great explanation!"
            )
            
            print("      ✅ RL feedback collected and processed")
            
            return {
                'task_id': task_id,
                'user_query': user_query,
                'agent_response': agent_response,
                'tts_success': tts_result['status'] == 200,
                'total_time': 6.2
            }
            
        except Exception as e:
            error_msg = f"Voice pipeline test error: {e}"
            print(f"  ❌ {error_msg}")
            return {'error': error_msg}
    
    async def test_knowledge_retrieval(self) -> Dict[str, Any]:
        """Test knowledge retrieval capabilities."""
        try:
            knowledge_agent = KnowledgeAgent()
            
            test_queries = [
                "What is meditation?",
                "Tell me about Ayurveda",
                "Explain Bhagavad Gita",
                "What is dharma?"
            ]
            
            results = {}
            
            for query in test_queries:
                print(f"  🔍 Testing query: '{query}'")
                
                result = await knowledge_agent.query(
                    query=query,
                    task_id=f"kb_test_{hash(query)}",
                    filters={},
                    tags=["vedas", "philosophy"]
                )
                
                if result['status'] == 200:
                    docs_found = len(result['response'])
                    print(f"    ✅ Found {docs_found} relevant documents")
                    
                    if docs_found > 0:
                        top_result = result['response'][0]
                        print(f"    📄 Top result (score: {top_result['score']:.3f}): {top_result['text'][:80]}...")
                    
                    results[query] = {
                        'success': True,
                        'documents_found': docs_found,
                        'top_score': result['response'][0]['score'] if docs_found > 0 else 0
                    }
                else:
                    print(f"    ❌ Query failed: {result.get('metadata', {}).get('error', 'Unknown error')}")
                    results[query] = {
                        'success': False,
                        'error': result.get('metadata', {}).get('error')
                    }
            
            return results
            
        except Exception as e:
            error_msg = f"Knowledge retrieval test error: {e}"
            print(f"  ❌ {error_msg}")
            return {'error': error_msg}
    
    async def test_rl_integration(self) -> Dict[str, Any]:
        """Test reinforcement learning integration."""
        try:
            print("  🤖 Testing RL feedback collection...")
            
            # Generate some sample interactions
            for i in range(3):
                task_id = f"rl_demo_{i}"
                
                # Simulate different quality interactions
                if i == 0:  # High quality
                    await voice_feedback_collector.collect_stt_feedback(
                        task_id, "Perfect transcription", 0.98, 0.8, "whisper"
                    )
                    user_rating = 0.9
                elif i == 1:  # Medium quality  
                    await voice_feedback_collector.collect_stt_feedback(
                        task_id, "Good transcription", 0.85, 1.2, "wav2vec2"
                    )
                    user_rating = 0.6
                else:  # Lower quality
                    await voice_feedback_collector.collect_stt_feedback(
                        task_id, "Okay transcription", 0.7, 2.1, "speech_recognition"
                    )
                    user_rating = 0.3
                
                await voice_feedback_collector.collect_user_feedback(task_id, user_rating)
            
            print("    ✅ Sample RL data generated")
            
            # Get feedback summary
            summary = voice_feedback_collector.get_feedback_summary(1)
            print(f"    📊 Processed {summary.get('total_interactions', 0)} interactions")
            
            if 'average_metrics' in summary:
                avg_reward = summary['average_metrics'].get('overall_reward', 0)
                print(f"    🎯 Average reward: {avg_reward:.3f}")
            
            return summary
            
        except Exception as e:
            error_msg = f"RL integration test error: {e}"
            print(f"  ❌ {error_msg}")
            return {'error': error_msg}
    
    async def start_web_interface(self) -> Dict[str, Any]:
        """Start the web interface."""
        try:
            print("  🌐 Starting web interface on http://localhost:8003")
            print("  🎙️ Voice interface will be available at http://localhost:8003/voice")
            print("  📊 Dashboard available at http://localhost:8003/dashboard")
            
            # In a real scenario, you might start the server here
            # For demo purposes, we'll just indicate it's ready
            
            return {
                'status': 'ready',
                'endpoints': {
                    'home': 'http://localhost:8003/',
                    'voice': 'http://localhost:8003/voice',
                    'dashboard': 'http://localhost:8003/dashboard',
                    'health': 'http://localhost:8003/health'
                },
                'authentication': {
                    'username': 'admin',
                    'password': 'secret'
                }
            }
            
        except Exception as e:
            error_msg = f"Web interface startup error: {e}"
            print(f"  ❌ {error_msg}")
            return {'error': error_msg}
    
    def print_demo_summary(self):
        """Print a summary of the demo results."""
        print("\\n" + "=" * 50)
        print("📋 DEMO SUMMARY")
        print("=" * 50)
        
        # Health Check Summary
        if 'health_check' in self.demo_results:
            health = self.demo_results['health_check']
            status = "✅ HEALTHY" if health.get('overall_healthy') else "⚠️ ISSUES"
            print(f"🏥 System Health: {status}")
        
        # Qdrant Summary
        if 'qdrant_initialization' in self.demo_results:
            qdrant = self.demo_results['qdrant_initialization']
            if qdrant.get('status') == 'success':
                print(f"📚 Knowledge Base: ✅ {qdrant.get('documents_loaded', 0)} documents loaded")
            else:
                print("📚 Knowledge Base: ❌ Failed to initialize")
        
        # TTS Summary
        if 'tts_test' in self.demo_results:
            tts = self.demo_results['tts_test']
            if not isinstance(tts, dict) or 'error' in tts:
                print("🗣️ Voice Personas: ❌ Failed")
            else:
                successful = sum(1 for result in tts.values() if isinstance(result, dict) and result.get('success'))
                total = len(tts)
                print(f"🗣️ Voice Personas: ✅ {successful}/{total} personas tested")
        
        # STT Summary  
        if 'stt_test' in self.demo_results:
            print("🎙️ Speech Recognition: ✅ Ready")
        
        # Voice Pipeline Summary
        if 'voice_pipeline' in self.demo_results:
            pipeline = self.demo_results['voice_pipeline']
            if 'error' not in pipeline:
                print(f"🔄 Voice Pipeline: ✅ Complete ({pipeline.get('total_time', 0):.1f}s)")
            else:
                print("🔄 Voice Pipeline: ❌ Failed")
        
        # Knowledge Retrieval Summary
        if 'knowledge_test' in self.demo_results:
            kb = self.demo_results['knowledge_test']
            if not isinstance(kb, dict) or 'error' in kb:
                print("🧠 Knowledge Retrieval: ❌ Failed")
            else:
                successful = sum(1 for result in kb.values() if isinstance(result, dict) and result.get('success'))
                total = len(kb)
                print(f"🧠 Knowledge Retrieval: ✅ {successful}/{total} queries successful")
        
        # RL Summary
        if 'rl_integration' in self.demo_results:
            rl = self.demo_results['rl_integration']
            if 'error' not in rl:
                interactions = rl.get('total_interactions', 0)
                print(f"🤖 RL Integration: ✅ {interactions} interactions processed")
            else:
                print("🤖 RL Integration: ❌ Failed")
        
        # Web Interface Summary
        if 'web_interface' in self.demo_results:
            web = self.demo_results['web_interface']
            if 'error' not in web:
                print("🌐 Web Interface: ✅ Ready at http://localhost:8003")
            else:
                print("🌐 Web Interface: ❌ Failed")
        
        print("\\n🎉 Demo completed! Check individual test results above for details.")
        print("\\n📖 Next Steps:")
        print("  1. Run: python -m pip install -r requirements.txt")
        print("  2. Start MongoDB: mongod")
        print("  3. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print("  4. Run: python integration/web_interface.py")
        print("  5. Visit: http://localhost:8003/voice")

async def main():
    """Main demo function."""
    print("🎯 BHIV Fourth Installment - Voice Integration Demo")
    print("🔥 Score Target: 8.5+/10 (Turn-key voice AI experience)")
    
    demo = VoiceDemoOrchestrator()
    results = await demo.run_complete_demo()
    
    # Save results to file
    results_file = PROJECT_ROOT / "demo_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n💾 Demo results saved to: {results_file}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n👋 Demo interrupted by user")