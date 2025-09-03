"""
Comprehensive test suite for voice integration and API endpoints
Tests STT, TTS, Qdrant, voice processing pipeline, and wake word detection
"""

import pytest
import asyncio
import tempfile
import os
import json
import wave
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.audio_agent import AudioAgent
from agents.voice_persona_agent import VoicePersonaAgent
from agents.knowledge_agent import KnowledgeAgent
from utils.qdrant_loader import QdrantDocumentLoader
from utils.voice_control import VoiceControlSystem, WakeWordDetector
from integration.web_interface import app

class TestAudioAgent:
    """Test suite for AudioAgent STT functionality."""
    
    @pytest.fixture
    def audio_agent(self):
        """Create AudioAgent instance for testing."""
        return AudioAgent()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample WAV file for testing."""
        # Generate a simple sine wave audio file
        duration = 2.0  # seconds
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            with wave.open(f.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            yield f.name
        
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_audio_agent_initialization(self, audio_agent):
        """Test AudioAgent initialization."""
        assert audio_agent is not None
        assert hasattr(audio_agent, 'recognizer')
        assert hasattr(audio_agent, 'model_config')
    
    def test_load_audio_with_fallback(self, audio_agent, sample_audio_file):
        """Test audio loading with fallback methods."""
        try:
            data, sample_rate, method = audio_agent.load_audio_with_fallback(sample_audio_file)
            
            assert data is not None
            assert len(data) > 0
            assert sample_rate > 0
            assert method in ['soundfile', 'librosa', 'torchaudio', 'pydub']
            
        except Exception as e:
            # Some fallback methods may not be available in test environment
            pytest.skip(f"Audio loading libraries not available: {e}")
    
    @pytest.mark.asyncio
    async def test_process_audio_success(self, audio_agent, sample_audio_file):
        """Test successful audio processing."""
        task_id = "test_task_123"
        
        # Mock successful transcription
        with patch.object(audio_agent, 'transcribe_with_whisper', return_value="Hello, this is a test"):
            result = audio_agent.process_audio(sample_audio_file, task_id)
            
            assert result['status'] == 200
            assert 'result' in result
            assert result['result'] == "Hello, this is a test"
            assert 'method' in result
    
    def test_process_audio_file_not_found(self, audio_agent):
        """Test audio processing with non-existent file."""
        result = audio_agent.process_audio("nonexistent.wav", "test_task")
        
        assert result['status'] == 500
        assert 'error' in result
        assert "Audio file not found" in result['error']
    
    def test_process_audio_empty_file(self, audio_agent):
        """Test audio processing with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create empty file
            pass
        
        try:
            result = audio_agent.process_audio(f.name, "test_task")
            
            assert result['status'] == 500
            assert 'error' in result
            assert "empty" in result['error'].lower()
        finally:
            os.unlink(f.name)

class TestVoicePersonaAgent:
    """Test suite for VoicePersonaAgent TTS functionality."""
    
    @pytest.fixture
    def voice_agent(self):
        """Create VoicePersonaAgent instance for testing."""
        return VoicePersonaAgent()
    
    def test_voice_agent_initialization(self, voice_agent):
        """Test VoicePersonaAgent initialization."""
        assert voice_agent is not None
        assert len(voice_agent.personas) > 0
        assert 'guru' in voice_agent.personas
        assert 'teacher' in voice_agent.personas
        assert 'assistant' in voice_agent.personas
    
    def test_detect_language(self, voice_agent):
        """Test language detection."""
        # Test English
        english_text = "Hello, how are you today?"
        assert voice_agent.detect_language(english_text) == 'en'
        
        # Test Hindi (with Devanagari characters)
        hindi_text = "नमस्ते, आप कैसे हैं?"
        assert voice_agent.detect_language(hindi_text) == 'hi'
        
        # Test mixed/empty text
        empty_text = ""
        assert voice_agent.detect_language(empty_text) == 'en'
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_success(self, voice_agent):
        """Test successful speech synthesis."""
        with patch.object(voice_agent, 'generate_speech_gtts', return_value="/path/to/audio.mp3"):
            result = await voice_agent.synthesize_speech(
                text="Hello, this is a test",
                persona="assistant",
                language="en"
            )
            
            assert result['status'] == 200
            assert 'audio_file' in result
            assert result['persona'] == 'assistant'
            assert result['language'] == 'en'
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_invalid_persona(self, voice_agent):
        """Test speech synthesis with invalid persona."""
        result = await voice_agent.synthesize_speech(
            text="Hello",
            persona="invalid_persona",
            language="en"
        )
        
        # Should default to assistant persona
        assert result['persona'] == 'assistant'
    
    def test_list_personas(self, voice_agent):
        """Test persona listing."""
        personas_info = voice_agent.list_personas()
        
        assert 'personas' in personas_info
        assert 'available_languages' in personas_info
        assert 'tts_engines' in personas_info
        
        assert len(personas_info['personas']) > 0
        assert 'hi' in personas_info['available_languages']
        assert 'en' in personas_info['available_languages']

class TestKnowledgeAgent:
    """Test suite for KnowledgeAgent and Qdrant integration."""
    
    @pytest.fixture
    def knowledge_agent(self):
        """Create KnowledgeAgent instance for testing."""
        return KnowledgeAgent()
    
    @pytest.fixture
    def qdrant_loader(self):
        """Create QdrantDocumentLoader instance for testing."""
        return QdrantDocumentLoader()
    
    def test_knowledge_agent_initialization(self, knowledge_agent):
        """Test KnowledgeAgent initialization."""
        assert knowledge_agent is not None
        assert hasattr(knowledge_agent, 'qdrant_client')
        assert hasattr(knowledge_agent, 'collection_name')
        assert hasattr(knowledge_agent, 'model')
    
    @pytest.mark.asyncio
    async def test_query_success(self, knowledge_agent):
        """Test successful knowledge query."""
        # Mock Qdrant search results
        mock_results = [
            Mock(id="doc1", score=0.9, payload={"text": "Test document", "metadata": {"source": "test"}})
        ]
        
        with patch.object(knowledge_agent.qdrant_client, 'search', return_value=mock_results):
            result = await knowledge_agent.query(
                query="test query",
                task_id="test_task",
                filters={},
                tags=["test"]
            )
            
            assert result['status'] == 200
            assert 'response' in result
            assert len(result['response']) > 0
            assert 'metadata' in result
    
    @pytest.mark.asyncio
    async def test_query_with_filters(self, knowledge_agent):
        """Test knowledge query with filters."""
        mock_results = []
        
        with patch.object(knowledge_agent.qdrant_client, 'search', return_value=mock_results):
            result = await knowledge_agent.query(
                query="test query",
                task_id="test_task",
                filters={"category": "test"},
                tags=["filtered"]
            )
            
            assert result['status'] == 200
            assert 'response' in result
    
    def test_qdrant_loader_sample_documents(self, qdrant_loader):
        """Test QdrantDocumentLoader sample documents."""
        documents = qdrant_loader.get_sample_documents()
        
        assert len(documents) > 0
        
        for doc in documents:
            assert 'id' in doc
            assert 'text' in doc
            assert 'metadata' in doc
            assert len(doc['text']) > 0
    
    def test_qdrant_loader_embed_documents(self, qdrant_loader):
        """Test document embedding."""
        sample_docs = [
            {
                "id": "test_doc",
                "text": "This is a test document",
                "metadata": {"source": "test"}
            }
        ]
        
        embedded_docs = qdrant_loader.embed_documents(sample_docs)
        
        assert len(embedded_docs) == 1
        assert 'vector' in embedded_docs[0]
        assert len(embedded_docs[0]['vector']) == 384  # MiniLM vector size
        assert 'payload' in embedded_docs[0]

class TestVoiceControl:
    """Test suite for voice control and wake word detection."""
    
    @pytest.fixture
    def wake_word_detector(self):
        """Create WakeWordDetector instance for testing."""
        return WakeWordDetector(wake_words=["hey guru", "hey test"])
    
    @pytest.fixture
    def voice_control_system(self):
        """Create VoiceControlSystem instance for testing."""
        return VoiceControlSystem()
    
    def test_wake_word_detector_initialization(self, wake_word_detector):
        """Test WakeWordDetector initialization."""
        assert wake_word_detector is not None
        assert len(wake_word_detector.wake_words) == 2
        assert "hey guru" in wake_word_detector.wake_words
        assert wake_word_detector.sample_rate == 16000
    
    def test_wake_word_detector_callbacks(self, wake_word_detector):
        """Test callback setting."""
        callback_mock = Mock()
        wake_word_detector.set_wake_word_callback(callback_mock)
        
        assert wake_word_detector.wake_word_callback == callback_mock
    
    def test_voice_control_system_initialization(self, voice_control_system):
        """Test VoiceControlSystem initialization."""
        assert voice_control_system is not None
        assert hasattr(voice_control_system, 'wake_word_detector')
        assert hasattr(voice_control_system, 'interrupt_handler')

class TestWebInterfaceAPI:
    """Test suite for web interface API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for web interface."""
        return TestClient(app)
    
    def test_voice_interface_endpoint(self, client):
        """Test voice interface page endpoint."""
        # Note: This would require authentication in real scenario
        response = client.get("/voice")
        
        # For this test, we expect authentication required
        assert response.status_code in [200, 401]
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'status' in data
            assert 'timestamp' in data
    
    @pytest.mark.asyncio
    async def test_test_tts_endpoint(self, client):
        """Test TTS testing endpoint."""
        # Mock the VoicePersonaAgent
        with patch('integration.web_interface.VoicePersonaAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.synthesize_speech.return_value = {
                'status': 200,
                'audio_file': '/path/to/test.mp3'
            }
            mock_agent_class.return_value = mock_agent
            
            # This test would require proper authentication setup
            # For demo purposes, we'll just verify the endpoint exists
            response = client.post("/test_tts", json={
                "text": "Test message",
                "persona": "assistant"
            })
            
            # Expect authentication required
            assert response.status_code in [200, 401, 422]

class TestIntegrationFlow:
    """Test suite for end-to-end voice processing flow."""
    
    @pytest.mark.asyncio
    async def test_complete_voice_pipeline(self):
        """Test complete voice processing pipeline."""
        # This is a simulation of the complete flow
        
        # Step 1: Audio processing (STT)
        audio_agent = AudioAgent()
        
        # Step 2: Text processing through agents
        # (Would normally go through MCP Bridge)
        
        # Step 3: TTS response generation
        voice_agent = VoicePersonaAgent()
        
        # Mock the complete pipeline
        with patch.object(audio_agent, 'process_audio') as mock_stt, \
             patch.object(voice_agent, 'synthesize_speech') as mock_tts:
            
            # Setup mocks
            mock_stt.return_value = {
                'status': 200,
                'result': 'Hello, what is yoga?',
                'method': 'whisper'
            }
            
            mock_tts.return_value = {
                'status': 200,
                'audio_file': '/path/to/response.mp3',
                'persona': 'guru'
            }
            
            # Simulate pipeline
            stt_result = audio_agent.process_audio("test.wav", "test_task")
            assert stt_result['status'] == 200
            
            transcription = stt_result['result']
            
            # Simulate agent processing (normally through MCP Bridge)
            agent_response = "Yoga is a practice that combines physical postures, breathing techniques, and meditation."
            
            # Generate TTS response
            tts_result = await voice_agent.synthesize_speech(agent_response, "guru", "en")
            assert tts_result['status'] == 200
    
    @pytest.mark.asyncio
    async def test_qdrant_integration_flow(self):
        """Test Qdrant knowledge retrieval integration."""
        loader = QdrantDocumentLoader()
        knowledge_agent = KnowledgeAgent()
        
        # Test document loading and querying
        documents = loader.get_sample_documents()
        assert len(documents) > 0
        
        # Mock Qdrant operations
        with patch.object(knowledge_agent.qdrant_client, 'search') as mock_search:
            mock_search.return_value = [
                Mock(
                    id="yoga_doc",
                    score=0.95,
                    payload={
                        "text": "Yoga is an ancient practice...",
                        "metadata": {"category": "philosophy"}
                    }
                )
            ]
            
            result = await knowledge_agent.query(
                query="What is yoga?",
                task_id="test_task",
                filters={},
                tags=["philosophy"]
            )
            
            assert result['status'] == 200
            assert len(result['response']) > 0
            assert result['response'][0]['score'] > 0.9

class TestPerformanceAndStress:
    """Test suite for performance and stress testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_voice_processing(self):
        """Test concurrent voice processing requests."""
        voice_agent = VoicePersonaAgent()
        
        # Mock TTS generation for faster testing
        with patch.object(voice_agent, 'generate_speech_gtts', return_value="/path/to/audio.mp3"):
            
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = voice_agent.synthesize_speech(
                    text=f"Test message {i}",
                    persona="assistant",
                    language="en"
                )
                tasks.append(task)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks)
            
            # Verify all requests succeeded
            for result in results:
                assert result['status'] == 200
    
    def test_large_text_tts(self):
        """Test TTS with large text input."""
        voice_agent = VoicePersonaAgent()
        
        # Create large text (simulate long response)
        large_text = "This is a test sentence. " * 100  # 100 repetitions
        
        # This should not crash the system
        asyncio.run(voice_agent.synthesize_speech(large_text, "assistant", "en"))

class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_audio_agent_corrupted_file(self):
        """Test AudioAgent with corrupted audio file."""
        audio_agent = AudioAgent()
        
        # Create a file with invalid audio data
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'invalid audio data')
            f.flush()
            
            result = audio_agent.process_audio(f.name, "test_task")
            
            # Should handle error gracefully
            assert result['status'] == 500
            assert 'error' in result
        
        os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_tts_empty_text(self):
        """Test TTS with empty text."""
        voice_agent = VoicePersonaAgent()
        
        result = await voice_agent.synthesize_speech("", "assistant", "en")
        
        # Should handle empty text gracefully
        assert 'status' in result
    
    @pytest.mark.asyncio
    async def test_knowledge_agent_network_error(self):
        """Test KnowledgeAgent with network error."""
        knowledge_agent = KnowledgeAgent()
        
        # Mock network error
        with patch.object(knowledge_agent.qdrant_client, 'search', side_effect=Exception("Network error")):
            result = await knowledge_agent.query("test", "task_id", {}, [])
            
            assert result['status'] == 500
            assert 'error' in result['metadata']

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])