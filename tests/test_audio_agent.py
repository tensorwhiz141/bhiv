import pytest
import uuid
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from agents.audio_agent import AudioAgent


class TestAudioAgent:
    """Test suite for AudioAgent."""
    
    @pytest.fixture
    def audio_agent(self):
        """Create an AudioAgent instance for testing."""
        return AudioAgent()
    
    @pytest.fixture
    def sample_audio_path(self):
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Write some fake WAV header data
            temp_file.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            temp_path = temp_file.name
        return temp_path
    
    def teardown_method(self, method):
        """Clean up temporary files after each test."""
        import glob
        for temp_file in glob.glob('/tmp/tmp*.wav'):
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_audio_agent_initialization(self, audio_agent):
        """Test AudioAgent initialization."""
        assert audio_agent is not None
        assert hasattr(audio_agent, 'processor')
        assert hasattr(audio_agent, 'model')
    
    @patch('agents.audio_agent.Wav2Vec2Processor')
    @patch('agents.audio_agent.Wav2Vec2ForCTC')
    def test_initialization_with_mocks(self, mock_model_class, mock_processor_class):
        """Test initialization with mocked dependencies."""
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = AudioAgent()
        
        assert agent.processor == mock_processor
        assert agent.model == mock_model
        mock_processor_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
    
    @patch('agents.audio_agent.librosa.load')
    @patch('agents.audio_agent.torch')
    def test_process_audio_success(self, mock_torch, mock_librosa, audio_agent, sample_audio_path):
        """Test successful audio processing."""
        # Setup mocks
        mock_audio_data = [0.1, 0.2, 0.3, 0.4]  # Fake audio samples
        mock_sample_rate = 16000
        mock_librosa.return_value = (mock_audio_data, mock_sample_rate)
        
        audio_agent.processor = Mock()
        audio_agent.model = Mock()
        
        # Mock processor output
        mock_inputs = {"input_values": "fake_tensor"}
        audio_agent.processor.return_value = mock_inputs
        
        # Mock model output
        mock_logits = Mock()
        mock_model_output = Mock()
        mock_model_output.logits = mock_logits
        audio_agent.model.return_value = mock_model_output
        
        # Mock torch operations
        mock_predicted_ids = "fake_predicted_ids"
        mock_torch.argmax.return_value = mock_predicted_ids
        
        # Mock processor decode
        audio_agent.processor.decode.return_value = "This is a test transcription"
        
        task_id = str(uuid.uuid4())
        
        # Execute
        result = audio_agent.process_audio(sample_audio_path, task_id)
        
        # Assertions
        assert result['status'] == 200
        assert result['result'] == "This is a test transcription"
        assert result['model'] == 'audio_agent'
        assert 'processing_time' in result
        assert result['keywords'] == ['audio', 'transcription']
        
        # Verify method calls
        mock_librosa.assert_called_once_with(sample_audio_path, sr=16000)
        audio_agent.processor.assert_called_once()
        audio_agent.model.assert_called_once()
        
        # Clean up
        os.unlink(sample_audio_path)
    
    def test_process_audio_file_not_found(self, audio_agent):
        """Test audio processing with non-existent file."""
        task_id = str(uuid.uuid4())
        result = audio_agent.process_audio("nonexistent.wav", task_id)
        
        assert result['status'] == 500
        assert 'error' in result
        assert "Audio processing failed" in result['error']
    
    @patch('agents.audio_agent.librosa.load')
    def test_process_audio_with_exception(self, mock_librosa, audio_agent, sample_audio_path):
        """Test audio processing when an exception occurs."""
        # Setup mock to raise exception
        mock_librosa.side_effect = Exception("Audio loading error")
        
        task_id = str(uuid.uuid4())
        result = audio_agent.process_audio(sample_audio_path, task_id)
        
        assert result['status'] == 500
        assert 'error' in result
        assert "Audio processing failed" in result['error']
        
        # Clean up
        os.unlink(sample_audio_path)
    
    @patch('agents.audio_agent.librosa.load')
    def test_process_audio_empty_transcription(self, mock_librosa, audio_agent, sample_audio_path):
        """Test handling of empty transcription."""
        # Setup mocks
        mock_librosa.return_value = ([0.0] * 1000, 16000)  # Silent audio
        
        audio_agent.processor = Mock()
        audio_agent.model = Mock()
        audio_agent.processor.return_value = {"input_values": "fake_tensor"}
        
        mock_model_output = Mock()
        mock_model_output.logits = "fake_logits"
        audio_agent.model.return_value = mock_model_output
        
        # Mock empty transcription
        audio_agent.processor.decode.return_value = ""
        
        task_id = str(uuid.uuid4())
        result = audio_agent.process_audio(sample_audio_path, task_id)
        
        # Should still return success but with empty result
        assert result['status'] == 200
        assert result['result'] == ""
        
        # Clean up
        os.unlink(sample_audio_path)
    
    @patch('agents.audio_agent.replay_buffer')
    @patch('agents.audio_agent.get_reward_from_output')
    @patch.object(AudioAgent, 'process_audio')
    def test_run_success(self, mock_process, mock_reward, mock_buffer, audio_agent, sample_audio_path):
        """Test successful run method."""
        # Setup mocks
        mock_process_result = {
            'status': 200,
            'result': 'This is a test audio transcription',
            'model': 'audio_agent',
            'keywords': ['audio', 'transcription'],
            'processing_time': 2.3
        }
        mock_process.return_value = mock_process_result
        mock_reward.return_value = 0.85
        
        task_id = str(uuid.uuid4())
        
        # Execute
        result = audio_agent.run(sample_audio_path, "", "edumentor_agent", "audio", task_id)
        
        # Assertions
        assert result['status'] == 200
        assert result['agent'] == 'audio_agent'
        assert result['input_type'] == 'audio'
        assert result['file_path'] == sample_audio_path
        
        # Verify methods were called
        mock_process.assert_called_once_with(sample_audio_path, task_id)
        mock_buffer.add_run.assert_called_once()
        mock_reward.assert_called_once()
        
        # Clean up
        os.unlink(sample_audio_path)
    
    @patch('agents.audio_agent.replay_buffer')
    @patch('agents.audio_agent.get_reward_from_output')
    @patch.object(AudioAgent, 'process_audio')
    def test_run_processing_failure(self, mock_process, mock_reward, mock_buffer, audio_agent):
        """Test run method when audio processing fails."""
        # Setup mock to return error
        mock_process_result = {
            'status': 500,
            'error': 'Audio processing failed: File not found',
            'keywords': []
        }
        mock_process.return_value = mock_process_result
        mock_reward.return_value = 0.0
        
        audio_path = "nonexistent.wav"
        task_id = str(uuid.uuid4())
        
        # Execute
        result = audio_agent.run(audio_path, "", "edumentor_agent", "audio", task_id)
        
        # Assertions
        assert result['status'] == 500
        assert 'error' in result
        assert result['agent'] == 'audio_agent'
        assert result['file_path'] == audio_path
        
        # Verify replay buffer was still called (for error tracking)
        mock_buffer.add_run.assert_called_once()
    
    @patch('agents.audio_agent.librosa.load')
    def test_process_audio_different_formats(self, mock_librosa, audio_agent):
        """Test processing different audio formats."""
        # Setup mock
        mock_librosa.return_value = ([0.1, 0.2, 0.3], 16000)
        
        audio_agent.processor = Mock()
        audio_agent.model = Mock()
        audio_agent.processor.return_value = {"input_values": "fake_tensor"}
        
        mock_model_output = Mock()
        mock_model_output.logits = "fake_logits"
        audio_agent.model.return_value = mock_model_output
        audio_agent.processor.decode.return_value = "Test transcription"
        
        # Test different audio formats
        formats = ['.wav', '.mp3', '.ogg', '.flac']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as temp_file:
                temp_file.write(b'fake_audio_data')
                temp_path = temp_file.name
            
            try:
                task_id = str(uuid.uuid4())
                result = audio_agent.process_audio(temp_path, task_id)
                assert result['status'] == 200
                assert result['result'] == "Test transcription"
            finally:
                os.unlink(temp_path)
    
    @patch('agents.audio_agent.librosa.load')
    def test_process_audio_different_sample_rates(self, mock_librosa, audio_agent, sample_audio_path):
        """Test processing audio with different sample rates."""
        # Test that librosa is called with sr=16000 regardless of original sample rate
        mock_librosa.return_value = ([0.1, 0.2, 0.3], 16000)
        
        audio_agent.processor = Mock()
        audio_agent.model = Mock()
        audio_agent.processor.return_value = {"input_values": "fake_tensor"}
        
        mock_model_output = Mock()
        mock_model_output.logits = "fake_logits"
        audio_agent.model.return_value = mock_model_output
        audio_agent.processor.decode.return_value = "Test transcription"
        
        task_id = str(uuid.uuid4())
        result = audio_agent.process_audio(sample_audio_path, task_id)
        
        # Verify librosa was called with sr=16000 (required by wav2vec2)
        mock_librosa.assert_called_once_with(sample_audio_path, sr=16000)
        assert result['status'] == 200
        
        # Clean up
        os.unlink(sample_audio_path)
    
    @patch('agents.audio_agent.logger')
    @patch.object(AudioAgent, 'process_audio')
    def test_logging_behavior(self, mock_process, mock_logger, audio_agent, sample_audio_path):
        """Test that appropriate logging occurs."""
        mock_process.return_value = {
            'status': 200,
            'result': 'Test transcription',
            'model': 'audio_agent',
            'keywords': ['audio', 'transcription']
        }
        
        task_id = str(uuid.uuid4())
        audio_agent.run(sample_audio_path, "", "edumentor_agent", "audio", task_id)
        
        # Verify logging calls were made
        assert mock_logger.info.called
        assert mock_logger.debug.called
        
        # Clean up
        os.unlink(sample_audio_path)
    
    def test_run_with_different_models(self, audio_agent, sample_audio_path):
        """Test run method with different model parameters."""
        with patch.object(audio_agent, 'process_audio') as mock_process:
            mock_process.return_value = {
                'status': 200,
                'result': 'Test transcription',
                'model': 'audio_agent',
                'keywords': ['audio', 'transcription']
            }
            
            models = ["edumentor_agent", "vedas_agent", "wellness_agent"]
            
            for model in models:
                result = audio_agent.run(sample_audio_path, "", model, "audio")
                assert 'status' in result
                # The agent type should remain 'audio_agent' regardless of model parameter
                assert result['agent'] == 'audio_agent'
        
        # Clean up
        os.unlink(sample_audio_path)
    
    @patch('agents.audio_agent.librosa.load')
    def test_process_audio_long_file(self, mock_librosa, audio_agent, sample_audio_path):
        """Test processing very long audio file."""
        # Simulate long audio file (10 minutes at 16kHz = 9.6M samples)
        long_audio = [0.1] * (16000 * 600)  # 10 minutes
        mock_librosa.return_value = (long_audio, 16000)
        
        audio_agent.processor = Mock()
        audio_agent.model = Mock()
        audio_agent.processor.return_value = {"input_values": "fake_tensor"}
        
        mock_model_output = Mock()
        mock_model_output.logits = "fake_logits"
        audio_agent.model.return_value = mock_model_output
        audio_agent.processor.decode.return_value = "Long transcription result"
        
        task_id = str(uuid.uuid4())
        result = audio_agent.process_audio(sample_audio_path, task_id)
        
        assert result['status'] == 200
        assert result['result'] == "Long transcription result"
        
        # Clean up
        os.unlink(sample_audio_path)


if __name__ == "__main__":
    pytest.main([__file__])
