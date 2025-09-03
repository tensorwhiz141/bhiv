import pytest
import uuid
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from agents.image_agent import ImageAgent


class TestImageAgent:
    """Test suite for ImageAgent."""
    
    @pytest.fixture
    def image_agent(self):
        """Create an ImageAgent instance for testing."""
        return ImageAgent()
    
    @pytest.fixture
    def mock_processor_output(self):
        """Mock image processor output."""
        return [{"generated_text": "A test image showing a cat sitting on a table."}]
    
    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary image file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            # Write some fake image data
            temp_file.write(b'\xff\xd8\xff\xe0\x00\x10JFIF')  # JPEG header
            temp_path = temp_file.name
        return temp_path
    
    def teardown_method(self, method):
        """Clean up temporary files after each test."""
        # Clean up any temporary files that might exist
        import glob
        for temp_file in glob.glob('/tmp/tmp*.jpg'):
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_image_agent_initialization(self, image_agent):
        """Test ImageAgent initialization."""
        assert image_agent is not None
        assert hasattr(image_agent, 'processor')
        assert hasattr(image_agent, 'model')
    
    @patch('agents.image_agent.BlipProcessor')
    @patch('agents.image_agent.BlipForConditionalGeneration')
    def test_initialization_with_mocks(self, mock_model_class, mock_processor_class):
        """Test initialization with mocked dependencies."""
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        agent = ImageAgent()
        
        assert agent.processor == mock_processor
        assert agent.model == mock_model
        mock_processor_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
    
    @patch('agents.image_agent.Image.open')
    def test_process_image_success(self, mock_image_open, image_agent, mock_processor_output, sample_image_path):
        """Test successful image processing."""
        # Setup mocks
        mock_image = Mock()
        mock_image_open.return_value = mock_image
        
        image_agent.processor = Mock()
        image_agent.model = Mock()
        
        # Mock processor and model outputs
        mock_inputs = {"pixel_values": "fake_tensor"}
        image_agent.processor.return_value = mock_inputs
        
        mock_output_ids = "fake_output_ids"
        image_agent.model.generate.return_value = mock_output_ids
        
        image_agent.processor.decode.return_value = "A test image showing a cat sitting on a table."
        
        task_id = str(uuid.uuid4())
        
        # Execute
        result = image_agent.process_image(sample_image_path, task_id)
        
        # Assertions
        assert result['status'] == 200
        assert result['result'] == "A test image showing a cat sitting on a table."
        assert result['model'] == 'image_agent'
        assert 'processing_time' in result
        assert result['keywords'] == ['image', 'caption']
        
        # Verify method calls
        mock_image_open.assert_called_once_with(sample_image_path)
        image_agent.processor.assert_called_once()
        image_agent.model.generate.assert_called_once()
        
        # Clean up
        os.unlink(sample_image_path)
    
    def test_process_image_file_not_found(self, image_agent):
        """Test image processing with non-existent file."""
        task_id = str(uuid.uuid4())
        result = image_agent.process_image("nonexistent.jpg", task_id)
        
        assert result['status'] == 500
        assert 'error' in result
        assert "Image processing failed" in result['error']
    
    @patch('agents.image_agent.Image.open')
    def test_process_image_with_exception(self, mock_image_open, image_agent, sample_image_path):
        """Test image processing when an exception occurs."""
        # Setup mock to raise exception
        mock_image_open.side_effect = Exception("Image loading error")
        
        task_id = str(uuid.uuid4())
        result = image_agent.process_image(sample_image_path, task_id)
        
        assert result['status'] == 500
        assert 'error' in result
        assert "Image processing failed" in result['error']
        
        # Clean up
        os.unlink(sample_image_path)
    
    @patch('agents.image_agent.replay_buffer')
    @patch('agents.image_agent.get_reward_from_output')
    @patch.object(ImageAgent, 'process_image')
    def test_run_success(self, mock_process, mock_reward, mock_buffer, image_agent, sample_image_path):
        """Test successful run method."""
        # Setup mocks
        mock_process_result = {
            'status': 200,
            'result': 'A beautiful landscape image',
            'model': 'image_agent',
            'keywords': ['image', 'caption'],
            'processing_time': 1.5
        }
        mock_process.return_value = mock_process_result
        mock_reward.return_value = 0.88
        
        task_id = str(uuid.uuid4())
        
        # Execute
        result = image_agent.run(sample_image_path, "", "edumentor_agent", "image", task_id)
        
        # Assertions
        assert result['status'] == 200
        assert result['agent'] == 'image_agent'
        assert result['input_type'] == 'image'
        assert result['file_path'] == sample_image_path
        
        # Verify methods were called
        mock_process.assert_called_once_with(sample_image_path, task_id)
        mock_buffer.add_run.assert_called_once()
        mock_reward.assert_called_once()
        
        # Clean up
        os.unlink(sample_image_path)
    
    @patch('agents.image_agent.replay_buffer')
    @patch('agents.image_agent.get_reward_from_output')
    @patch.object(ImageAgent, 'process_image')
    def test_run_processing_failure(self, mock_process, mock_reward, mock_buffer, image_agent):
        """Test run method when image processing fails."""
        # Setup mock to return error
        mock_process_result = {
            'status': 500,
            'error': 'Image processing failed: File not found',
            'keywords': []
        }
        mock_process.return_value = mock_process_result
        mock_reward.return_value = 0.0
        
        image_path = "nonexistent.jpg"
        task_id = str(uuid.uuid4())
        
        # Execute
        result = image_agent.run(image_path, "", "edumentor_agent", "image", task_id)
        
        # Assertions
        assert result['status'] == 500
        assert 'error' in result
        assert result['agent'] == 'image_agent'
        assert result['file_path'] == image_path
        
        # Verify replay buffer was still called (for error tracking)
        mock_buffer.add_run.assert_called_once()
    
    @patch('agents.image_agent.Image.open')
    def test_process_image_different_formats(self, mock_image_open, image_agent):
        """Test processing different image formats."""
        # Setup mock
        mock_image = Mock()
        mock_image_open.return_value = mock_image
        
        image_agent.processor = Mock()
        image_agent.model = Mock()
        image_agent.processor.return_value = {"pixel_values": "fake_tensor"}
        image_agent.model.generate.return_value = "fake_output_ids"
        image_agent.processor.decode.return_value = "Test caption"
        
        # Test different image formats
        formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as temp_file:
                temp_file.write(b'fake_image_data')
                temp_path = temp_file.name
            
            try:
                task_id = str(uuid.uuid4())
                result = image_agent.process_image(temp_path, task_id)
                assert result['status'] == 200
                assert result['result'] == "Test caption"
            finally:
                os.unlink(temp_path)
    
    @patch('agents.image_agent.logger')
    @patch.object(ImageAgent, 'process_image')
    def test_logging_behavior(self, mock_process, mock_logger, image_agent, sample_image_path):
        """Test that appropriate logging occurs."""
        mock_process.return_value = {
            'status': 200,
            'result': 'Test caption',
            'model': 'image_agent',
            'keywords': ['image', 'caption']
        }
        
        task_id = str(uuid.uuid4())
        image_agent.run(sample_image_path, "", "edumentor_agent", "image", task_id)
        
        # Verify logging calls were made
        assert mock_logger.info.called
        assert mock_logger.debug.called
        
        # Clean up
        os.unlink(sample_image_path)
    
    def test_run_with_different_models(self, image_agent, sample_image_path):
        """Test run method with different model parameters."""
        with patch.object(image_agent, 'process_image') as mock_process:
            mock_process.return_value = {
                'status': 200,
                'result': 'Test caption',
                'model': 'image_agent',
                'keywords': ['image', 'caption']
            }
            
            models = ["edumentor_agent", "vedas_agent", "wellness_agent"]
            
            for model in models:
                result = image_agent.run(sample_image_path, "", model, "image")
                assert 'status' in result
                # The agent type should remain 'image_agent' regardless of model parameter
                assert result['agent'] == 'image_agent'
        
        # Clean up
        os.unlink(sample_image_path)
    
    @patch('agents.image_agent.Image.open')
    def test_process_image_empty_caption(self, mock_image_open, image_agent, sample_image_path):
        """Test handling of empty or None captions."""
        # Setup mocks
        mock_image = Mock()
        mock_image_open.return_value = mock_image
        
        image_agent.processor = Mock()
        image_agent.model = Mock()
        image_agent.processor.return_value = {"pixel_values": "fake_tensor"}
        image_agent.model.generate.return_value = "fake_output_ids"
        image_agent.processor.decode.return_value = ""  # Empty caption
        
        task_id = str(uuid.uuid4())
        result = image_agent.process_image(sample_image_path, task_id)
        
        # Should still return success but with empty result
        assert result['status'] == 200
        assert result['result'] == ""
        
        # Clean up
        os.unlink(sample_image_path)


if __name__ == "__main__":
    pytest.main([__file__])
