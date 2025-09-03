import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from agents.text_agent import TextAgent


class TestTextAgent:
    """Test suite for TextAgent."""
    
    @pytest.fixture
    def text_agent(self):
        """Create a TextAgent instance for testing."""
        return TextAgent()
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response."""
        mock_response = Mock()
        mock_response.content = "This is a test summary of the input text."
        return mock_response
    
    def test_text_agent_initialization(self, text_agent):
        """Test TextAgent initialization."""
        assert text_agent is not None
        assert hasattr(text_agent, 'llm')
        assert hasattr(text_agent, 'model_config')
    
    @patch('agents.text_agent.ChatGroq')
    def test_process_text_success(self, mock_groq, text_agent, mock_llm_response):
        """Test successful text processing."""
        # Setup mock
        text_agent.llm = Mock()
        text_agent.llm.invoke.return_value = mock_llm_response
        
        # Test data
        test_text = "This is a test text that needs to be summarized."
        task_id = str(uuid.uuid4())
        
        # Execute
        result = text_agent.process_text(test_text, task_id)
        
        # Assertions
        assert result['status'] == 200
        assert result['result'] == "This is a test summary of the input text."
        assert result['model'] == 'text_agent'
        assert 'processing_time' in result
        assert 'tokens_used' in result
        assert result['keywords'] == ['text', 'summary']
        assert result['attempts'] == 1
    
    @patch('agents.text_agent.ChatGroq')
    def test_process_text_with_retries(self, mock_groq, text_agent):
        """Test text processing with retries on failure."""
        # Setup mock to fail twice then succeed
        text_agent.llm = Mock()
        text_agent.llm.invoke.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Mock(content="Success after retries")
        ]
        
        test_text = "Test text"
        task_id = str(uuid.uuid4())
        
        # Execute
        with patch('agents.text_agent.time.sleep'):  # Speed up test
            result = text_agent.process_text(test_text, task_id, retries=3)
        
        # Assertions
        assert result['status'] == 200
        assert result['result'] == "Success after retries"
        assert result['attempts'] == 3
        assert text_agent.llm.invoke.call_count == 3
    
    @patch('agents.text_agent.ChatGroq')
    def test_process_text_all_retries_fail(self, mock_groq, text_agent):
        """Test text processing when all retries fail."""
        # Setup mock to always fail
        text_agent.llm = Mock()
        text_agent.llm.invoke.side_effect = Exception("Persistent API Error")
        
        test_text = "Test text"
        task_id = str(uuid.uuid4())
        
        # Execute
        with patch('agents.text_agent.time.sleep'):  # Speed up test
            result = text_agent.process_text(test_text, task_id, retries=2)
        
        # Assertions
        assert result['status'] == 500
        assert 'error' in result
        assert "failed after 2 attempts" in result['error']
        assert result['attempts'] == 2
        assert text_agent.llm.invoke.call_count == 2
    
    @patch('agents.text_agent.replay_buffer')
    @patch('agents.text_agent.get_reward_from_output')
    def test_run_success(self, mock_reward, mock_buffer, text_agent, mock_llm_response):
        """Test successful run method."""
        # Setup mocks
        text_agent.llm = Mock()
        text_agent.llm.invoke.return_value = mock_llm_response
        mock_reward.return_value = 0.85
        
        # Test data
        input_text = "Test input text"
        task_id = str(uuid.uuid4())
        
        # Execute
        result = text_agent.run(input_text, "", "edumentor_agent", "text", task_id)
        
        # Assertions
        assert result['status'] == 200
        assert result['agent'] == 'text_agent'
        assert result['input_type'] == 'text'
        assert result['input_length'] == len(input_text)
        
        # Verify replay buffer was called
        mock_buffer.add_run.assert_called_once()
        mock_reward.assert_called_once()
    
    def test_run_empty_input(self, text_agent):
        """Test run method with empty input."""
        result = text_agent.run("", "", "edumentor_agent", "text")
        
        # Should still process (empty string is valid input)
        assert 'status' in result
        assert result['input_length'] == 0
    
    @patch('agents.text_agent.ChatGroq')
    def test_process_text_long_input(self, mock_groq, text_agent, mock_llm_response):
        """Test processing very long text input."""
        text_agent.llm = Mock()
        text_agent.llm.invoke.return_value = mock_llm_response
        
        # Create long text (10000 characters)
        long_text = "A" * 10000
        task_id = str(uuid.uuid4())
        
        result = text_agent.process_text(long_text, task_id)
        
        assert result['status'] == 200
        assert result['result'] == "This is a test summary of the input text."
        # Verify the prompt was created (LLM was called)
        text_agent.llm.invoke.assert_called_once()
    
    @patch('agents.text_agent.ChatGroq')
    def test_process_text_special_characters(self, mock_groq, text_agent, mock_llm_response):
        """Test processing text with special characters."""
        text_agent.llm = Mock()
        text_agent.llm.invoke.return_value = mock_llm_response
        
        special_text = "Text with émojis 🚀 and spëcial çharacters & symbols @#$%"
        task_id = str(uuid.uuid4())
        
        result = text_agent.process_text(special_text, task_id)
        
        assert result['status'] == 200
        assert result['result'] == "This is a test summary of the input text."
    
    def test_run_with_different_models(self, text_agent, mock_llm_response):
        """Test run method with different model parameters."""
        text_agent.llm = Mock()
        text_agent.llm.invoke.return_value = mock_llm_response
        
        models = ["edumentor_agent", "vedas_agent", "wellness_agent"]
        
        for model in models:
            result = text_agent.run("Test input", "", model, "text")
            assert 'status' in result
            # The agent type should remain 'text_agent' regardless of model parameter
            assert result['agent'] == 'text_agent'
    
    @patch('agents.text_agent.logger')
    def test_logging_behavior(self, mock_logger, text_agent, mock_llm_response):
        """Test that appropriate logging occurs."""
        text_agent.llm = Mock()
        text_agent.llm.invoke.return_value = mock_llm_response
        
        task_id = str(uuid.uuid4())
        text_agent.run("Test input", "", "edumentor_agent", "text", task_id)
        
        # Verify logging calls were made
        assert mock_logger.info.called
        assert mock_logger.debug.called
    
    def test_token_counting_approximation(self, text_agent, mock_llm_response):
        """Test token counting approximation."""
        text_agent.llm = Mock()
        text_agent.llm.invoke.return_value = mock_llm_response
        
        test_text = "This is a test with exactly ten words in it."
        task_id = str(uuid.uuid4())
        
        result = text_agent.process_text(test_text, task_id)
        
        # Should count input words + output words
        expected_tokens = len(test_text.split()) + len(mock_llm_response.content.split())
        assert result['tokens_used'] == expected_tokens


if __name__ == "__main__":
    pytest.main([__file__])
