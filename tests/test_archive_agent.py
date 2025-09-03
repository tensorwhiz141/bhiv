import pytest
import uuid
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from agents.archive_agent import ArchiveAgent


class TestArchiveAgent:
    """Test suite for ArchiveAgent."""
    
    @pytest.fixture
    def archive_agent(self):
        """Create an ArchiveAgent instance for testing."""
        return ArchiveAgent()
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response."""
        mock_response = Mock()
        mock_response.content = "This is a test summary of the PDF content."
        return mock_response
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF text content."""
        return "This is sample PDF content that needs to be processed and summarized."
    
    def test_archive_agent_initialization(self, archive_agent):
        """Test ArchiveAgent initialization."""
        assert archive_agent is not None
        assert hasattr(archive_agent, 'llm')
        assert hasattr(archive_agent, 'model_config')
    
    @patch('agents.archive_agent.PyPDF2.PdfReader')
    def test_extract_pdf_text_success(self, mock_pdf_reader, archive_agent):
        """Test successful PDF text extraction."""
        # Setup mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF text content"
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page, mock_page]  # 2 pages
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"fake pdf content")
            temp_path = temp_file.name
        
        try:
            # Execute
            result = archive_agent.extract_pdf_text(temp_path)
            
            # Assertions
            assert result == "Sample PDF text content\nSample PDF text content"
            mock_pdf_reader.assert_called_once()
        finally:
            os.unlink(temp_path)
    
    @patch('agents.archive_agent.PyPDF2.PdfReader')
    def test_extract_pdf_text_file_not_found(self, mock_pdf_reader, archive_agent):
        """Test PDF extraction with non-existent file."""
        result = archive_agent.extract_pdf_text("nonexistent.pdf")
        assert result == ""
        mock_pdf_reader.assert_not_called()
    
    @patch('agents.archive_agent.PyPDF2.PdfReader')
    def test_extract_pdf_text_exception(self, mock_pdf_reader, archive_agent):
        """Test PDF extraction with PyPDF2 exception."""
        mock_pdf_reader.side_effect = Exception("PDF parsing error")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"fake pdf content")
            temp_path = temp_file.name
        
        try:
            result = archive_agent.extract_pdf_text(temp_path)
            assert result == ""
        finally:
            os.unlink(temp_path)
    
    @patch('agents.archive_agent.ChatGroq')
    def test_process_pdf_success(self, mock_groq, archive_agent, mock_llm_response, sample_pdf_content):
        """Test successful PDF processing."""
        # Setup mock
        archive_agent.llm = Mock()
        archive_agent.llm.invoke.return_value = mock_llm_response
        
        task_id = str(uuid.uuid4())
        
        # Execute
        result = archive_agent.process_pdf(sample_pdf_content, task_id)
        
        # Assertions
        assert result['status'] == 200
        assert result['result'] == "This is a test summary of the PDF content."
        assert result['model'] == 'archive_agent'
        assert 'processing_time' in result
        assert 'text_length' in result
        assert result['text_length'] == len(sample_pdf_content)
        assert result['keywords'] == ['pdf', 'summary']
        assert result['attempts'] == 1
    
    @patch('agents.archive_agent.ChatGroq')
    def test_process_pdf_with_retries(self, mock_groq, archive_agent, sample_pdf_content):
        """Test PDF processing with retries."""
        # Setup mock to fail twice then succeed
        archive_agent.llm = Mock()
        archive_agent.llm.invoke.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Mock(content="Success after retries")
        ]
        
        task_id = str(uuid.uuid4())
        
        # Execute
        with patch('agents.archive_agent.time.sleep'):  # Speed up test
            result = archive_agent.process_pdf(sample_pdf_content, task_id, retries=3)
        
        # Assertions
        assert result['status'] == 200
        assert result['result'] == "Success after retries"
        assert result['attempts'] == 3
        assert archive_agent.llm.invoke.call_count == 3
    
    @patch('agents.archive_agent.ChatGroq')
    def test_process_pdf_all_retries_fail(self, mock_groq, archive_agent, sample_pdf_content):
        """Test PDF processing when all retries fail."""
        # Setup mock to always fail
        archive_agent.llm = Mock()
        archive_agent.llm.invoke.side_effect = Exception("Persistent API Error")
        
        task_id = str(uuid.uuid4())
        
        # Execute
        with patch('agents.archive_agent.time.sleep'):  # Speed up test
            result = archive_agent.process_pdf(sample_pdf_content, task_id, retries=2)
        
        # Assertions
        assert result['status'] == 500
        assert 'error' in result
        assert "failed after 2 attempts" in result['error']
        assert result['attempts'] == 2
    
    def test_process_pdf_long_text_truncation(self, archive_agent, mock_llm_response):
        """Test that long PDF text is properly truncated."""
        archive_agent.llm = Mock()
        archive_agent.llm.invoke.return_value = mock_llm_response
        
        # Create text longer than 2000 characters
        long_text = "A" * 3000
        task_id = str(uuid.uuid4())
        
        result = archive_agent.process_pdf(long_text, task_id)
        
        assert result['status'] == 200
        assert result['text_truncated'] == True
        assert result['text_length'] == 3000
        
        # Verify that the prompt sent to LLM was truncated
        call_args = archive_agent.llm.invoke.call_args[0][0]
        assert len(call_args) < len(long_text) + 100  # Account for prompt text
    
    @patch('agents.archive_agent.replay_buffer')
    @patch('agents.archive_agent.get_reward_from_output')
    @patch.object(ArchiveAgent, 'extract_pdf_text')
    def test_run_success(self, mock_extract, mock_reward, mock_buffer, archive_agent, mock_llm_response, sample_pdf_content):
        """Test successful run method."""
        # Setup mocks
        mock_extract.return_value = sample_pdf_content
        archive_agent.llm = Mock()
        archive_agent.llm.invoke.return_value = mock_llm_response
        mock_reward.return_value = 0.90
        
        # Test data
        pdf_path = "test.pdf"
        task_id = str(uuid.uuid4())
        
        # Execute
        result = archive_agent.run(pdf_path, "", "edumentor_agent", "pdf", task_id)
        
        # Assertions
        assert result['status'] == 200
        assert result['agent'] == 'archive_agent'
        assert result['input_type'] == 'pdf'
        assert result['file_path'] == pdf_path
        assert result['content_length'] == len(sample_pdf_content)
        
        # Verify methods were called
        mock_extract.assert_called_once_with(pdf_path)
        mock_buffer.add_run.assert_called_once()
        mock_reward.assert_called_once()
    
    @patch('agents.archive_agent.replay_buffer')
    @patch('agents.archive_agent.get_reward_from_output')
    @patch.object(ArchiveAgent, 'extract_pdf_text')
    def test_run_extraction_failure(self, mock_extract, mock_reward, mock_buffer, archive_agent):
        """Test run method when PDF extraction fails."""
        # Setup mock to return empty content
        mock_extract.return_value = ""
        mock_reward.return_value = 0.0
        
        pdf_path = "test.pdf"
        task_id = str(uuid.uuid4())
        
        # Execute
        result = archive_agent.run(pdf_path, "", "edumentor_agent", "pdf", task_id)
        
        # Assertions
        assert result['status'] == 400
        assert 'error' in result
        assert "Failed to extract PDF content" in result['error']
        assert result['agent'] == 'archive_agent'
        assert result['file_path'] == pdf_path
        
        # Verify replay buffer was still called (for error tracking)
        mock_buffer.add_run.assert_called_once()
    
    @patch('agents.archive_agent.PyPDF2.PdfReader')
    def test_extract_pdf_empty_pages(self, mock_pdf_reader, archive_agent):
        """Test PDF extraction with empty pages."""
        # Setup mock with empty pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = ""
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "   "  # Only whitespace
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"fake pdf content")
            temp_path = temp_file.name
        
        try:
            result = archive_agent.extract_pdf_text(temp_path)
            assert result == "\n   "  # Should preserve structure even if empty
        finally:
            os.unlink(temp_path)
    
    @patch('agents.archive_agent.logger')
    def test_logging_behavior(self, mock_logger, archive_agent, mock_llm_response, sample_pdf_content):
        """Test that appropriate logging occurs."""
        archive_agent.llm = Mock()
        archive_agent.llm.invoke.return_value = mock_llm_response
        
        with patch.object(archive_agent, 'extract_pdf_text', return_value=sample_pdf_content):
            task_id = str(uuid.uuid4())
            archive_agent.run("test.pdf", "", "edumentor_agent", "pdf", task_id)
        
        # Verify logging calls were made
        assert mock_logger.info.called
        assert mock_logger.debug.called
    
    def test_token_counting_with_pdf(self, archive_agent, mock_llm_response, sample_pdf_content):
        """Test token counting for PDF processing."""
        archive_agent.llm = Mock()
        archive_agent.llm.invoke.return_value = mock_llm_response
        
        task_id = str(uuid.uuid4())
        result = archive_agent.process_pdf(sample_pdf_content, task_id)
        
        # Should count input words + output words
        expected_tokens = len(sample_pdf_content.split()) + len(mock_llm_response.content.split())
        assert result['tokens_used'] == expected_tokens


if __name__ == "__main__":
    pytest.main([__file__])
