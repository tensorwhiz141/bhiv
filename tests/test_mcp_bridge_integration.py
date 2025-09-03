import pytest
import asyncio
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from mcp_bridge import app
import json


class TestMCPBridgeIntegration:
    """Integration tests for the MCP Bridge API."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the MCP bridge."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_task_payload(self):
        """Sample task payload for testing."""
        return {
            "agent": "edumentor_agent",
            "input": "Test input text for processing",
            "pdf_path": "",
            "input_type": "text",
            "retries": 3,
            "fallback_model": "edumentor_agent"
        }
    
    @pytest.fixture
    def sample_files(self):
        """Create sample files for testing."""
        files = {}
        
        # Create a sample text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a sample text file for testing.")
            files['text'] = f.name
        
        # Create a sample PDF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write("Sample PDF content for testing.")
            files['pdf'] = f.name
        
        return files
    
    def teardown_method(self, method):
        """Clean up temporary files after each test."""
        import glob
        for pattern in ['/tmp/tmp*.txt', '/tmp/tmp*.pdf', '/tmp/tmp*.jpg', '/tmp/tmp*.wav']:
            for temp_file in glob.glob(pattern):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    @patch('mcp_bridge.mongo_collection')
    @patch('agents.text_agent.TextAgent.run')
    def test_handle_task_text_success(self, mock_agent_run, mock_mongo, client, sample_task_payload):
        """Test successful text task handling."""
        # Mock agent response
        mock_agent_run.return_value = {
            "status": 200,
            "result": "This is a test summary",
            "model": "text_agent",
            "confidence": 0.9,
            "keywords": ["test", "summary"]
        }
        
        # Mock MongoDB insert
        mock_mongo.insert_one = AsyncMock(return_value=Mock(inserted_id="test_id"))
        
        response = client.post("/handle_task", json=sample_task_payload)
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == 200
        assert "task_id" in result
        assert "agent_output" in result
        assert result["agent_output"]["result"] == "This is a test summary"
    
    @patch('mcp_bridge.mongo_collection')
    @patch('agents.archive_agent.ArchiveAgent.run')
    def test_handle_task_pdf_success(self, mock_agent_run, mock_mongo, client, sample_files):
        """Test successful PDF task handling."""
        # Mock agent response
        mock_agent_run.return_value = {
            "status": 200,
            "result": "PDF summary content",
            "model": "archive_agent",
            "confidence": 0.85,
            "keywords": ["pdf", "summary"]
        }
        
        # Mock MongoDB insert
        mock_mongo.insert_one = AsyncMock(return_value=Mock(inserted_id="test_pdf_id"))
        
        payload = {
            "agent": "edumentor_agent",
            "input": "Process this PDF",
            "pdf_path": sample_files['pdf'],
            "input_type": "pdf",
            "retries": 3,
            "fallback_model": "edumentor_agent"
        }
        
        response = client.post("/handle_task", json=payload)
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == 200
        assert result["agent_output"]["result"] == "PDF summary content"
    
    @patch('mcp_bridge.mongo_collection')
    @patch('agents.image_agent.ImageAgent.run')
    def test_handle_task_image_success(self, mock_agent_run, mock_mongo, client):
        """Test successful image task handling."""
        # Mock agent response
        mock_agent_run.return_value = {
            "status": 200,
            "result": "Image shows a cat sitting on a table",
            "model": "image_agent",
            "confidence": 0.88,
            "keywords": ["image", "caption"]
        }
        
        # Mock MongoDB insert
        mock_mongo.insert_one = AsyncMock(return_value=Mock(inserted_id="test_image_id"))
        
        payload = {
            "agent": "edumentor_agent",
            "input": "test_image.jpg",
            "pdf_path": "test_image.jpg",
            "input_type": "image",
            "retries": 3,
            "fallback_model": "edumentor_agent"
        }
        
        response = client.post("/handle_task", json=payload)
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == 200
        assert result["agent_output"]["result"] == "Image shows a cat sitting on a table"
    
    @patch('mcp_bridge.mongo_collection')
    @patch('agents.audio_agent.AudioAgent.run')
    def test_handle_task_audio_success(self, mock_agent_run, mock_mongo, client):
        """Test successful audio task handling."""
        # Mock agent response
        mock_agent_run.return_value = {
            "status": 200,
            "result": "This is a transcription of the audio file",
            "model": "audio_agent",
            "confidence": 0.92,
            "keywords": ["audio", "transcription"]
        }
        
        # Mock MongoDB insert
        mock_mongo.insert_one = AsyncMock(return_value=Mock(inserted_id="test_audio_id"))
        
        payload = {
            "agent": "edumentor_agent",
            "input": "test_audio.wav",
            "pdf_path": "test_audio.wav",
            "input_type": "audio",
            "retries": 3,
            "fallback_model": "edumentor_agent"
        }
        
        response = client.post("/handle_task", json=payload)
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == 200
        assert result["agent_output"]["result"] == "This is a transcription of the audio file"
    
    def test_handle_task_invalid_agent(self, client, sample_task_payload):
        """Test handling task with invalid agent."""
        payload = sample_task_payload.copy()
        payload["agent"] = "nonexistent_agent"
        
        response = client.post("/handle_task", json=payload)
        
        assert response.status_code == 200  # API returns 200 but with error in response
        result = response.json()
        assert result["status"] == 404
        assert "not found" in result["agent_output"]["error"].lower()
    
    @patch('mcp_bridge.mongo_collection')
    @patch('agents.text_agent.TextAgent.run')
    def test_handle_task_agent_failure(self, mock_agent_run, mock_mongo, client, sample_task_payload):
        """Test handling task when agent fails."""
        # Mock agent failure
        mock_agent_run.side_effect = Exception("Agent processing failed")
        
        # Mock MongoDB insert
        mock_mongo.insert_one = AsyncMock(return_value=Mock(inserted_id="test_error_id"))
        
        response = client.post("/handle_task", json=sample_task_payload)
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == 500
        assert "error" in result["agent_output"]
    
    def test_handle_task_missing_fields(self, client):
        """Test handling task with missing required fields."""
        incomplete_payload = {
            "agent": "edumentor_agent"
            # Missing required fields
        }
        
        response = client.post("/handle_task", json=incomplete_payload)
        
        # Should return validation error
        assert response.status_code == 422
    
    @patch('mcp_bridge.requests.post')
    @patch('mcp_bridge.mongo_collection')
    def test_handle_task_external_agent(self, mock_mongo, mock_requests, client):
        """Test handling task with external HTTP agent."""
        # Mock agent registry to return HTTP agent config
        with patch('mcp_bridge.agent_registry') as mock_registry:
            mock_registry.agent_configs = {
                "external_agent": {
                    "connection_type": "http_api",
                    "endpoint": "http://external-agent.com/process",
                    "headers": {"Authorization": "Bearer test-token"}
                }
            }
            
            # Mock external API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "result": "External agent result",
                "confidence": 0.87
            }
            mock_requests.return_value = mock_response
            
            # Mock MongoDB insert
            mock_mongo.insert_one = AsyncMock(return_value=Mock(inserted_id="external_id"))
            
            payload = {
                "agent": "external_agent",
                "input": "Test external processing",
                "pdf_path": "",
                "input_type": "text",
                "retries": 3,
                "fallback_model": "edumentor_agent"
            }
            
            response = client.post("/handle_task", json=payload)
            
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == 200
            assert result["agent_output"]["result"] == "External agent result"
    
    @patch('mcp_bridge.mongo_client')
    def test_health_check_healthy(self, mock_mongo_client, client):
        """Test health check when all services are healthy."""
        # Mock MongoDB ping
        mock_mongo_client.admin.command = AsyncMock(return_value={"ok": 1})
        
        response = client.get("/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert "services" in result
        assert "metrics" in result
        assert "uptime_seconds" in result
    
    @patch('mcp_bridge.mongo_client')
    def test_health_check_mongodb_down(self, mock_mongo_client, client):
        """Test health check when MongoDB is down."""
        # Mock MongoDB connection failure
        mock_mongo_client.admin.command = AsyncMock(side_effect=Exception("Connection failed"))
        
        response = client.get("/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "degraded"
        assert "unhealthy" in result["services"]["mongodb"]
    
    def test_get_config(self, client):
        """Test getting current configuration."""
        response = client.get("/config")
        
        assert response.status_code == 200
        result = response.json()
        assert "agents" in result
        assert "mongodb" in result
        assert "timestamp" in result
    
    @patch('mcp_bridge.importlib.reload')
    def test_reload_config_success(self, mock_reload, client):
        """Test successful configuration reload."""
        # Mock successful reload
        mock_reload.return_value = None
        
        response = client.post("/config/reload")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "Configuration reloaded" in result["message"]
    
    @patch('mcp_bridge.importlib.reload')
    def test_reload_config_failure(self, mock_reload, client):
        """Test configuration reload failure."""
        # Mock reload failure
        mock_reload.side_effect = Exception("Reload failed")
        
        response = client.post("/config/reload")
        
        assert response.status_code == 500
        result = response.json()
        assert "Failed to reload config" in result["detail"]
    
    @patch('mcp_bridge.mongo_db')
    @patch('mcp_bridge.mongo_collection')
    def test_get_metrics(self, mock_collection, mock_db, client):
        """Test getting system metrics."""
        # Mock database stats
        mock_db.command = AsyncMock(return_value={
            "dataSize": 1024 * 1024,  # 1MB
            "storageSize": 2 * 1024 * 1024  # 2MB
        })
        
        # Mock collection stats
        mock_collection.estimated_document_count = AsyncMock(return_value=100)
        mock_collection.find.return_value.sort.return_value.limit.return_value.to_list = AsyncMock(
            return_value=[
                {"agent": "text_agent", "output": {"processing_time": 1.5}},
                {"agent": "image_agent", "output": {"processing_time": 2.3}}
            ]
        )
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert "database" in result
        assert "performance" in result
        assert "system" in result
        assert result["database"]["total_documents"] == 100
    
    @patch('mcp_bridge.asyncio.gather')
    def test_handle_multi_task(self, mock_gather, client):
        """Test multi-task handling endpoint."""
        # Mock successful multi-task processing
        mock_results = [
            {
                "task_id": "task1",
                "agent_output": {"result": "Result 1", "status": 200},
                "status": 200
            },
            {
                "task_id": "task2", 
                "agent_output": {"result": "Result 2", "status": 200},
                "status": 200
            }
        ]
        mock_gather.return_value = mock_results
        
        payload = {
            "files": [
                {"path": "file1.txt", "type": "text", "data": "Content 1"},
                {"path": "file2.txt", "type": "text", "data": "Content 2"}
            ],
            "agent": "edumentor_agent",
            "task_type": "summarize"
        }
        
        response = client.post("/handle_multi_task", json=payload)
        
        assert response.status_code == 200
        result = response.json()
        assert result["total_files"] == 2
        assert result["successful_files"] == 2
        assert result["failed_files"] == 0
        assert len(result["results"]) == 2
    
    def test_handle_multi_task_no_files(self, client):
        """Test multi-task endpoint with no files."""
        payload = {
            "files": [],
            "agent": "edumentor_agent",
            "task_type": "summarize"
        }
        
        response = client.post("/handle_multi_task", json=payload)
        
        assert response.status_code == 400
        assert "No files provided" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])
