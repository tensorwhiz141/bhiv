import pytest
import asyncio
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from integration.web_interface import app
import json
import base64


class TestWebInterfaceIntegration:
    """Integration tests for the web interface."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the web interface."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create basic auth headers for testing."""
        credentials = base64.b64encode(b"admin:secret").decode("ascii")
        return {"Authorization": f"Basic {credentials}"}
    
    @pytest.fixture
    def sample_files(self):
        """Create sample files for testing."""
        files = {}
        
        # Create a sample text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a sample text file for testing.")
            files['text'] = f.name
        
        # Create a sample "PDF" file (just text with .pdf extension)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write("Sample PDF content for testing.")
            files['pdf'] = f.name
        
        # Create a sample "image" file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jpg', delete=False) as f:
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF')  # JPEG header
            files['image'] = f.name
        
        return files
    
    def teardown_method(self, method):
        """Clean up temporary files after each test."""
        import glob
        for pattern in ['/tmp/tmp*.txt', '/tmp/tmp*.pdf', '/tmp/tmp*.jpg']:
            for temp_file in glob.glob(pattern):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def test_home_page_requires_auth(self, client):
        """Test that home page requires authentication."""
        response = client.get("/")
        assert response.status_code == 401
    
    def test_home_page_with_auth(self, client, auth_headers):
        """Test home page with authentication."""
        response = client.get("/", headers=auth_headers)
        assert response.status_code == 200
        assert "BHIV Core" in response.text
        assert "file upload" in response.text.lower()
    
    def test_dashboard_requires_auth(self, client):
        """Test that dashboard requires authentication."""
        response = client.get("/dashboard")
        assert response.status_code == 401
    
    @patch('integration.web_interface.mongo_db')
    def test_dashboard_with_auth(self, mock_db, client, auth_headers):
        """Test dashboard with authentication."""
        # Mock database responses
        mock_db.task_logs.find.return_value.sort.return_value.limit.return_value.to_list = AsyncMock(return_value=[])
        mock_db.nlo_collection.aggregate.return_value.to_list = AsyncMock(return_value=[])
        
        response = client.get("/dashboard", headers=auth_headers)
        assert response.status_code == 200
        assert "Dashboard" in response.text
    
    @patch('integration.web_interface.requests.post')
    def test_upload_single_file(self, mock_post, client, auth_headers, sample_files):
        """Test uploading a single file."""
        # Mock the MCP bridge response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "task_id": "test-task-123",
            "agent_output": {
                "result": "Test summary of uploaded file",
                "confidence": 0.9,
                "keywords": ["test", "upload"]
            }
        }
        mock_post.return_value = mock_response
        
        # Upload file
        with open(sample_files['text'], 'rb') as f:
            response = client.post(
                "/upload",
                headers=auth_headers,
                files={"files": ("test.txt", f, "text/plain")},
                data={
                    "agent": "edumentor_agent",
                    "task_description": "Test upload"
                }
            )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert "task_id" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["filename"] == "test.txt"
    
    @patch('integration.web_interface.requests.post')
    def test_upload_multiple_files(self, mock_post, client, auth_headers, sample_files):
        """Test uploading multiple files."""
        # Mock the MCP bridge response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "task_id": "test-task-456",
            "agent_output": {
                "result": "Test summary",
                "confidence": 0.8,
                "keywords": ["test"]
            }
        }
        mock_post.return_value = mock_response
        
        # Upload multiple files
        files_to_upload = [
            ("files", ("test1.txt", open(sample_files['text'], 'rb'), "text/plain")),
            ("files", ("test2.pdf", open(sample_files['pdf'], 'rb'), "application/pdf"))
        ]
        
        try:
            response = client.post(
                "/upload",
                headers=auth_headers,
                files=files_to_upload,
                data={
                    "agent": "edumentor_agent",
                    "task_description": "Test multiple upload"
                }
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "completed"
            assert len(result["results"]) == 2
        finally:
            # Close file handles
            for _, (_, file_handle, _) in files_to_upload:
                file_handle.close()
    
    def test_upload_no_files(self, client, auth_headers):
        """Test upload endpoint with no files."""
        response = client.post(
            "/upload",
            headers=auth_headers,
            data={
                "agent": "edumentor_agent",
                "task_description": "Test no files"
            }
        )
        
        # Should return an error
        assert response.status_code == 422  # FastAPI validation error
    
    @patch('integration.web_interface.requests.post')
    def test_upload_with_api_error(self, mock_post, client, auth_headers, sample_files):
        """Test upload when MCP bridge returns an error."""
        # Mock API error
        mock_post.side_effect = Exception("API connection failed")
        
        with open(sample_files['text'], 'rb') as f:
            response = client.post(
                "/upload",
                headers=auth_headers,
                files={"files": ("test.txt", f, "text/plain")},
                data={
                    "agent": "edumentor_agent",
                    "task_description": "Test error"
                }
            )
        
        assert response.status_code == 500
        result = response.json()
        assert result["status"] == "error"
        assert "error" in result
    
    def test_task_status_not_found(self, client, auth_headers):
        """Test task status endpoint with non-existent task."""
        response = client.get("/task_status/nonexistent-task", headers=auth_headers)
        assert response.status_code == 404
    
    @patch('integration.web_interface.active_tasks')
    def test_task_status_found(self, mock_tasks, client, auth_headers):
        """Test task status endpoint with existing task."""
        from datetime import datetime
        
        # Mock active task
        task_id = "test-task-789"
        mock_tasks.__contains__.return_value = True
        mock_tasks.__getitem__.return_value = {
            "status": "completed",
            "files": ["test.txt"],
            "agent": "edumentor_agent",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "results": [{"filename": "test.txt", "result": {"status": 200}}]
        }
        
        response = client.get(f"/task_status/{task_id}", headers=auth_headers)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert result["files"] == ["test.txt"]
    
    @patch('integration.web_interface.get_nlos_by_task_id')
    def test_download_nlo_json(self, mock_get_nlo, client, auth_headers):
        """Test downloading NLO as JSON."""
        # Mock NLO data
        mock_nlo = {
            "task_id": "test-task-download",
            "summary": "Test NLO summary",
            "confidence": 0.9,
            "tags": ["test", "download"]
        }
        mock_get_nlo.return_value = mock_nlo
        
        response = client.get("/download_nlo/test-task-download?format=json", headers=auth_headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    @patch('integration.web_interface.get_nlos_by_task_id')
    def test_download_nlo_not_found(self, mock_get_nlo, client, auth_headers):
        """Test downloading non-existent NLO."""
        mock_get_nlo.return_value = None
        
        response = client.get("/download_nlo/nonexistent?format=json", headers=auth_headers)
        assert response.status_code == 404
    
    def test_download_nlo_invalid_format(self, client, auth_headers):
        """Test downloading NLO with invalid format."""
        response = client.get("/download_nlo/test-task?format=invalid", headers=auth_headers)
        assert response.status_code == 400
    
    @patch('integration.web_interface.get_nlos_by_subject')
    def test_api_nlos_by_subject(self, mock_get_nlos, client, auth_headers):
        """Test API endpoint for getting NLOs by subject."""
        # Mock NLO data
        mock_nlos = [
            {"task_id": "1", "subject_tag": "test", "summary": "Test 1"},
            {"task_id": "2", "subject_tag": "test", "summary": "Test 2"}
        ]
        mock_get_nlos.return_value = mock_nlos
        
        response = client.get("/api/nlos?subject=test&limit=5", headers=auth_headers)
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2
        assert result[0]["subject_tag"] == "test"
    
    @patch('integration.web_interface.mongo_db')
    def test_api_nlos_recent(self, mock_db, client, auth_headers):
        """Test API endpoint for getting recent NLOs."""
        # Mock database response
        mock_cursor = Mock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"task_id": "1", "summary": "Recent 1"},
            {"task_id": "2", "summary": "Recent 2"}
        ])
        mock_db.nlo_collection.find.return_value.sort.return_value.limit.return_value = mock_cursor
        
        response = client.get("/api/nlos?limit=2", headers=auth_headers)
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2
    
    @patch('integration.web_interface.mongo_client')
    @patch('integration.web_interface.requests.get')
    def test_health_check(self, mock_requests, mock_mongo_client, client):
        """Test health check endpoint."""
        # Mock MongoDB ping
        mock_mongo_client.admin.command = AsyncMock(return_value={"ok": 1})
        
        # Mock MCP bridge health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response
        
        response = client.get("/health")
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert "services" in result
        assert "mongodb" in result["services"]
        assert "mcp_bridge" in result["services"]
    
    @patch('integration.web_interface.mongo_client')
    def test_health_check_mongodb_down(self, mock_mongo_client, client):
        """Test health check when MongoDB is down."""
        # Mock MongoDB connection failure
        mock_mongo_client.admin.command = AsyncMock(side_effect=Exception("Connection failed"))
        
        response = client.get("/health")
        assert response.status_code == 503
        result = response.json()
        assert result["status"] == "unhealthy"
    
    def test_authentication_invalid_credentials(self, client):
        """Test authentication with invalid credentials."""
        invalid_credentials = base64.b64encode(b"invalid:wrong").decode("ascii")
        headers = {"Authorization": f"Basic {invalid_credentials}"}
        
        response = client.get("/", headers=headers)
        assert response.status_code == 401
    
    def test_authentication_missing_credentials(self, client):
        """Test authentication with missing credentials."""
        response = client.get("/")
        assert response.status_code == 401
    
    @patch('integration.web_interface.requests.post')
    def test_upload_different_agents(self, mock_post, client, auth_headers, sample_files):
        """Test upload with different agent types."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "task_id": "test-agent-task",
            "agent_output": {"result": "Agent-specific result", "confidence": 0.85}
        }
        mock_post.return_value = mock_response
        
        agents = ["edumentor_agent", "vedas_agent", "wellness_agent"]
        
        for agent in agents:
            with open(sample_files['text'], 'rb') as f:
                response = client.post(
                    "/upload",
                    headers=auth_headers,
                    files={"files": ("test.txt", f, "text/plain")},
                    data={
                        "agent": agent,
                        "task_description": f"Test with {agent}"
                    }
                )
            
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "completed"


    def test_concurrent_uploads(self, client, auth_headers, sample_files):
        """Test handling of concurrent file uploads."""
        import threading
        import time

        results = []

        def upload_file(file_path, filename):
            with patch('integration.web_interface.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {
                    "task_id": f"concurrent-{filename}",
                    "agent_output": {"result": f"Result for {filename}", "confidence": 0.8}
                }
                mock_post.return_value = mock_response

                with open(file_path, 'rb') as f:
                    response = client.post(
                        "/upload",
                        headers=auth_headers,
                        files={"files": (filename, f, "text/plain")},
                        data={"agent": "edumentor_agent", "task_description": "Concurrent test"}
                    )
                results.append(response.status_code)

        # Start multiple upload threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=upload_file, args=(sample_files['text'], f"test{i}.txt"))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All uploads should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__])
