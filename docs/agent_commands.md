# BHIV Core Agent Commands Documentation

This document provides comprehensive commands for running all agents in the BHIV Core system via CLI and API.

## Table of Contents
- [Prerequisites](#prerequisites)
- [CLI Commands](#cli-commands)
- [API Commands](#api-commands)
- [Available Agents](#available-agents)
- [File Formats](#file-formats)
- [Testing Scripts](#testing-scripts)

## Prerequisites

### Start Required Services n 
```bash
# 1. Start MCP Bridge Server (required for API calls)
python mcp_bridge.py
# Server runs on http://localhost:8002

# 2. Optional: Start Simple API Server (for direct endpoint access)
python simple_api.py --port 8001
# Server runs on http://localhost:8000
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## CLI Commands

### Basic Syntax
```bash
python cli_runner.py <task> "<input_text>" <agent> [--file <file_path>] [--input-type <type>]
```

### Text Processing
```bash
# Educational queries
python cli_runner.py explain "What is machine learning?" edumentor_agent --input-type text
python cli_runner.py summarize "Explain quantum physics concepts" edumentor_agent --input-type text

# Spiritual/philosophical queries
python cli_runner.py ask "What is dharma according to Vedas?" vedas_agent --input-type text
python cli_runner.py guidance "How to find inner peace?" vedas_agent --input-type text

# Wellness advice
python cli_runner.py advice "How to reduce stress naturally?" wellness_agent --input-type text
python cli_runner.py help "Best practices for mental health" wellness_agent --input-type text

# General text processing
python cli_runner.py process "Analyze this text content" text_agent --input-type text
```

### PDF Processing
```bash
# Summarize PDF documents
python cli_runner.py summarize "Summarize this research paper" edumentor_agent --file document.pdf --input-type pdf
python cli_runner.py analyze "Extract key insights from this PDF" archive_agent --file report.pdf --input-type pdf
python cli_runner.py extract "Get main points from document" edumentor_agent --file book.pdf --input-type pdf

# Educational PDF analysis
python cli_runner.py explain "Explain concepts in this textbook" edumentor_agent --file textbook.pdf --input-type pdf
```

### Image Processing
```bash
# Image description and analysis
python cli_runner.py describe "Describe what you see in this image" edumentor_agent --file photo.jpg --input-type image
python cli_runner.py caption "Generate a caption for this image" image_agent --file picture.png --input-type image
python cli_runner.py analyze "Analyze the contents of this image" edumentor_agent --file diagram.jpeg --input-type image

# Educational image analysis
python cli_runner.py explain "Explain the diagram in this image" edumentor_agent --file chart.png --input-type image
```

### Audio Processing
```bash
# Audio transcription
python cli_runner.py transcribe "Convert this speech to text" edumentor_agent --file recording.wav --input-type audio
python cli_runner.py convert "Transcribe this audio file" audio_agent --file lecture.mp3 --input-type audio
python cli_runner.py speech "Extract text from this audio" edumentor_agent --file interview.flac --input-type audio

# Educational audio processing
python cli_runner.py summarize "Summarize this lecture audio" edumentor_agent --file lecture.wav --input-type audio
```

## API Commands

### JSON API Requests (POST /handle_task)

#### Text Processing
```bash
# Educational content
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "edumentor_agent",
    "input": "Explain artificial intelligence and machine learning",
    "input_type": "text"
  }'

# Spiritual guidance
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "vedas_agent",
    "input": "What is the concept of karma in Hindu philosophy?",
    "input_type": "text"
  }'

# Wellness advice
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "wellness_agent",
    "input": "How to maintain work-life balance?",
    "input_type": "text"
  }'

# General text processing
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "text_agent",
    "input": "Analyze this business proposal text",
    "input_type": "text"
  }'
```

#### PDF Processing
```bash
# Document summarization
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "archive_agent",
    "input": "Summarize this research document",
    "pdf_path": "/path/to/document.pdf",
    "input_type": "pdf"
  }'

# Educational PDF analysis
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "edumentor_agent",
    "input": "Extract key learning points from this textbook",
    "pdf_path": "/path/to/textbook.pdf",
    "input_type": "pdf"
  }'
```

#### Image Processing
```bash
# Image description
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "image_agent",
    "input": "Describe the contents of this image",
    "pdf_path": "/path/to/image.jpg",
    "input_type": "image"
  }'

# Educational image analysis
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "edumentor_agent",
    "input": "Explain the scientific diagram in this image",
    "pdf_path": "/path/to/diagram.png",
    "input_type": "image"
  }'
```

#### Audio Processing
```bash
# Audio transcription
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "audio_agent",
    "input": "Transcribe the speech in this audio file",
    "pdf_path": "/path/to/audio.wav",
    "input_type": "audio"
  }'

# Educational audio processing
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "edumentor_agent",
    "input": "Summarize this educational lecture",
    "pdf_path": "/path/to/lecture.mp3",
    "input_type": "audio"
  }'
```

### File Upload API (POST /handle_task_with_file)

#### PDF Upload and Processing
```bash
curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=edumentor_agent" \
  -F "input=Summarize this uploaded document" \
  -F "input_type=pdf" \
  -F "file=@document.pdf"

curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=archive_agent" \
  -F "input=Extract key insights from this PDF" \
  -F "input_type=pdf" \
  -F "file=@research_paper.pdf"
```

#### Image Upload and Processing
```bash
curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=image_agent" \
  -F "input=Describe what you see in this image" \
  -F "input_type=image" \
  -F "file=@photo.jpg"

curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=edumentor_agent" \
  -F "input=Explain the concepts shown in this diagram" \
  -F "input_type=image" \
  -F "file=@diagram.png"
```

#### Audio Upload and Processing
```bash
curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=audio_agent" \
  -F "input=Transcribe this uploaded audio" \
  -F "input_type=audio" \
  -F "file=@recording.wav"

curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=edumentor_agent" \
  -F "input=Summarize this lecture recording" \
  -F "input_type=audio" \
  -F "file=@lecture.mp3"
```

### Python Requests Examples

#### Basic Request Function
```python
import requests
import json

def make_api_request(agent, input_text, input_type="text", file_path=""):
    """Make API request to BHIV Core."""
    url = "http://localhost:8002/handle_task"
    payload = {
        "agent": agent,
        "input": input_text,
        "input_type": input_type,
        "pdf_path": file_path
    }
    response = requests.post(url, json=payload)
    return response.json()

# Usage examples
result = make_api_request("edumentor_agent", "Explain quantum physics", "text")
result = make_api_request("archive_agent", "Summarize document", "pdf", "document.pdf")
result = make_api_request("image_agent", "Describe image", "image", "photo.jpg")
result = make_api_request("audio_agent", "Transcribe audio", "audio", "recording.wav")
```

#### File Upload Function
```python
def upload_and_process(agent, input_text, file_path, input_type):
    """Upload file and process with agent."""
    url = "http://localhost:8002/handle_task_with_file"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'agent': agent,
            'input': input_text,
            'input_type': input_type
        }
        response = requests.post(url, files=files, data=data)
    
    return response.json()

# Usage examples
result = upload_and_process("edumentor_agent", "Analyze document", "report.pdf", "pdf")
result = upload_and_process("image_agent", "Describe image", "photo.jpg", "image")
result = upload_and_process("audio_agent", "Transcribe", "audio.wav", "audio")
```

## Available Agents

| Agent ID | Description | Best For | Input Types |
|----------|-------------|----------|-------------|
| `edumentor_agent` | Educational content and explanations | Learning, tutorials, explanations | text, pdf, image, audio |
| `vedas_agent` | Spiritual and philosophical guidance | Vedic wisdom, spirituality, philosophy | text |
| `wellness_agent` | Health and wellness advice | Mental health, wellness tips, lifestyle | text |
| `archive_agent` | PDF document processing | Document summarization, extraction | pdf |
| `image_agent` | Image analysis and description | Image captioning, visual analysis | image |
| `audio_agent` | Audio transcription and processing | Speech-to-text, audio analysis | audio |
| `text_agent` | General text processing | Text analysis, summarization | text |
| `stream_transformer_agent` | General purpose processing | Fallback for various tasks | text, pdf, image, audio |

## File Formats

### Supported Audio Formats
- `.wav` (recommended)
- `.mp3`
- `.ogg`
- `.flac`
- `.m4a`

### Supported Image Formats
- `.jpg`, `.jpeg`
- `.png`
- `.gif`
- `.bmp`
- `.tiff`

### Supported Document Formats
- `.pdf`

## Testing Scripts

### Complete Test Script
```python
#!/usr/bin/env python3
"""Comprehensive test script for all BHIV Core agents."""

import requests
import json
import os

BASE_URL = "http://localhost:8002"

def test_all_agents():
    """Test all available agents with different input types."""
    
    tests = [
        # Text processing tests
        {
            "name": "Edumentor Text",
            "payload": {
                "agent": "edumentor_agent",
                "input": "Explain machine learning algorithms",
                "input_type": "text"
            }
        },
        {
            "name": "Vedas Text",
            "payload": {
                "agent": "vedas_agent", 
                "input": "What is dharma in Hindu philosophy?",
                "input_type": "text"
            }
        },
        {
            "name": "Wellness Text",
            "payload": {
                "agent": "wellness_agent",
                "input": "How to manage stress effectively?",
                "input_type": "text"
            }
        },
        
        # File processing tests (update paths as needed)
        {
            "name": "PDF Processing",
            "payload": {
                "agent": "archive_agent",
                "input": "Summarize this document",
                "pdf_path": "test.pdf",
                "input_type": "pdf"
            }
        },
        {
            "name": "Image Processing", 
            "payload": {
                "agent": "image_agent",
                "input": "Describe this image",
                "pdf_path": "test.jpg",
                "input_type": "image"
            }
        },
        {
            "name": "Audio Processing",
            "payload": {
                "agent": "audio_agent",
                "input": "Transcribe this audio",
                "pdf_path": "test.wav", 
                "input_type": "audio"
            }
        }
    ]
    
    for test in tests:
        print(f"\n--- Testing {test['name']} ---")
        try:
            response = requests.post(f"{BASE_URL}/handle_task", json=test['payload'])
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_all_agents()
```

### Quick Health Check
```bash
# Check if MCP Bridge is running
curl -X GET "http://localhost:8002/health" || echo "MCP Bridge not running"

# Check if Simple API is running  
curl -X GET "http://localhost:8000/" || echo "Simple API not running"
```

## Usage Tips

1. **Always start the MCP Bridge server first**: `python mcp_bridge.py`
2. **Use absolute file paths** or ensure files are in the project directory
3. **Check file permissions** - ensure the server can read your files
4. **Monitor logs** - check console output for detailed error messages
5. **Use appropriate agents** - match the agent to your task type for best results
6. **Test with small files first** - verify functionality before processing large files

## Error Handling

Common issues and solutions:

- **Connection refused**: Start the MCP Bridge server
- **File not found**: Check file paths and permissions
- **Agent not found**: Verify agent name spelling
- **Processing failed**: Check file format and size
- **Timeout errors**: Try smaller files or increase timeout values

## API Response Format

All API responses follow this structure:
```json
{
  "task_id": "uuid-string",
  "agent_output": {
    "result": "processed output",
    "model": "agent_name", 
    "tokens_used": 150,
    "cost_estimate": 0.0,
    "status": 200,
    "keywords": ["tag1", "tag2"]
  },
  "status": 200
}
```

For errors:
```json
{
  "task_id": "uuid-string",
  "agent_output": {
    "error": "error description",
    "status": 500,
    "keywords": []
  },
  "status": 500
}
```