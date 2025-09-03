# BHIV Core Complete Usage Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Setup & Installation](#setup--installation)
3. [Starting Services](#starting-services)
4. [CLI Usage](#cli-usage)
5. [API Usage](#api-usage)
6. [Web Interface](#web-interface)
7. [Dashboard Access](#dashboard-access)
8. [Testing All Features](#testing-all-features)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Configuration](#advanced-configuration)

## 🎯 Project Overview

BHIV Core is an intelligent multimodal AI system with:
- **Reinforcement Learning Layer** for adaptive model selection
- **Multiple Agents**: Text, PDF, Image, Audio, and specialized agents
- **Fallback Mechanisms** for reliability
- **Real-time Analytics** and performance monitoring
- **Multiple Interfaces**: CLI, API, and Web Dashboard

### Core Components
- **MCP Bridge**: Central task routing and processing
- **Agent Registry**: Dynamic agent management
- **LLM Router**: Intelligent model selection with fallback
- **RL System**: Learning from every interaction
- **MongoDB Integration**: Comprehensive logging
- **Web Dashboard**: Real-time monitoring and analytics

## 🚀 Setup & Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional packages for enhanced features
pip install spacy
python -m spacy download en_core_web_sm
```

### Environment Configuration
Create a `.env` file in the project root:
```bash
# API Keys
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_KEY_BACKUP=your_backup_gemini_key

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=bhiv_core

# RL Configuration
USE_RL=true
RL_EXPLORATION_RATE=0.2
RL_CONFIDENCE_THRESHOLD=0.7

# Timeout Configuration
DEFAULT_TIMEOUT=120
IMAGE_PROCESSING_TIMEOUT=180
AUDIO_PROCESSING_TIMEOUT=240
PDF_PROCESSING_TIMEOUT=150
FILE_UPLOAD_TIMEOUT=300
```

## 🔧 Starting Services

### 1. Start MongoDB (if using)
```bash
# Windows
net start MongoDB

# Linux/Mac
sudo systemctl start mongod
```

### 2. Start MCP Bridge (Core Service)
```bash
# Terminal 1: Start the main MCP bridge
python mcp_bridge.py

# Should show:
# INFO: Started server process
# INFO: Uvicorn running on http://0.0.0.0:8002
```

### 3. Start Web Interface (Optional)
```bash
# Terminal 2: Start web interface
python integration/web_interface.py

# Or use the restart script with extended timeouts
python restart_web_interface.py

# Should show:
# INFO: Uvicorn running on http://0.0.0.0:8003
```

### 4. Verify Services
```bash
# Check MCP Bridge health
curl http://localhost:8002/health

# Check Web Interface
curl http://localhost:8003/health
```

## 💻 CLI Usage

### Basic Commands

#### Text Processing
```bash
# Simple text processing
python cli_runner.py summarize "Explain quantum physics" edumentor_agent

# With RL enabled
python cli_runner.py summarize "Explain machine learning" edumentor_agent --use-rl

# With custom exploration rate
python cli_runner.py summarize "Analyze this topic" vedas_agent --use-rl --exploration-rate 0.3
```

#### File Processing
```bash
# PDF processing
python cli_runner.py summarize "Analyze this document" edumentor_agent --file document.pdf --input-type pdf

# Image processing
python cli_runner.py describe "What's in this image?" image_agent --file photo.jpg --input-type image

# Audio processing
python cli_runner.py transcribe "Convert to text" audio_agent --file recording.wav --input-type audio
```

#### Batch Processing
```bash
# Process multiple files
python cli_runner.py summarize "Analyze documents" edumentor_agent --batch ./documents/

# Batch with specific file types
python cli_runner.py summarize "Process images" image_agent --batch ./images/ --input-type image

# Batch with output file
python cli_runner.py summarize "Batch analysis" edumentor_agent --batch ./files/ --output results.json
```

#### Advanced CLI Options
```bash
# Full feature example
python cli_runner.py summarize "Complex analysis" edumentor_agent \
  --file document.pdf \
  --input-type pdf \
  --use-rl \
  --exploration-rate 0.25 \
  --retries 5 \
  --delay 3 \
  --output analysis_results.json \
  --format json \
  --verbose \
  --rl-stats

# Available agents
python cli_runner.py --help
```

### CLI Flags Reference
| Flag | Description | Example |
|------|-------------|---------|
| `--use-rl` | Enable RL model selection | `--use-rl` |
| `--no-rl` | Disable RL | `--no-rl` |
| `--exploration-rate` | Set RL exploration (0.0-1.0) | `--exploration-rate 0.3` |
| `--rl-stats` | Show RL statistics after processing | `--rl-stats` |
| `--file` | Input file path | `--file document.pdf` |
| `--input-type` | File type (text/pdf/image/audio) | `--input-type image` |
| `--batch` | Process directory of files | `--batch ./files/` |
| `--output` | Output file path | `--output results.json` |
| `--format` | Output format (json/text/csv) | `--format json` |
| `--retries` | Number of retry attempts | `--retries 5` |
| `--delay` | Delay between retries (seconds) | `--delay 3` |
| `--verbose` | Enable detailed logging | `--verbose` |

## 🌐 API Usage

### Base URLs
- **MCP Bridge**: `http://localhost:8002`
- **Web Interface**: `http://localhost:8003`

### Core Endpoints

#### 1. Text Processing
```bash
# Simple text task
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "edumentor_agent",
    "input": "Explain artificial intelligence",
    "input_type": "text"
  }'
```

#### 2. File Upload and Processing
```bash
# Upload and process file
curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=edumentor_agent" \
  -F "input=Analyze this document" \
  -F "input_type=pdf" \
  -F "file=@document.pdf"

# Image processing
curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=image_agent" \
  -F "input=Describe this image" \
  -F "input_type=image" \
  -F "file=@photo.jpg"

# Audio processing
curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=audio_agent" \
  -F "input=Transcribe this audio" \
  -F "input_type=audio" \
  -F "file=@recording.wav"
```

#### 3. Advanced API Features
```bash
# With retry and fallback configuration
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "edumentor_agent",
    "input": "Complex analysis task",
    "input_type": "text",
    "retries": 5,
    "fallback_model": "vedas_agent"
  }'

# Multi-task processing
curl -X POST "http://localhost:8002/handle_multi_task" \
  -H "Content-Type: application/json" \
  -d '{
    "files": [
      {"path": "doc1.pdf", "type": "pdf", "data": "Analyze document 1"},
      {"path": "img1.jpg", "type": "image", "data": "Describe image 1"}
    ],
    "agent": "edumentor_agent",
    "task_type": "multi_analyze"
  }'
```

#### 4. System Management
```bash
# Health check
curl "http://localhost:8002/health"

# Get configuration
curl "http://localhost:8002/config"

# Reload configuration
curl -X POST "http://localhost:8002/config/reload"

# Get task status
curl "http://localhost:8002/task_status/{task_id}"
```

### API Response Format
```json
{
  "task_id": "uuid-string",
  "agent_output": {
    "result": "Agent response text",
    "model": "model_used",
    "tokens_used": 150,
    "cost_estimate": 0.0045,
    "status": 200,
    "retry_count": 0,
    "fallback_used": false,
    "processing_time": 2.34
  },
  "status": 200
}
```

## 🖥️ Web Interface

### Accessing the Web Interface
1. **Start the web interface**: `python integration/web_interface.py`
2. **Open browser**: Navigate to `http://localhost:8003`
3. **Login** (if authentication is enabled)

### Web Interface Features

#### 1. Dashboard
- **URL**: `http://localhost:8003/dashboard`
- **Features**: Real-time analytics, task monitoring, performance metrics

#### 2. File Upload Interface
- **URL**: `http://localhost:8003/upload`
- **Supports**: PDF, Images, Audio files
- **Features**: Drag-and-drop, progress tracking, batch upload

#### 3. Task Management
- **URL**: `http://localhost:8003/tasks`
- **Features**: View task history, status tracking, results download

#### 4. Configuration
- **URL**: `http://localhost:8003/config`
- **Features**: Agent settings, RL configuration, timeout settings

### Web Interface Usage Examples

#### Single File Upload
1. Go to `http://localhost:8003/upload`
2. Select agent (edumentor_agent, image_agent, etc.)
3. Enter task description
4. Upload file
5. Monitor progress and view results

#### Batch Processing
1. Select multiple files
2. Choose appropriate agent
3. Set batch processing options
4. Monitor progress in real-time
5. Download results when complete

## 📊 Dashboard Access

### Option 1: Web Dashboard
- **URL**: `http://localhost:8003/dashboard`
- **Features**: Real-time data, interactive charts, task management

### Option 2: Standalone HTML Dashboard
- **File**: `dashboard_standalone.html`
- **Usage**: Open in any browser, load data from JSON files
- **Features**: Offline viewing, data import/export

### Option 3: CLI Dashboard
```bash
# Comprehensive summary
python learning_dashboard.py --summary

# Model performance analysis
python learning_dashboard.py --models --top 10

# Fallback frequency analysis
python learning_dashboard.py --fallbacks

# Export report
python learning_dashboard.py --summary --export dashboard_report.txt
```

### Dashboard Features
- **Model Performance**: Compare average rewards, success rates
- **Cost Analysis**: Token usage, estimated costs per model
- **Trend Analysis**: Performance over time, learning curves
- **Error Monitoring**: Fallback frequency, error patterns
- **RL Statistics**: Exploration rates, confidence scores

## 🧪 Testing All Features

### 1. Basic Functionality Test
```bash
# Test all agents with simple text
python cli_runner.py summarize "Test message" edumentor_agent
python cli_runner.py summarize "Test message" vedas_agent
python cli_runner.py summarize "Test message" wellness_agent
```

### 2. File Processing Test
```bash
# Create test files
echo "Test content" > test.txt
echo "PDF test content" > test.pdf

# Test file processing
python cli_runner.py summarize "Analyze" edumentor_agent --file test.txt --input-type text
python cli_runner.py summarize "Analyze" edumentor_agent --file test.pdf --input-type pdf
```

### 3. RL System Test
```bash
# Test RL functionality
python cli_runner.py summarize "RL test 1" edumentor_agent --use-rl --rl-stats
python cli_runner.py summarize "RL test 2" vedas_agent --use-rl --rl-stats
python cli_runner.py summarize "RL test 3" wellness_agent --use-rl --rl-stats

# Check RL learning
python learning_dashboard.py --summary
```

### 4. API Test Script
```bash
# Create and run API test
cat > test_api.py << 'EOF'
import requests
import json

# Test basic API
response = requests.post("http://localhost:8002/handle_task", 
    json={"agent": "edumentor_agent", "input": "API test", "input_type": "text"})
print("API Test:", response.json())

# Test health endpoint
health = requests.get("http://localhost:8002/health")
print("Health:", health.json())
EOF

python test_api.py
```

### 5. Web Interface Test
1. Open `http://localhost:8003/dashboard`
2. Upload a test file via `http://localhost:8003/upload`
3. Monitor task progress
4. Check dashboard updates

### 6. Complete System Test
```bash
# Run comprehensive test
python cli_runner.py summarize "Complete system test" edumentor_agent \
  --use-rl \
  --exploration-rate 0.2 \
  --retries 3 \
  --verbose \
  --rl-stats \
  --output system_test_results.json

# Check results
cat system_test_results.json
python learning_dashboard.py --summary
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Service Not Starting
```bash
# Check if ports are in use
netstat -an | findstr :8002
netstat -an | findstr :8003

# Kill existing processes
python restart_web_interface.py
```

#### 2. Timeout Errors
```bash
# Check current timeout settings
python -c "from config.settings import TIMEOUT_CONFIG; print(TIMEOUT_CONFIG)"

# Increase timeouts via environment variables
set IMAGE_PROCESSING_TIMEOUT=300
set DEFAULT_TIMEOUT=180
```

#### 3. RL Not Learning
```bash
# Check RL configuration
python -c "from config.settings import RL_CONFIG; print(RL_CONFIG)"

# Verify learning logs
python learning_dashboard.py --models
```

#### 4. MongoDB Connection Issues
```bash
# Check MongoDB status
mongo --eval "db.adminCommand('ismaster')"

# Test connection
python -c "from config.settings import MONGO_CONFIG; print(MONGO_CONFIG)"
```

### Debug Mode
```bash
# Enable verbose logging
python cli_runner.py --verbose summarize "Debug test" edumentor_agent

# Check logs
tail -f logs/bhiv_core.log
```

## ⚙️ Advanced Configuration

### Environment Variables
```bash
# Core settings
export USE_RL=true
export RL_EXPLORATION_RATE=0.25
export DEFAULT_TIMEOUT=180

# Model settings
export GROQ_API_KEY=your_key
export GEMINI_API_KEY=your_key

# Database settings
export MONGO_URI=mongodb://localhost:27017
export MONGO_DATABASE=bhiv_core
```

### Configuration Files
- `config/settings.py`: Core system configuration
- `config/agent_configs.json`: Agent-specific settings
- `logs/learning_log.json`: RL learning data
- `logs/model_logs.json`: Model selection logs

### Performance Tuning
```bash
# Optimize for speed
export RL_EXPLORATION_RATE=0.1
export DEFAULT_TIMEOUT=60

# Optimize for accuracy
export RL_EXPLORATION_RATE=0.3
export RL_CONFIDENCE_THRESHOLD=0.8
```

## 📈 Monitoring & Analytics

### Real-time Monitoring
- **Web Dashboard**: `http://localhost:8003/dashboard`
- **CLI Dashboard**: `python learning_dashboard.py --summary`
- **API Health**: `curl http://localhost:8002/health`

### Performance Metrics
- Model success rates and average rewards
- Response times and token usage
- Fallback frequency and error patterns
- RL learning progress and confidence scores

### Log Analysis
```bash
# View recent activity
tail -f logs/bhiv_core.log

# Analyze learning patterns
python learning_dashboard.py --models --top 10

# Export performance report
python learning_dashboard.py --summary --export performance_report.txt
```

---

## 🎉 Quick Start Checklist

1. ✅ **Install dependencies**: `pip install -r requirements.txt`
2. ✅ **Set environment variables**: Create `.env` file
3. ✅ **Start MCP Bridge**: `python mcp_bridge.py`
4. ✅ **Start Web Interface**: `python integration/web_interface.py`
5. ✅ **Test CLI**: `python cli_runner.py summarize "Hello" edumentor_agent`
6. ✅ **Test API**: `curl -X POST http://localhost:8002/handle_task -d '{"agent":"edumentor_agent","input":"test"}'`
7. ✅ **Open Dashboard**: `http://localhost:8003/dashboard`
8. ✅ **Run RL Test**: `python cli_runner.py summarize "RL test" edumentor_agent --use-rl --rl-stats`

Your BHIV Core system is now fully operational! 🚀

## 🔬 Advanced Testing Scenarios

### Multimodal Processing Test
```bash
# Test all input types in sequence
python cli_runner.py summarize "Text analysis" edumentor_agent --input-type text
python cli_runner.py summarize "PDF analysis" edumentor_agent --file sample.pdf --input-type pdf
python cli_runner.py describe "Image analysis" image_agent --file sample.jpg --input-type image
python cli_runner.py transcribe "Audio analysis" audio_agent --file sample.wav --input-type audio
```

### Stress Testing
```bash
# Concurrent API requests
for i in {1..10}; do
  curl -X POST "http://localhost:8002/handle_task" \
    -H "Content-Type: application/json" \
    -d "{\"agent\":\"edumentor_agent\",\"input\":\"Stress test $i\",\"input_type\":\"text\"}" &
done
wait
```

### RL Learning Verification
```bash
# Generate learning data
for i in {1..20}; do
  python cli_runner.py summarize "Learning test $i" edumentor_agent --use-rl
done

# Analyze learning progress
python learning_dashboard.py --summary
python learning_dashboard.py --models --metric avg_reward
```

## 📋 Feature Verification Checklist

### Core Functionality
- [ ] **Text Processing**: Basic text analysis works
- [ ] **PDF Processing**: Document extraction and analysis
- [ ] **Image Processing**: Image description and analysis
- [ ] **Audio Processing**: Audio transcription and analysis
- [ ] **Batch Processing**: Multiple file processing
- [ ] **Error Handling**: Graceful failure and recovery

### RL System
- [ ] **Model Selection**: RL chooses appropriate models
- [ ] **Learning**: Performance improves over time
- [ ] **Exploration**: System tries different approaches
- [ ] **Fallback**: Automatic model switching on failures
- [ ] **Logging**: All decisions are recorded

### Interfaces
- [ ] **CLI**: Command-line interface works
- [ ] **API**: REST API endpoints respond correctly
- [ ] **Web Interface**: Web dashboard accessible
- [ ] **File Upload**: Web file upload works
- [ ] **Real-time Updates**: Dashboard shows live data

### Monitoring
- [ ] **Health Checks**: System status monitoring
- [ ] **Performance Metrics**: Response times tracked
- [ ] **Cost Tracking**: Token usage monitored
- [ ] **Error Monitoring**: Failures logged and analyzed
- [ ] **RL Analytics**: Learning progress visible

## 🚨 Emergency Procedures

### System Recovery
```bash
# Complete system restart
pkill -f "python.*mcp_bridge"
pkill -f "python.*web_interface"
sleep 5
python mcp_bridge.py &
python integration/web_interface.py &
```

### Data Backup
```bash
# Backup learning data
cp logs/learning_log.json logs/learning_log_backup_$(date +%Y%m%d).json
cp logs/model_logs.json logs/model_logs_backup_$(date +%Y%m%d).json
cp logs/agent_logs.json logs/agent_logs_backup_$(date +%Y%m%d).json
```

### Reset RL System
```bash
# Clear learning data (use with caution)
echo "[]" > logs/learning_log.json
echo "[]" > logs/model_logs.json
echo "[]" > logs/agent_logs.json
```

## 📞 Support & Maintenance

### Regular Maintenance
```bash
# Weekly: Clean old logs
find logs/ -name "*.log" -mtime +7 -delete

# Monthly: Backup learning data
python learning_dashboard.py --summary --export monthly_report_$(date +%Y%m).txt

# Quarterly: Performance review
python learning_dashboard.py --models --top 20 --export quarterly_analysis.txt
```

### Performance Optimization
```bash
# Monitor system resources
top -p $(pgrep -f "python.*mcp_bridge")
top -p $(pgrep -f "python.*web_interface")

# Check memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

## 🎯 Production Deployment Tips

### Security Considerations
- Set strong authentication credentials
- Use HTTPS in production
- Implement rate limiting
- Monitor API access logs
- Regular security updates

### Scalability
- Use load balancers for multiple instances
- Implement Redis for session management
- Use MongoDB replica sets
- Monitor resource usage
- Implement auto-scaling

### Monitoring in Production
- Set up health check endpoints
- Implement alerting for failures
- Monitor response times
- Track error rates
- Monitor RL performance metrics

---

## 🏆 Success Indicators

Your BHIV Core system is working correctly when:

1. **✅ All Services Running**: MCP Bridge and Web Interface respond to health checks
2. **✅ Agents Responding**: All agents (edumentor, vedas, wellness, image, audio) process requests
3. **✅ RL Learning**: Dashboard shows improving performance over time
4. **✅ Fallback Working**: System gracefully handles model failures
5. **✅ Data Logging**: MongoDB and JSON logs capture all activities
6. **✅ Dashboard Active**: Real-time metrics and analytics available
7. **✅ File Processing**: All file types (PDF, images, audio) process successfully
8. **✅ API Stable**: REST endpoints respond within timeout limits

**Congratulations! Your BHIV Core system is production-ready! 🎉**
