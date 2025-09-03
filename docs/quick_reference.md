# BHIV Core Quick Reference

## 🚀 Quick Start Commands

### Start Services
```bash
# Terminal 1: Core service
python mcp_bridge.py

# Terminal 2: Web interface
python integration/web_interface.py

# Or restart with extended timeouts
python restart_web_interface.py
```

### Test System
```bash
# Basic test
python cli_runner.py summarize "Hello world" edumentor_agent

# RL test
python cli_runner.py summarize "RL test" edumentor_agent --use-rl --rl-stats

# File test
python cli_runner.py summarize "Analyze" edumentor_agent --file test.pdf --input-type pdf
```

## 📱 Access Points

| Interface | URL | Purpose |
|-----------|-----|---------|
| **Web Dashboard** | `http://localhost:8003/dashboard` | Real-time analytics |
| **File Upload** | `http://localhost:8003/upload` | Web file processing |
| **API Health** | `http://localhost:8002/health` | System status |
| **Standalone Dashboard** | `dashboard_standalone.html` | Offline analytics |

## 🤖 Available Agents

| Agent | Purpose | Input Types | Example |
|-------|---------|-------------|---------|
| **edumentor_agent** | Educational content | text, pdf, image | General learning materials |
| **vedas_agent** | Spiritual/philosophical | text, pdf | Religious texts, philosophy |
| **wellness_agent** | Health & wellness | text, pdf | Health advice, wellness tips |
| **image_agent** | Image analysis | image | Photo description, analysis |
| **audio_agent** | Audio processing | audio | Transcription, analysis |

## 💻 CLI Commands

### Basic Usage
```bash
# Text processing
python cli_runner.py summarize "Your text" AGENT

# File processing
python cli_runner.py summarize "Description" AGENT --file FILE --input-type TYPE

# Batch processing
python cli_runner.py summarize "Description" AGENT --batch FOLDER/
```

### RL Options
```bash
--use-rl                    # Enable RL
--no-rl                     # Disable RL
--exploration-rate 0.3      # Set exploration rate
--rl-stats                  # Show RL statistics
```

### File Options
```bash
--file path/to/file         # Input file
--input-type pdf            # File type (text/pdf/image/audio)
--batch ./folder/           # Process folder
--output results.json       # Output file
--format json               # Output format (json/text/csv)
```

### System Options
```bash
--retries 5                 # Retry attempts
--delay 3                   # Delay between retries
--verbose                   # Detailed logging
```

## 🌐 API Endpoints

### Core Endpoints
```bash
# Text processing
POST /handle_task
{
  "agent": "edumentor_agent",
  "input": "Your text",
  "input_type": "text"
}

# File upload
POST /handle_task_with_file
Form data:
- agent: edumentor_agent
- input: Description
- input_type: pdf
- file: [file upload]

# Health check
GET /health

# Configuration
GET /config
POST /config/reload
```

### cURL Examples
```bash
# Text API
curl -X POST "http://localhost:8002/handle_task" \
  -H "Content-Type: application/json" \
  -d '{"agent":"edumentor_agent","input":"Test","input_type":"text"}'

# File upload API
curl -X POST "http://localhost:8002/handle_task_with_file" \
  -F "agent=edumentor_agent" \
  -F "input=Analyze this" \
  -F "input_type=pdf" \
  -F "file=@document.pdf"
```

## 📊 Dashboard Commands

### CLI Dashboard
```bash
# Summary report
python learning_dashboard.py --summary

# Model analysis
python learning_dashboard.py --models --top 10

# Fallback analysis
python learning_dashboard.py --fallbacks

# Export report
python learning_dashboard.py --summary --export report.txt
```

### Dashboard Metrics
- **Total Tasks**: Number of processed tasks
- **Avg Reward**: Average RL reward score
- **Success Rate**: Percentage of successful tasks
- **Active Models**: Number of models in use
- **Fallback Rate**: Frequency of model fallbacks

## 🔧 Configuration

### Environment Variables
```bash
# Core settings
USE_RL=true
RL_EXPLORATION_RATE=0.2
DEFAULT_TIMEOUT=120

# API keys
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key

# Timeouts
IMAGE_PROCESSING_TIMEOUT=180
AUDIO_PROCESSING_TIMEOUT=240
PDF_PROCESSING_TIMEOUT=150
FILE_UPLOAD_TIMEOUT=300
```

### Key Files
- `config/settings.py` - Core configuration
- `logs/learning_log.json` - RL learning data
- `logs/model_logs.json` - Model selection logs
- `dashboard_standalone.html` - Offline dashboard

## 🚨 Troubleshooting

### Common Issues
```bash
# Service not starting
netstat -an | findstr :8002
python restart_web_interface.py

# Timeout errors
set DEFAULT_TIMEOUT=180
set IMAGE_PROCESSING_TIMEOUT=300

# RL not learning
python learning_dashboard.py --models
python -c "from config.settings import RL_CONFIG; print(RL_CONFIG)"

# Check logs
tail -f logs/bhiv_core.log
```

### Debug Mode
```bash
# Verbose logging
python cli_runner.py --verbose summarize "Debug" edumentor_agent

# Check system health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## 📋 Testing Checklist

### Basic Tests
- [ ] CLI text processing works
- [ ] API responds to requests
- [ ] Web dashboard loads
- [ ] File upload works
- [ ] RL system learns

### File Type Tests
- [ ] PDF processing
- [ ] Image analysis
- [ ] Audio transcription
- [ ] Batch processing

### RL System Tests
- [ ] Model selection works
- [ ] Fallback triggers correctly
- [ ] Learning data logs
- [ ] Dashboard shows metrics

## 🎯 Performance Tips

### Optimization
```bash
# Fast processing (less exploration)
export RL_EXPLORATION_RATE=0.1

# Accurate processing (more exploration)
export RL_EXPLORATION_RATE=0.3

# Increase timeouts for complex tasks
export DEFAULT_TIMEOUT=300
```

### Monitoring
```bash
# Real-time monitoring
python learning_dashboard.py --summary

# System resources
top -p $(pgrep -f "python.*mcp_bridge")

# API health
curl http://localhost:8002/health
```

## 🔄 Maintenance

### Regular Tasks
```bash
# Weekly: Check system health
python learning_dashboard.py --summary

# Monthly: Backup data
cp logs/learning_log.json logs/backup_$(date +%Y%m%d).json

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete
```

### System Reset
```bash
# Restart services
pkill -f "python.*mcp_bridge"
pkill -f "python.*web_interface"
python mcp_bridge.py &
python integration/web_interface.py &

# Reset RL data (caution!)
echo "[]" > logs/learning_log.json
```

---

## 📞 Quick Help

| Need | Command | URL |
|------|---------|-----|
| **Start System** | `python mcp_bridge.py` | - |
| **Test CLI** | `python cli_runner.py summarize "test" edumentor_agent` | - |
| **View Dashboard** | - | `http://localhost:8003/dashboard` |
| **Check Health** | `curl http://localhost:8002/health` | `http://localhost:8002/health` |
| **Upload File** | - | `http://localhost:8003/upload` |
| **View Analytics** | `python learning_dashboard.py --summary` | `dashboard_standalone.html` |

**For detailed documentation, see: `docs/complete_usage_guide.md`**
