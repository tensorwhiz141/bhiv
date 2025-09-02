# BHIV Core - Second Installment

An advanced AI processing pipeline with multi-modal input support, reinforcement learning, Named Learning Object (NLO) generation, and production-ready web interface.

## 🚀 Features

### Core Capabilities
- **Multi-Modal Processing**: Handle text, PDF, image, and audio inputs seamlessly
- **Multi-Input Processing**: Combine different file types in a single analysis
- **Named Learning Objects**: Structured educational content with Bloom's taxonomy
- **Web Interface**: Bootstrap-based UI with real-time processing and authentication
- **Enhanced CLI**: Batch processing with progress bars and multiple output formats

### AI & Machine Learning
- **Reinforcement Learning**: Adaptive agent and model selection with UCB optimization
- **Dynamic Exploration**: Task-complexity-based exploration rates
- **Automatic Retraining**: Continuous model improvement from historical data
- **Bloom's Taxonomy Extraction**: Automatic cognitive level classification
- **Subject Tag Generation**: NLP-powered content categorization

### Production Features
- **Health Monitoring**: Comprehensive health checks and metrics
- **MongoDB Integration**: Persistent NLO storage with indexing
- **Retry Logic**: Exponential backoff for failed API calls
- **Error Recovery**: Graceful degradation and fallback mechanisms
- **Async Processing**: Concurrent multi-file processing
- **Docker Support**: Production-ready containerization

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   CLI Runner    │    │   Simple API    │
│   (Port 8003)   │    │  (Enhanced)     │    │   (Port 8001)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   MCP Bridge    │
                    │   (Port 8002)   │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Agent Registry  │    │ Nipun Adapter   │    │   MongoDB       │
│ (Dynamic Config)│    │ (NLO Generator) │    │ (NLO Storage)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Agent    │    │  Archive Agent  │    │  Image Agent    │    │  Audio Agent    │
│   (Enhanced)    │    │   (Enhanced)    │    │   (BLIP Model)  │    │ (Wav2Vec2 Model)│
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │
                    ┌─────────────────┐         ┌─────────────────┐
                    │ Reinforcement   │         │ RL Retraining   │
                    │ Learning System │         │    System       │
                    │ (UCB Enhanced)  │         │  (Automated)    │
                    └─────────────────┘         └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- MongoDB 5.0+ (for NLO storage)
- Groq API key
- 8GB+ RAM (16GB recommended for production)
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd BHIV-Second-Installment

# Install dependencies
pip install -r requirements.txt

# Install NLP models
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Running the Services

```bash
# Start MongoDB (ensure it's running)
sudo systemctl start mongod

# Start the MCP Bridge API
python mcp_bridge.py

# Start the Web Interface (in another terminal)
python integration/web_interface.py

# Optional: Start the Simple API
python simple_api.py
```

### Access the Application

- **Web Interface**: http://localhost:8003 (admin/secret or user/secret)
- **API Documentation**: http://localhost:8002/docs
- **Health Checks**: http://localhost:8002/health
