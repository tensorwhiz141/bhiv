# LLM Router Documentation

## Overview

The LLM Router (`integration/llm_router.py`) is the central component responsible for routing queries to appropriate Large Language Models (LLMs) with intelligent fallback mechanisms, RL-based selection, and comprehensive logging. It provides a unified interface for multiple models while ensuring reliability and optimal performance.

## Features

- **Multi-Model Support**: LLaMA, Vedas Agent, Edumentor Agent, Wellness Agent
- **RL-Based Selection**: Intelligent model selection using reinforcement learning
- **Automatic Fallback**: Seamless fallback to alternative models on failures
- **Token Tracking**: Comprehensive token usage and cost estimation
- **Error Recovery**: Graceful error handling with detailed logging
- **Calculator Integration**: Built-in mathematical expression evaluation
- **Performance Monitoring**: Real-time performance metrics and logging

## Architecture

```
┌─────────────────┐
│   User Query    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ TransformerAdapter │
│  - Query Analysis  │
│  - Model Selection │
│  - Fallback Logic  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Calculator    │    │  RL Selector    │    │ Fallback Chain  │
│   (Math Ops)    │    │ (UCB-based)     │    │ (Auto-retry)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────┐
                    │  Model Execution │
                    │  - LLaMA (Groq)  │
                    │  - Vedas Agent   │
                    │  - Edumentor     │
                    │  - Wellness      │
                    └─────────┬───────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Response + Logs │
                    │ - Reward Calc   │
                    │ - Token Count   │
                    │ - Cost Estimate │
                    └─────────────────┘
```

## Core Components

### TransformerAdapter Class

The main class that handles all LLM routing operations.

```python
class TransformerAdapter:
    def __init__(self):
        self.model_config = MODEL_CONFIG
        self.calculator = Calculator()
        self.llm = ChatGroq(...)
        self.fallback_hierarchy = {...}
    
    def run_with_model(self, model: str, query: str, live_feed: str = "", 
                      task_id: str = None, retry_count: int = 0, 
                      max_retries: int = 2) -> Dict[str, Any]:
        """Main method for executing queries with fallback support."""
```

### Supported Models

#### 1. LLaMA (via Groq)
- **Model**: `llama-3.1-8b-instant`
- **Provider**: Groq API
- **Use Case**: General-purpose text generation
- **Token Tracking**: Full support

#### 2. Vedas Agent
- **Endpoint**: `http://localhost:8001/ask-vedas`
- **Specialization**: Vedic philosophy and spiritual guidance
- **Authentication**: Gemini API key
- **Backup**: Automatic API key fallback

#### 3. Edumentor Agent
- **Endpoint**: `http://localhost:8001/edumentor`
- **Specialization**: Educational content and tutoring
- **Authentication**: Gemini API key
- **Backup**: Automatic API key fallback

#### 4. Wellness Agent
- **Endpoint**: `http://localhost:8001/wellness`
- **Specialization**: Health and wellness advice
- **Authentication**: Gemini API key
- **Backup**: Automatic API key fallback

## Usage

### Basic Usage

```python
from integration.llm_router import TransformerAdapter

adapter = TransformerAdapter()

# Simple query
result = adapter.run_with_model(
    model="edumentor_agent",
    query="Explain machine learning",
    task_id="task-123"
)

print(result['result'])  # Model response
print(result['tokens_used'])  # Token count
print(result['cost_estimate'])  # Estimated cost
```

### With Live Feed

```python
# Include additional context
result = adapter.run_with_model(
    model="vedas_agent",
    query="What is dharma?",
    live_feed="User is interested in Hindu philosophy",
    task_id="task-456"
)
```

### Mathematical Calculations

```python
# Automatic calculator detection
result = adapter.run_with_model(
    model="edumentor_agent",
    query="5 + 3 * 2",
    task_id="calc-789"
)
# Returns: {'result': 11, 'status': 200, ...}
```

## Fallback Mechanism

### Fallback Hierarchy

```python
fallback_hierarchy = {
    'llama': ['edumentor_agent', 'vedas_agent', 'wellness_agent'],
    'edumentor_agent': ['vedas_agent', 'wellness_agent', 'llama'],
    'vedas_agent': ['edumentor_agent', 'wellness_agent', 'llama'],
    'wellness_agent': ['edumentor_agent', 'vedas_agent', 'llama']
}
```

### Fallback Process

1. **Primary Model Execution**: Attempt with requested model
2. **Error Detection**: Catch network, API, or processing errors
3. **Fallback Selection**: Choose next model from hierarchy
4. **Retry Logic**: Attempt with fallback model
5. **Logging**: Record fallback action and outcome
6. **Final Response**: Return result or comprehensive error

### Fallback Metadata

Each response includes fallback information:

```python
{
    'result': 'Model response',
    'retry_count': 1,
    'fallback_used': True,
    'final_model': 'vedas_agent',
    'original_model': 'edumentor_agent',
    'status': 200
}
```

## RL Integration

### Model Selection

The router integrates with the RL model selector for intelligent model choice:

```python
# RL-based model selection (only on first attempt)
if retry_count == 0:
    selected_model = model_selector.select_model({
        "task_id": task_id,
        "model": model,
        "query": query
    })
    if selected_model != model:
        logger.info(f"RL model selector chose {selected_model} over {model}")
        model = selected_model
```

### Reward Tracking

Every model execution is tracked for RL learning:

```python
reward = get_reward_from_output(result, task_id)
model_selector.update_history(task_id, model, reward)
```

## Token and Cost Tracking

### Token Estimation

```python
def estimate_cost(self, tokens_used: int, model: str) -> float:
    """Estimate cost based on tokens and model."""
    rates = {
        'llama': 0.0001,
        'vedas_agent': 0.0002,
        'edumentor_agent': 0.0002,
        'wellness_agent': 0.0002
    }
    return tokens_used * rates.get(model, 0.0001)
```

### Token Calculation

- **LLaMA**: Actual token count from API response
- **Other Agents**: Estimated based on word count (1 token ≈ 0.75 words)

## Error Handling

### Error Types

1. **Network Errors**: Connection timeouts, DNS failures
2. **API Errors**: Authentication, rate limiting, server errors
3. **Processing Errors**: Invalid responses, parsing failures
4. **Configuration Errors**: Missing endpoints, invalid models

### Error Response Format

```python
{
    'error': 'All models failed. Last error: Connection timeout',
    'status': 500,
    'keywords': [],
    'retry_count': 2,
    'fallback_used': True,
    'final_model': 'wellness_agent',
    'original_model': 'edumentor_agent'
}
```

## Configuration

### Model Configuration

```python
MODEL_CONFIG = {
    "llama": {
        "api_url": "http://localhost:1234/v1/chat/completions",
        "model_name": "llama-3.1-8b-instruct"
    },
    "vedas_agent": {
        "endpoint": "http://localhost:8001/ask-vedas",
        "headers": {"Content-Type": "application/json"},
        "api_key": os.getenv("GEMINI_API_KEY"),
        "backup_api_key": os.getenv("GEMINI_API_KEY_BACKUP")
    },
    # ... other models
}
```

### Environment Variables

```bash
# Groq API for LLaMA
GROQ_API_KEY=your_groq_api_key

# Gemini API for specialized agents
GEMINI_API_KEY=your_primary_gemini_key
GEMINI_API_KEY_BACKUP=your_backup_gemini_key
```

## Logging and Monitoring

### Log Levels

- **INFO**: Model selection, fallback actions, successful completions
- **WARNING**: Retry attempts, fallback usage
- **ERROR**: Model failures, configuration issues
- **DEBUG**: Detailed execution flow, token calculations

### Log Examples

```
INFO - Routing query to model: edumentor_agent, query: Explain machine learning...
INFO - RL model selector chose vedas_agent over edumentor_agent
ERROR - Error running model edumentor_agent (attempt 1): Connection timeout
INFO - Attempting fallback from edumentor_agent to vedas_agent
INFO - Updated model vedas_agent with reward 1.425, avg_reward: 1.398
```

## Performance Optimization

### Best Practices

1. **Model Selection**: Use RL recommendations for optimal performance
2. **Caching**: Implement response caching for repeated queries
3. **Timeout Management**: Set appropriate timeouts for each model
4. **Load Balancing**: Distribute load across available models
5. **Monitoring**: Track performance metrics and adjust accordingly

### Performance Metrics

- **Response Time**: Time from query to response
- **Success Rate**: Percentage of successful completions
- **Fallback Rate**: Frequency of fallback usage
- **Token Efficiency**: Tokens per successful response
- **Cost Efficiency**: Cost per successful response

## Testing

### Unit Tests

```python
def test_model_execution():
    adapter = TransformerAdapter()
    result = adapter.run_with_model("edumentor_agent", "Test query")
    assert result['status'] == 200
    assert 'result' in result

def test_fallback_mechanism():
    # Test with invalid model to trigger fallback
    adapter = TransformerAdapter()
    result = adapter.run_with_model("invalid_model", "Test query")
    assert result['fallback_used'] == True
```

### Integration Tests

```bash
# Test all models
python -m pytest tests/test_llm_router.py

# Test specific model
python -c "
from integration.llm_router import TransformerAdapter
adapter = TransformerAdapter()
result = adapter.run_with_model('edumentor_agent', 'Test query')
print(result)
"
```

## Troubleshooting

### Common Issues

#### Model Not Responding

```
Error running model edumentor_agent: Connection timeout
```

**Solutions**:
1. Check if the model service is running
2. Verify endpoint URLs in configuration
3. Check network connectivity
4. Validate API keys

#### High Fallback Rate

```
Fallback rate: 45% for edumentor_agent
```

**Solutions**:
1. Check model service stability
2. Increase timeout values
3. Verify API key validity
4. Monitor service logs

#### Poor RL Performance

```
RL model selector consistently choosing suboptimal models
```

**Solutions**:
1. Check reward function implementation
2. Verify exploration/exploitation balance
3. Review historical performance data
4. Adjust RL parameters

### Debug Mode

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

adapter = TransformerAdapter()
result = adapter.run_with_model("edumentor_agent", "Debug query")
```

## API Reference

### TransformerAdapter Methods

```python
def run_with_model(self, model: str, query: str, live_feed: str = "", 
                  task_id: str = None, retry_count: int = 0, 
                  max_retries: int = 2) -> Dict[str, Any]:
    """Execute query with specified model and fallback support."""

def estimate_cost(self, tokens_used: int, model: str) -> float:
    """Estimate cost based on tokens and model."""

def _execute_model(self, model: str, query: str, live_feed: str, 
                  task_id: str) -> Dict[str, Any]:
    """Execute a specific model with proper error handling."""
```

### Response Format

```python
{
    'result': str,              # Model response
    'model': str,               # Final model used
    'tokens_used': int,         # Token count
    'cost_estimate': float,     # Estimated cost
    'status': int,              # HTTP-style status code
    'keywords': List[str],      # Extracted keywords
    'retry_count': int,         # Number of retries
    'fallback_used': bool,      # Whether fallback was used
    'final_model': str,         # Final model used
    'original_model': str,      # Originally requested model
    'sources': List[Dict],      # Source documents (if applicable)
}
```
