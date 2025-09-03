# BHIV Core Reinforcement Learning Layer

## Overview

The BHIV Core Reinforcement Learning (RL) Layer is a lightweight, production-ready intelligence system that learns from every decision and continuously improves model and agent selection. It acts as an observational learning layer that tracks performance, logs decisions, and provides intelligent recommendations without disrupting the core workflow.

## Key Features

- **Non-Intrusive Learning**: RL layer operates as an observer, never blocking core functionality
- **Dynamic Model Selection**: UCB-based algorithm for optimal model selection
- **Intelligent Agent Routing**: Context-aware agent selection with fallback mechanisms
- **Comprehensive Logging**: All decisions, rewards, and performance metrics are tracked
- **Fallback Intelligence**: Automatic model/agent switching on failures with learning
- **Real-time Analytics**: Performance dashboards and trend analysis
- **Production Ready**: Configurable, scalable, and fault-tolerant

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   MCP Bridge    │───▶│  Agent Router   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   RL Context    │    │ Agent Selector  │
                       │    Logger       │    │   (RL-based)    │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  LLM Router     │    │ Model Selector  │
                       │  (w/ Fallback)  │    │   (UCB-based)   │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Reward Function │    │ Learning Logs   │
                       │   & Feedback    │    │  & Analytics    │
                       └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. RL Context (`reinforcement/rl_context.py`)
Central logging and action tracking system that records every decision made by the system.

### 2. Model Selector (`reinforcement/model_selector.py`)
UCB-based model selection with dynamic exploration rates and task-specific weights.

### 3. Agent Selector (`reinforcement/agent_selector.py`)
Intelligent agent routing based on historical performance and task complexity.

### 4. Reward Functions (`reinforcement/reward_functions.py`)
Sophisticated reward calculation based on output quality, response time, and success metrics.

### 5. Replay Buffer (`reinforcement/replay_buffer.py`)
Stores past decisions and outcomes for learning and analysis.

## Configuration

### Environment Variables

```bash
# RL Configuration
USE_RL=true                          # Enable/disable RL layer
RL_EXPLORATION_RATE=0.2              # Initial exploration rate (0.0-1.0)
RL_MIN_EXPLORATION=0.05              # Minimum exploration rate
RL_EXPLORATION_DECAY=0.995           # Exploration decay factor
RL_MEMORY_SIZE=1000                  # Maximum memory entries
RL_CONFIDENCE_THRESHOLD=0.7          # Confidence threshold for decisions
RL_ENABLE_UCB=true                   # Enable Upper Confidence Bound
RL_ENABLE_FALLBACK_LEARNING=true     # Learn from fallback actions
RL_LOG_TO_MONGO=true                 # Log to MongoDB
```

### Configuration File (`config/settings.py`)

```python
RL_CONFIG = {
    "use_rl": True,
    "exploration_rate": 0.2,
    "buffer_file": "logs/learning_log.json",
    "model_log_file": "logs/model_logs.json",
    "agent_log_file": "logs/agent_logs.json",
    "memory_size": 1000,
    "min_exploration_rate": 0.05,
    "exploration_decay": 0.995,
    "confidence_threshold": 0.7,
    "enable_ucb": True,
    "enable_fallback_learning": True,
    "log_to_mongo": True
}
```

## Usage

### CLI with RL

```bash
# Enable RL for single task
python cli_runner.py summarize "Analyze this document" edumentor_agent --file document.pdf --use-rl

# Disable RL for batch processing
python cli_runner.py summarize "Process documents" edumentor_agent --batch ./documents --no-rl

# Custom exploration rate
python cli_runner.py summarize "Test task" edumentor_agent --use-rl --exploration-rate 0.3

# Show RL statistics after processing
python cli_runner.py summarize "Batch task" edumentor_agent --batch ./files --rl-stats
```

### Programmatic Usage

```python
from reinforcement.model_selector import model_selector
from reinforcement.agent_selector import agent_selector
from config.settings import RL_CONFIG

# Check if RL is enabled
if RL_CONFIG['use_rl']:
    # Get RL-recommended model
    selected_model = model_selector.select_model({
        "task_id": "task-123",
        "model": "edumentor_agent",
        "query": "What is machine learning?",
        "input_type": "text"
    })
    
    # Get RL-recommended agent
    selected_agent = agent_selector.select_agent({
        "task_id": "task-123",
        "input": "Process this document",
        "input_type": "pdf"
    })
```

## Learning Dashboard

Monitor RL performance with the built-in dashboard:

```bash
# Comprehensive summary
python learning_dashboard.py --summary

# Model performance analysis
python learning_dashboard.py --models --top 10

# Fallback frequency analysis
python learning_dashboard.py --fallbacks

# Export report
python learning_dashboard.py --summary --export rl_report.txt
```

## Fallback Mechanism

The RL layer includes intelligent fallback logic:

1. **Primary Model Fails**: Automatically tries fallback models
2. **Fallback Learning**: Records which fallbacks work best
3. **Adaptive Routing**: Learns to avoid problematic models
4. **Error Recovery**: Graceful degradation with full logging

### Fallback Hierarchy

```python
fallback_hierarchy = {
    'llama': ['edumentor_agent', 'vedas_agent', 'wellness_agent'],
    'edumentor_agent': ['vedas_agent', 'wellness_agent', 'llama'],
    'vedas_agent': ['edumentor_agent', 'wellness_agent', 'llama'],
    'wellness_agent': ['edumentor_agent', 'vedas_agent', 'llama']
}
```

## Reward System

The RL layer uses a sophisticated reward function that considers:

- **Output Quality**: Length, coherence, keyword relevance
- **Response Time**: Faster responses get higher rewards
- **Success Rate**: Successful completions vs. errors
- **Cost Efficiency**: Token usage and estimated costs
- **User Feedback**: Implicit feedback from usage patterns

### Reward Calculation

```python
def calculate_reward(output, task_id):
    base_reward = 0.5
    
    # Quality metrics
    if output.get('status') == 200:
        base_reward += 0.3
    
    # Length and content quality
    result_length = len(output.get('result', ''))
    if result_length > 100:
        base_reward += 0.2
    
    # Keyword relevance
    keywords = output.get('keywords', [])
    if keywords:
        base_reward += min(len(keywords) * 0.1, 0.3)
    
    # Response time bonus
    response_time = output.get('response_time', 0)
    if response_time < 5.0:
        base_reward += 0.2
    
    return min(base_reward, 2.0)  # Cap at 2.0
```

## Monitoring and Analytics

### Real-time Metrics

- Model performance trends
- Agent success rates
- Fallback frequency
- Cost optimization
- Response time analysis

### Log Files

- `logs/learning_log.json`: Complete task history
- `logs/model_logs.json`: Model selection decisions
- `logs/agent_logs.json`: Agent routing decisions
- `logs/agent_memory.json`: Agent memory cache

### MongoDB Collections

- `task_logs`: Complete task execution logs
- `token_costs`: Token usage and cost tracking
- `rl_actions`: RL decision logs
- `model_performance`: Model performance metrics
- `fallback_logs`: Fallback action logs

## Best Practices

1. **Start with Exploration**: Use higher exploration rates initially
2. **Monitor Performance**: Regular dashboard reviews
3. **Gradual Deployment**: Enable RL on subset of tasks first
4. **Fallback Testing**: Verify fallback chains work correctly
5. **Cost Monitoring**: Track token usage and costs
6. **Regular Cleanup**: Archive old logs periodically

## Troubleshooting

### Common Issues

1. **RL Not Learning**: Check exploration rate and reward function
2. **Poor Model Selection**: Verify reward signals are meaningful
3. **Fallback Loops**: Check fallback hierarchy configuration
4. **Memory Issues**: Adjust memory size and cleanup frequency

### Debug Mode

```bash
# Enable verbose logging
python cli_runner.py --verbose --use-rl --rl-stats

# Check RL configuration
python -c "from config.settings import RL_CONFIG; print(RL_CONFIG)"
```

## Future Enhancements

- **Deep RL Integration**: Neural network-based selection
- **Multi-objective Optimization**: Balance multiple metrics
- **A/B Testing Framework**: Compare RL vs. static selection
- **Advanced Analytics**: Predictive performance modeling
- **Auto-tuning**: Self-adjusting hyperparameters

## Support

For issues and questions:
- Check logs in `logs/` directory
- Review configuration in `config/settings.py`
- Use `--verbose` flag for detailed logging
- Run learning dashboard for performance insights
