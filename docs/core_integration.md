# BHIV Core Integration Documentation

## Overview

This document describes the core integration components of the BHIV system, including the standard agent interface, orchestration layer, and integration points with other system components.

## Standard Agent Interface

All agents in the BHIV system implement a standard interface to ensure consistency and interoperability.

### Interface Definition

```python
def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard interface for all agents.
    
    Args:
        input_data: Dictionary containing input parameters
            Required keys:
                - input: The main input data (text, file path, etc.)
                - task_id: Unique identifier for the task (optional)
            Optional keys:
                - input_type: Type of input (text, audio, image, etc.)
                - model: Model to use for processing
                - tags: List of tags for the task
                - retries: Number of retry attempts
                - live_feed: Live feed data (for streaming)
    
    Returns:
        Dictionary with processing results in standardized format:
            - result: The main output of the agent
            - status: HTTP-style status code (200 for success)
            - model: The model used for processing
            - agent: The agent name
            - keywords: List of relevant keywords
            - processing_time: Time taken to process
            - tokens_used: Number of tokens used (if applicable)
            - cost_estimate: Estimated cost (if applicable)
    """
```

### Example Implementation

```python
class TextAgent(BaseAgent):
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task_id = input_data.get('task_id', str(uuid.uuid4()))
        input_text = input_data.get('input', '')
        input_type = input_data.get('input_type', 'text')
        model = input_data.get('model', 'edumentor_agent')
        tags = input_data.get('tags', [])
        live_feed = input_data.get('live_feed', '')
        retries = input_data.get('retries', 3)
        
        # Process the input
        result = self.process_text(input_text, task_id, retries)
        
        # Add metadata
        result['agent'] = 'text_agent'
        result['input_type'] = input_type
        
        return result
```

## Core Orchestration Layer

The core orchestration layer provides a lightweight mechanism for invoking multiple agents in sequence with support for fallback mechanisms.

### Key Features

1. **Agent Routing**: Automatically selects appropriate agents based on task context
2. **Fallback Handling**: Provides fallback mechanisms when primary agents fail
3. **Sequence Execution**: Executes multiple tasks in sequence
4. **Standardized I/O**: Ensures consistent input/output handling across all agents

### Core Orchestrator API

#### Single Task Execution

```python
def execute_task(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single task using the core orchestrator.
    
    Args:
        task_payload: Dictionary containing task information
            - input: The input data to process
            - agent: The agent to use (optional)
            - task_id: Unique identifier for the task
            - input_type: Type of input data
            - tags: List of tags for the task
            - retries: Number of retry attempts
            - fallback_agent: Fallback agent if primary fails
    
    Returns:
        Dictionary with task results and execution metadata
    """
```

#### Sequence Execution

```python
def execute_sequence(task_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Execute a sequence of tasks in order.
    
    Args:
        task_sequence: List of task payloads to execute in sequence
        
    Returns:
        List of results from each task execution
    """
```

### Example Usage

```python
# Single task execution
task_payload = {
    "input": "Explain quantum computing in simple terms",
    "agent": "text_agent",
    "input_type": "text",
    "tags": ["education", "physics"]
}

result = execute_task(task_payload)

# Sequence execution
task_sequence = [
    {
        "input": "Record a lecture on quantum computing",
        "agent": "audio_agent",
        "input_type": "audio"
    },
    {
        "input": "Summarize the lecture transcript",
        "agent": "text_agent",
        "input_type": "text"
    }
]

results = execute_sequence(task_sequence)
```

## Integration Points

### With Nipun's DB Storage

The core orchestration layer integrates with MongoDB for logging task executions and storing results. All task executions are automatically logged to the database with the following information:

- Task ID
- Agent used
- Input data
- Output results
- Processing time
- Status
- Timestamp

### With Anmol's Endpoints

The orchestration layer can route tasks to Anmol's existing endpoints through the HTTP API connection type. Agent configurations specify the endpoint URL and required headers.

### Configuration

Agent configurations are stored in `config/agent_configs.json`:

```json
{
  "edumentor_agent": {
    "connection_type": "python_module",
    "module_path": "agents.stream_transformer_agent",
    "class_name": "StreamTransformerAgent",
    "id": "edumentor_agent",
    "tags": ["summarize", "pdf", "text"],
    "weight": 0.8
  },
  "knowledge_agent": {
    "connection_type": "http_api",
    "endpoint": "http://localhost:8001/query-kb",
    "headers": {"Content-Type": "application/json"},
    "id": "knowledge_agent",
    "tags": ["semantic_search", "vedabase"]
  }
}
```

## Testing

To test the core integration:

1. Start the core API server:
   ```bash
   python core_api.py
   ```

2. Use the API documentation at `http://localhost:8003/docs` to test endpoints

3. Execute sample tasks through the API

## Future Enhancements

1. **Enhanced RL Integration**: Deeper integration with the reinforcement learning system for better agent selection
2. **Advanced Sequencing**: Support for conditional task execution based on previous results
3. **Parallel Execution**: Execute independent tasks in parallel for improved performance
4. **Monitoring Dashboard**: Real-time monitoring of task executions and agent performance

## Conclusion

The BHIV core integration provides a standardized, flexible framework for agent interaction and task orchestration. By following the standard interface and using the core orchestrator, new agents can be easily integrated into the system while maintaining consistency and reliability.