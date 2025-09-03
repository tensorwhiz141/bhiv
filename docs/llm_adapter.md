Transformer Adapter Documentation
Overview
The Transformer Adapter (integration/llm_router.py) enables the BHIV Core system to process tasks using multiple Large Language Models (LLMs) such as GPT, Gemini, Claude, and Grok. It provides a unified interface to route tasks, handle model-specific requirements, track token usage, and manage errors.
Adding a New Model
To add a new LLM (e.g., Grok), follow these steps:

Add a Model Function:
In llm_router.py, create a new method in the TransformerAdapter class:def run_grok(self, query: str) -> Dict[str, Any]:
    logger.info(f"Processing query with Grok: {query}")
    return {
        "result": "Grok response",
        "model": "grok",
        "tokens_used": self._estimate_tokens(query, "Grok response"),
        "cost_estimate": 0.0
    }




Update the Model Registry:
Add the model to the model_registry dictionary in __init__:self.model_registry["grok"] = self.run_grok




Configure API Keys (if applicable):
Add the model’s API key to config/settings.py:MODEL_CONFIG["grok"] = {"api_key": "your_grok_api_key"}




Test the Model:
Use the run_with_model function:adapter = TransformerAdapter()
result = adapter.run_with_model("grok", "Test query")





Token and Cost Tracking

Token Estimation: Tokens are estimated using a heuristic (1 token ≈ 0.75 words) based on input and output text length.
Input: Count words using regex (\w+).
Output: Count words in the LLM response.
Formula: tokens_used = (input_words + output_words) / 0.75.


Cost Estimation: Currently a placeholder (0.0). Update with actual cost calculations when API pricing is available.
Output Format: Each model returns:{
    "result": "Model response",
    "model": "model_name",
    "tokens_used": 500,
    "cost_estimate": 0.0
}



Error Handling

Invalid Query: Returns {"error": "Invalid query", "status": 400} if the query is empty or not a string.
Unsupported Model: Returns {"error": "Model not supported", "status": 400} if the model_name is not in the registry.
Execution Errors: Returns {"error": "Model execution failed: <error>", "status": 500} for runtime issues.

Integration with BHIV Core

MCP: Receives tasks via FastAPI (POST /run_task) and routes to the adapter via the agent registry.
Agent Registry: Calls run_with_model(model_name, query) for LLM tasks.
Output: Results are logged by the MCP in MongoDB and passed to the Nipun Learning Adapter.

Example Usage
adapter = TransformerAdapter()
result = adapter.run_with_model("gpt", "Summarize this PDF")
# Output: {"result": "Stub GPT answer", "model": "gpt", "tokens_used": 10, "cost_estimate": 0.0}


# LLM Adapter Documentation

## Overview
The LLM Adapter (`llm_router.py`) routes queries to various language models (e.g., LLaMA, Grok) and integrates with the RL layer for model selection and reward tracking.

## Functionality
- **Model Routing**: Supports LLaMA, Vedas Agent, Edumentor Agent, and Wellness Agent.
- **RL Integration**:
  - Uses `model_selector.py` for RL-based model selection.
  - Logs model selections and rewards to `rl_context`.
  - Updates model performance history with rewards.
- **Token and Cost Tracking**:
  - Tracks input/output tokens and estimates costs (mocked).
  - Logs token usage to MongoDB and `replay_buffer`.

## Usage
```python
from integration.llm_router import TransformerAdapter
adapter = TransformerAdapter()
result = adapter.run_with_model("edumentor_agent", "Summarize this text: AI advancements.")