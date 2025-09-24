import logging
from utils.logger import get_logger
from typing import Dict, Any
import uuid

logger = get_logger(__name__)

class BaseAgent:
    """Base agent with memory scaffold and standard interface."""
    def __init__(self):
        self.memory = {}  # Placeholder for short-term JSON cache
    
    def store_memory(self, key: str, value: str):
        logger.info(f"Storing memory: {key}")
        self.memory[key] = value
    
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
        task_id = input_data.get('task_id', str(uuid.uuid4()))
        input_value = input_data.get('input', '')
        input_type = input_data.get('input_type', 'text')
        model = input_data.get('model', 'default')
        tags = input_data.get('tags', [])
        live_feed = input_data.get('live_feed', '')
        retries = input_data.get('retries', 3)
        
        logger.info(f"BaseAgent.run() called with task_id: {task_id}")
        
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the run method")