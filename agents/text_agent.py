import logging
import time
from typing import Dict, Any
import uuid
from utils.logger import get_logger
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.replay_buffer import replay_buffer
from config.settings import MODEL_CONFIG
import os
from agents.base_agent import BaseAgent

# Conditional import for langchain
try:
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatGroq = None

logger = get_logger(__name__)

class TextAgent(BaseAgent):
    """Agent for processing text inputs using Groq."""
    def __init__(self):
        super().__init__()
        self.model_config = MODEL_CONFIG.get("edumentor_agent", {})
        self.llm = None
        if LANGCHAIN_AVAILABLE and ChatGroq:
            try:
                self.llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.1-8b-instant"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM: {e}")

    def process_text(self, text: str, task_id: str, retries: int = 3) -> Dict[str, Any]:
        """Summarize text using Groq's API with retry logic."""
        start_time = time.time()

        # Check if LLM is available
        if not self.llm:
            logger.error("LLM not available for text processing")
            return {
                "error": "LLM not available for text processing",
                "status": 500,
                "keywords": [],
                "processing_time": 0,
                "attempts": 0
            }

        for attempt in range(retries):
            try:
                logger.info(f"Processing text (attempt {attempt + 1}/{retries}) for task {task_id}")

                prompt = f"Summarize the following text in 50-100 words: {text}"
                response = self.llm.invoke(prompt)
                summary = response.content

                processing_time = time.time() - start_time

                logger.info(f"Text processing successful for task {task_id} in {processing_time:.2f}s")
                logger.debug(f"Input: {text[:100]}..., Summary: {summary[:100]}...")

                return {
                    "result": summary,
                    "model": "text_agent",
                    "tokens_used": len(text.split()) + len(summary.split()),  # Approximate
                    "cost_estimate": 0.0,  # Free tier for development
                    "status": 200,
                    "keywords": ["text", "summary"],
                    "processing_time": processing_time,
                    "inference_time": processing_time,  # For single API call
                    "attempts": attempt + 1
                }

            except Exception as e:
                processing_time = time.time() - start_time
                logger.warning(f"Text processing attempt {attempt + 1}/{retries} failed for task {task_id}: {str(e)}")

                if attempt < retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    logger.error(f"Text processing failed after {retries} attempts for task {task_id}: {str(e)}")
                    return {
                        "error": f"Text processing failed after {retries} attempts: {str(e)}",
                        "status": 500,
                        "keywords": [],
                        "processing_time": processing_time,
                        "attempts": retries
                    }

        # This should never be reached, but added for type safety
        return {
            "error": "Unexpected error in text processing",
            "status": 500,
            "keywords": [],
            "processing_time": time.time() - start_time,
            "attempts": retries
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard interface for TextAgent.
        
        Args:
            input_data: Dictionary containing input parameters
                Required keys:
                    - input: The text to process
                    - task_id: Unique identifier for the task (optional)
                Optional keys:
                    - input_type: Type of input (default: text)
                    - model: Model to use for processing (default: edumentor_agent)
                    - tags: List of tags for the task
                    - live_feed: Live feed data (for streaming)
                    - retries: Number of retry attempts (default: 3)
        
        Returns:
            Dictionary with processing results in standardized format
        """
        task_id = input_data.get('task_id', str(uuid.uuid4()))
        input_text = input_data.get('input', '')
        input_type = input_data.get('input_type', 'text')
        model = input_data.get('model', 'edumentor_agent')
        tags = input_data.get('tags', [])
        live_feed = input_data.get('live_feed', '')
        retries = input_data.get('retries', 3)
        
        logger.info(f"TextAgent starting task {task_id} with input_type: {input_type}, model: {model}")

        # Enhanced logging with input metadata
        input_length = len(input_text) if input_text else 0
        logger.debug(f"Task {task_id} - Input length: {input_length} characters")

        result = self.process_text(input_text, task_id, retries)

        # Add metadata to result
        result['agent'] = 'text_agent'
        result['input_type'] = input_type
        result['input_length'] = input_length

        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, input_text, result, "text_agent", model, reward)

        logger.info(f"TextAgent completed task {task_id} with status: {result.get('status', 'unknown')}")
        return result

if __name__ == "__main__":
    agent = TextAgent()
    test_input = "Algebraic equations are mathematical statements that show the equality of two expressions. They often involve variables and constants and are used to model real-world problems. Solving these equations involves finding the values of the variables that make the equation true."
    
    # Test with new interface
    input_data = {
        "input": test_input,
        "input_type": "text"
    }
    result = agent.run(input_data)
    print(result)