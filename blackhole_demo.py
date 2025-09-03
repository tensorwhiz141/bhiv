#!/usr/bin/env python3
"""
BHIV Core demo pipeline for processing multimodal inputs.
"""

import logging
from typing import Dict, Any
import requests
from agents.agent_registry import agent_registry
from utils.logger import get_logger
from utils.stream_handler import StreamHandler
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.replay_buffer import replay_buffer

logger = get_logger(__name__)

def run_demo_pipeline(input_path: str, live_feed: str = "", task: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run the BHIV Core demo pipeline."""
    logger.info(f"Starting BHIV Core demo pipeline with input: {input_path}, type: {task.get('input_type', 'text')}")
    
    task_id = task.get('task_id', str(uuid.uuid4()))
    agent_id = agent_registry.find_agent(task or {"task": "summarize", "keywords": ["summarize"], "model": "edumentor_agent", "input_type": "text"})
    agent_config = agent_registry.get_agent_config(agent_id)
    
    if not agent_config:
        logger.error(f"Invalid or unsupported agent: {agent_id}")
        output = {"error": "Invalid agent", "status": 400}
        reward = get_reward_from_output(output, task_id)
        replay_buffer.add_run(task_id, input_path, output, agent_id, task.get('model', 'unknown'), reward)
        return output
    
    try:
        if agent_config['connection_type'] == 'python_module':
            from agents.stream_transformer_agent import StreamTransformerAgent
            agent = StreamTransformerAgent()
            result = agent.run(input_path, live_feed, agent_id, task.get('input_type', 'text'), task_id)
        elif agent_config['connection_type'] == 'http_api':
            endpoint = agent_config['endpoint']
            headers = agent_config['headers']
            from agents.stream_transformer_agent import StreamTransformerAgent
            agent = StreamTransformerAgent()
            content = agent.extract_pdf_text(input_path) if task.get('input_type') == "pdf" else input_path
            response = requests.post(endpoint, json={'query': f"{content} {live_feed}".strip(), 'user_id': 'bhiv_core', 'input_type': task.get('input_type', 'text')}, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
        else:
            raise ValueError(f"Unsupported connection type: {agent_config['connection_type']}")
        
        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, input_path, result, agent_id, task.get('model', agent_id), reward)
        logger.info(f"Pipeline result: {result}")
        return {"result": result, "model": agent_id, "status": 200, "reward": reward}
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        output = {"error": f"Pipeline failed: {str(e)}", "status": 500}
        reward = get_reward_from_output(output, task_id)
        replay_buffer.add_run(task_id, input_path, output, agent_id, task.get('model', agent_id), reward)
        return output

if __name__ == "__main__":
    test_input = "test.pdf"  # Replace with a real input path
    test_feed = "Live feed: AI advancements in 2025."
    test_task = {"task": "summarize", "keywords": ["summarize"], "model": "edumentor_agent", "input_type": "pdf"}
    result = run_demo_pipeline(test_input, test_feed, test_task)
    print(result)


# """
# BHIV Core demo pipeline for processing PDFs and live feeds.
# """

# import logging
# from typing import Dict, Any

# import requests
# from agents.agent_registry import agent_registry
# from utils.logger import get_logger
# from utils.stream_handler import StreamHandler

# logger = get_logger(__name__)

# def run_demo_pipeline(pdf_path: str, live_feed: str = "", task: Dict[str, Any] = None) -> Dict[str, Any]:
#     """Run the BHIV Core demo pipeline."""
#     logger.info("Starting BHIV Core demo pipeline")
    
#     agent_id = agent_registry.find_agent(task or {"task": "summarize", "keywords": ["summarize"], "model": "edumentor_agent"})
#     agent_config = agent_registry.get_agent_config(agent_id)
    
#     if not agent_config:
#         logger.error(f"Invalid or unsupported agent: {agent_id}")
#         return {"error": "Invalid agent", "status": 400}
    
#     try:
#         if agent_config['connection_type'] == 'python_module':
#             from agents.stream_transformer_agent import StreamTransformerAgent
#             agent = StreamTransformerAgent()
#             result = agent.run(pdf_path, live_feed, agent_id)
#         elif agent_config['connection_type'] == 'http_api':
#             endpoint = agent_config['endpoint']
#             headers = agent_config['headers']
#             from agents.stream_transformer_agent import StreamTransformerAgent
#             agent = StreamTransformerAgent()
#             pdf_content = agent.extract_pdf_text(pdf_path)
#             response = requests.post(endpoint, json={'query': f"{pdf_content} {live_feed}".strip(), 'user_id': 'bhiv_core'}, headers=headers, timeout=15)
#             response.raise_for_status()
#             result = response.json()
#         else:
#             raise ValueError(f"Unsupported connection type: {agent_config['connection_type']}")
        
#         logger.info(f"Pipeline result: {result}")
#         return {"result": result, "model": agent_id, "status": 200}
#     except Exception as e:
#         logger.error(f"Error in pipeline: {str(e)}")
#         return {"error": f"Pipeline failed: {str(e)}", "status": 500}

# if __name__ == "__main__":
#     test_pdf = "test.pdf"  # Replace with a real PDF path
#     test_feed = "Live feed: AI advancements in 2025."
#     test_task = {"task": "summarize", "keywords": ["summarize"], "model": "edumentor_agent"}
#     result = run_demo_pipeline(test_pdf, test_feed, test_task)
#     print(result)

