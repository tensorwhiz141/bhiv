#!/usr/bin/env python3
"""
MCP Test - Entry point for testing MCP functionality via CLI.
"""

import requests
from utils.logger import get_logger
from typing import Dict

logger = get_logger(__name__)

def run_task(task: Dict) -> Dict:
    """Run a task via the MCP bridge API."""
    agent_id = task.get('model', 'edumentor_agent')
    input_text = task.get('data', '')
    pdf_path = task.get('pdf_path', '')
    try:
        response = requests.post(
            "http://localhost:8000/handle_task",
            json={"agent": agent_id, "input": input_text, "pdf_path": pdf_path},
            timeout=15
        )
        response.raise_for_status()
        logger.info(f"Task result: {response.json()}")
        return response.json()
    except Exception as e:
        logger.error(f"Error running task: {str(e)}")
        return {"error": f"Task execution failed: {str(e)}", "status": 500}

if __name__ == "__main__":
    test_task = {
        "task": "summarize",
        "data": "Sample PDF content",
        "model": "edumentor_agent",
        "keywords": ["summarize"],
        "pdf_path": "test.pdf"  # Replace with a real PDF path
    }
    result = run_task(test_task)
    print(result)



# import requests
# from utils.logger import get_logger
# from typing import Dict

# logger = get_logger(__name__)

# def run_task(task: Dict) -> Dict:
#     """Run a task via the MCP bridge API."""
#     agent_id = task.get('model', 'llama')
#     input_text = task.get('data', '')
#     try:
#         response = requests.post(
#             "http://localhost:8000/handle_task",
#             json={"agent": agent_id, "input": input_text}
#         )
#         response.raise_for_status()
#         logger.info(f"Task result: {response.json()}")
#         return response.json()
#     except Exception as e:
#         logger.error(f"Error running task: {str(e)}")
#         return {"error": f"Task execution failed: {str(e)}", "status": 500}

# if __name__ == "__main__":
#     test_task = {"task": "summarize", "data": "Sample PDF content", "model": "llama", "keywords": ["summarize"]}
#     result = run_task(test_task)
#     print(result)