"""
Core Orchestration Layer for BHIV System

This module provides a lightweight orchestration layer that can invoke multiple agents
in sequence, with support for fallback mechanisms and standardized input/output handling.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from agents.agent_registry import agent_registry
from utils.logger import get_logger
from reinforcement.rl_context import rl_context
from reinforcement.reward_functions import get_reward_from_output

logger = get_logger(__name__)

class CoreOrchestrator:
    """Lightweight orchestration layer for invoking multiple agents in sequence."""
    
    def __init__(self):
        self.agent_registry = agent_registry
        logger.info("CoreOrchestrator initialized")
    
    def execute_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task by routing it to the appropriate agent.
        
        Args:
            task_payload: Dictionary containing task information
                Required keys:
                    - input: The input data to process
                    - agent: The agent to use (optional, will be selected if not provided)
                Optional keys:
                    - task_id: Unique identifier for the task
                    - input_type: Type of input data
                    - tags: List of tags for the task
                    - retries: Number of retry attempts
                    - fallback_agent: Fallback agent if primary fails
        
        Returns:
            Dictionary with task results and execution metadata
        """
        task_id = task_payload.get('task_id', str(uuid.uuid4()))
        input_data = task_payload.get('input', '')
        agent_name = task_payload.get('agent')
        input_type = task_payload.get('input_type', 'text')
        tags = task_payload.get('tags', [])
        retries = task_payload.get('retries', 3)
        fallback_agent = task_payload.get('fallback_agent', 'edumentor_agent')
        
        logger.info(f"Executing task {task_id} with agent {agent_name} on input type {input_type}")
        
        # If no agent specified, find appropriate agent based on task context
        if not agent_name:
            task_context = {
                "task": "process",
                "keywords": tags,
                "model": agent_name,
                "input_type": input_type,
                "tags": tags,
                "task_id": task_id
            }
            agent_name = self.agent_registry.find_agent(task_context)
            logger.info(f"Selected agent {agent_name} for task {task_id}")
        
        # Get agent configuration
        agent_config = self.agent_registry.get_agent_config(agent_name)
        if not agent_config:
            error_msg = f"Agent {agent_name} not found in registry"
            logger.error(error_msg)
            return {
                "task_id": task_id,
                "error": error_msg,
                "status": 404
            }
        
        # Execute task with the selected agent
        result = self._execute_with_agent(
            agent_config, task_payload, task_id, retries, fallback_agent
        )
        
        # Log to RL context
        reward = get_reward_from_output(result, task_id)
        rl_context.log_action(
            task_id=task_id,
            agent=agent_name,
            model=result.get('model', agent_name),
            action="execute_task",
            reward=reward,
            metadata={
                "input_type": input_type,
                "tags": tags,
                "fallback_used": result.get('fallback_used', False)
            }
        )
        
        return {
            "task_id": task_id,
            "agent_output": result,
            "status": "success" if result.get('status', 500) == 200 else "error"
        }
    
    def _execute_with_agent(self, agent_config: Dict[str, Any], task_payload: Dict[str, Any], 
                           task_id: str, retries: int, fallback_agent: str) -> Dict[str, Any]:
        """
        Execute task with a specific agent, including fallback handling.
        
        Args:
            agent_config: Configuration for the agent
            task_payload: Task payload data
            task_id: Unique task identifier
            retries: Number of retry attempts
            fallback_agent: Fallback agent name
            
        Returns:
            Dictionary with execution results
        """
        input_data = task_payload.get('input', '')
        input_type = task_payload.get('input_type', 'text')
        tags = task_payload.get('tags', [])
        live_feed = task_payload.get('live_feed', '')
        
        # Try primary agent
        for attempt in range(retries):
            try:
                if agent_config['connection_type'] == 'python_module':
                    # Import and instantiate the agent
                    import importlib
                    module_path = agent_config['module_path']
                    class_name = agent_config['class_name']
                    module = importlib.import_module(module_path)
                    agent_class = getattr(module, class_name)
                    agent = agent_class()
                    
                    # Prepare input data for the agent
                    agent_input = {
                        'input': input_data,
                        'input_type': input_type,
                        'model': task_payload.get('agent', 'default'),
                        'tags': tags,
                        'live_feed': live_feed,
                        'task_id': task_id,
                        'retries': 1  # Single attempt per retry loop
                    }
                    
                    # Execute the agent
                    result = agent.run(agent_input)
                    result['fallback_used'] = False
                    return result
                    
                elif agent_config['connection_type'] == 'http_api':
                    # Handle HTTP API agents
                    import requests
                    endpoint = agent_config['endpoint']
                    headers = agent_config.get('headers', {})
                    
                    # Add API key if available
                    if 'api_key' in agent_config and agent_config['api_key']:
                        headers['X-API-Key'] = agent_config['api_key']
                    
                    # Prepare request payload
                    request_payload = {
                        'query': input_data,
                        'user_id': 'bhiv_core',
                        'task_id': task_id,
                        'input_type': input_type,
                        'tags': tags
                    }
                    
                    logger.info(f"Sending request to {endpoint} with payload: {request_payload}")
                    response = requests.post(endpoint, json=request_payload, headers=headers, timeout=120)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Ensure result has required fields
                    if 'status' not in result:
                        result['status'] = 200
                    if 'model' not in result:
                        result['model'] = task_payload.get('agent', 'default')
                        
                    result['fallback_used'] = False
                    logger.info(f"Received response from {endpoint}: {result}")
                    return result
                    
                else:
                    error_msg = f"Unknown connection type for agent: {agent_config['connection_type']}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed for agent: {str(e)}")
                if attempt == retries - 1:
                    # All attempts failed, try fallback
                    logger.info(f"Switching to fallback agent {fallback_agent}")
                    fallback_config = self.agent_registry.get_agent_config(fallback_agent)
                    if fallback_config:
                        try:
                            if fallback_config['connection_type'] == 'python_module':
                                import importlib
                                module_path = fallback_config['module_path']
                                class_name = fallback_config['class_name']
                                module = importlib.import_module(module_path)
                                agent_class = getattr(module, class_name)
                                agent = agent_class()
                                
                                # Prepare input data for fallback agent
                                agent_input = {
                                    'input': input_data,
                                    'input_type': input_type,
                                    'model': fallback_agent,
                                    'tags': tags,
                                    'live_feed': live_feed,
                                    'task_id': task_id,
                                    'retries': 1
                                }
                                
                                result = agent.run(agent_input)
                                result['fallback_used'] = True
                                return result
                                
                            elif fallback_config['connection_type'] == 'http_api':
                                import requests
                                endpoint = fallback_config['endpoint']
                                headers = fallback_config.get('headers', {})
                                
                                if 'api_key' in fallback_config and fallback_config['api_key']:
                                    headers['X-API-Key'] = fallback_config['api_key']
                                
                                request_payload = {
                                    'query': input_data,
                                    'user_id': 'bhiv_core',
                                    'task_id': task_id,
                                    'input_type': input_type,
                                    'tags': tags
                                }
                                
                                response = requests.post(endpoint, json=request_payload, headers=headers, timeout=120)
                                response.raise_for_status()
                                result = response.json()
                                
                                if 'status' not in result:
                                    result['status'] = 200
                                if 'model' not in result:
                                    result['model'] = fallback_agent
                                    
                                result['fallback_used'] = True
                                return result
                                
                        except Exception as fallback_e:
                            logger.error(f"Fallback agent {fallback_agent} also failed: {str(fallback_e)}")
                            return {
                                "error": f"Both primary and fallback agents failed. Primary: {str(e)}, Fallback: {str(fallback_e)}",
                                "status": 500,
                                "fallback_used": True
                            }
                    else:
                        return {
                            "error": f"Primary agent failed and fallback agent {fallback_agent} not found: {str(e)}",
                            "status": 500,
                            "fallback_used": False
                        }
        
        # This should not be reached, but added for safety
        return {
            "error": "Unexpected error in agent execution",
            "status": 500,
            "fallback_used": False
        }
    
    def execute_sequence(self, task_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a sequence of tasks in order.
        
        Args:
            task_sequence: List of task payloads to execute in sequence
            
        Returns:
            List of results from each task execution
        """
        results = []
        for i, task_payload in enumerate(task_sequence):
            logger.info(f"Executing task {i+1}/{len(task_sequence)} in sequence")
            result = self.execute_task(task_payload)
            results.append(result)
            
            # If task failed, log but continue with sequence
            if result.get('status') != 'success':
                logger.warning(f"Task {i+1} failed but continuing with sequence: {result.get('error')}")
        
        return results

# Global orchestrator instance
core_orchestrator = CoreOrchestrator()

def execute_task(task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to execute a single task."""
    return core_orchestrator.execute_task(task_payload)

def execute_sequence(task_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience function to execute a sequence of tasks."""
    return core_orchestrator.execute_sequence(task_sequence)