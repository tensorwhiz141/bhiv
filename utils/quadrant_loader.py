#!/usr/bin/env python3
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from agents.agent_registry import agent_registry
from utils.logger import get_logger
from reinforcement.replay_buffer import replay_buffer
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.rl_context import RLContext
from agents.agent_memory_handler import agent_memory_handler
from utils.mongo_logger import mongo_logger
import uuid
import importlib
import requests
import motor.motor_asyncio
from config.settings import MONGO_CONFIG, TIMEOUT_CONFIG
import shutil
import os
from tempfile import NamedTemporaryFile

logger = get_logger(__name__)
app = FastAPI(title="BHIV Core MCP Bridge", version="2.0.0")
rl_context = RLContext()

# Async MongoDB client
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONFIG['uri'])
mongo_db = mongo_client[MONGO_CONFIG['database']]
mongo_collection = mongo_db[MONGO_CONFIG['collection']]

# Health check data
health_status = {
    "startup_time": datetime.now(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "agent_status": {}
}

class TaskPayload(BaseModel):
    agent: str
    input: str
    pdf_path: str = ""
    input_type: str = "text"
    retries: int = 3
    fallback_model: str = "edumentor_agent"
    tags: List[str] = []

class QueryPayload(BaseModel):
    query: str
    filters: Dict[str, Any] = None
    task_id: str = None
    tags: List[str] = ["semantic_search"]

async def handle_task_request(payload: TaskPayload) -> dict:
    """Handle task request with agent routing."""
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    logger.info(f"[MCP_BRIDGE] Task ID: {task_id} | Agent: {payload.agent} | Input: {payload.input[:50]}... | File: {payload.pdf_path} | Type: {payload.input_type} | Tags: {payload.tags}")
    health_status["total_requests"] += 1

    try:
        # Find appropriate agent based on task context
        task_context = {
            "task": "process",
            "keywords": [payload.agent, payload.input_type, *payload.tags],
            "model": payload.agent,
            "input_type": payload.input_type,
            "tags": payload.tags,
            "task_id": task_id
        }
        agent_id = agent_registry.find_agent(task_context)
        agent_config = agent_registry.get_agent_config(agent_id)
        
        if not agent_config:
            logger.error(f"[MCP_BRIDGE] Task {task_id}: Agent config not found for {agent_id}")
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Route to appropriate handler based on connection type
        result = None
        attempts = 0
        timeout = TIMEOUT_CONFIG.get(payload.input_type, TIMEOUT_CONFIG.get('default_timeout', 120))

        while attempts < payload.retries:
            attempts += 1
            try:
                if agent_config['connection_type'] == 'python_module':
                    module_path = agent_config['module_path']
                    class_name = agent_config['class_name']
                    module = importlib.import_module(module_path)
                    agent_class = getattr(module, class_name)
                    agent = agent_class()
                    input_path = payload.pdf_path if payload.pdf_path else payload.input
                    result = agent.run(input_path, "", payload.agent, payload.input_type, task_id)
                    break
                elif agent_config['connection_type'] == 'http_api':
                    endpoint = agent_config['endpoint']
                    headers = agent_config.get('headers', {})
                    # Add API key if available
                    if 'api_key' in agent_config and agent_config['api_key']:
                        headers['X-API-Key'] = agent_config['api_key']
                    request_payload = {
                        'query': payload.input,
                        'user_id': 'bhiv_core',
                        'task_id': task_id,
                        'input_type': payload.input_type,
                        'tags': payload.tags
                    }
                    if payload.pdf_path:
                        request_payload['file_path'] = payload.pdf_path
                    logger.info(f"[MCP_BRIDGE] Task {task_id}: Sending request to {endpoint} with payload: {request_payload}, headers: {headers}")
                    response = requests.post(endpoint, json=request_payload, headers=headers, timeout=timeout)
                    response.raise_for_status()
                    result = response.json()
                    # Ensure result has required fields
                    if 'status' not in result:
                        result['status'] = 200
                    if 'model' not in result:
                        result['model'] = payload.agent
                    logger.info(f"[MCP_BRIDGE] Task {task_id}: Received response from {endpoint}: {result}")
                    break
                else:
                    logger.warning(f"[MCP_BRIDGE] Task {task_id}: Unknown connection type for {agent_id}, using fallback")
                    from agents.stream_transformer_agent import StreamTransformerAgent
                    agent = StreamTransformerAgent()
                    input_path = payload.pdf_path if payload.pdf_path else payload.input
                    result = agent.run(input_path, "", payload.agent, payload.input_type, task_id)
                    break
            except Exception as e:
                logger.warning(f"[MCP_BRIDGE] Task {task_id}: Attempt {attempts}/{payload.retries} failed: {str(e)}")
                if attempts == payload.retries and payload.fallback_model:
                    logger.info(f"[MCP_BRIDGE] Task {task_id}: Switching to fallback model {payload.fallback_model}")
                    payload.agent = payload.fallback_model
                    agent_id = agent_registry.find_agent({"task": "process", "keywords": [payload.fallback_model], "model": payload.fallback_model})
                    agent_config = agent_registry.get_agent_config(agent_id)
                    if not agent_config:
                        logger.error(f"[MCP_BRIDGE] Task {task_id}: Fallback agent {payload.fallback_model} not found")
                        raise HTTPException(status_code=404, detail=f"Fallback agent {payload.fallback_model} not found")
                if attempts == payload.retries:
                    error_msg = f"Task execution failed after {payload.retries} attempts: {str(e)}"
                    logger.error(f"[MCP_BRIDGE] Task {task_id}: {error_msg}")
                    raise HTTPException(status_code=500, detail=error_msg)
        
        # Enhanced logging
        processing_time = (datetime.now() - start_time).total_seconds()
        health_status["successful_requests"] += 1
        health_status["agent_status"][agent_id] = {"last_used": datetime.now().isoformat(), "status": "healthy"}

        # Log to MongoDB
        task_log_data = {
            "task_id": task_id,
            "agent": agent_id,
            "input": payload.input,
            "file_path": payload.pdf_path,
            "output": result,
            "timestamp": datetime.now(),
            "input_type": payload.input_type,
            "tags": payload.tags,
            "processing_time": processing_time,
            "success": result.get('status', 500) == 200
        }
        await mongo_collection.insert_one(task_log_data)
        await mongo_logger.log_task_execution(task_log_data)

        # Log token/cost data
        if 'tokens_used' in result or 'cost_estimate' in result:
            await mongo_logger.log_token_cost({
                'task_id': task_id,
                'model': result.get('model', payload.agent),
                'agent': agent_id,
                'tokens_used': result.get('tokens_used', 0),
                'cost_estimate': result.get('cost_estimate', 0.0),
                'processing_time': processing_time
            })

        # Add to agent memory
        memory_entry = {
            'task_id': task_id,
            'input': payload.input,
            'output': result,
            'input_type': payload.input_type,
            'tags': payload.tags,
            'model': result.get('model', payload.agent),
            'status': result.get('status', 200),
            'response_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        agent_memory_handler.add_memory(agent_id, memory_entry)

        # RL logging
        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, payload.input, result, agent_id, payload.agent, reward)
        rl_context.log_action(
            task_id=task_id,
            agent=agent_id,
            model=result.get('model', payload.agent),
            action="process_task",
            reward=reward,
            metadata={"input_type": payload.input_type, "tags": payload.tags}
        )

        return {"task_id": task_id, "agent_output": result, "status": "success"}
    
    except Exception as e:
        logger.error(f"[MCP_BRIDGE] Error processing task {task_id}: {str(e)}")
        health_status["failed_requests"] += 1
        health_status["agent_status"][agent_id] = {"last_used": datetime.now().isoformat(), "status": f"error: {str(e)}"}
        error_output = {"error": f"Task processing failed: {str(e)}", "status": 500}
        reward = get_reward_from_output(error_output, task_id)
        replay_buffer.add_run(task_id, payload.input, error_output, agent_id, payload.agent, reward)
        rl_context.log_action(
            task_id=task_id,
            agent=agent_id,
            model="none",
            action="process_task_failed",
            reward=reward,
            metadata={"input_type": payload.input_type, "tags": payload.tags}
        )
        return {"task_id": task_id, "agent_output": error_output, "status": "error"}

@app.post("/handle_task")
async def handle_task(payload: TaskPayload):
    """Handle task via JSON payload."""
    return await handle_task_request(payload)

@app.post("/handle_task_with_file")
async def handle_task_with_file(
    agent: str = Form(...),
    input: str = Form(...),
    file: UploadFile = File(None),
    input_type: str = Form("text"),
    retries: int = Form(3),
    fallback_model: str = Form("edumentor_agent"),
    tags: str = Form("")
):
    """Handle task with file upload."""
    task_id = str(uuid.uuid4())
    temp_file_path = ""
    
    try:
        if file:
            with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file_path = temp_file.name
                shutil.copyfileobj(file.file, temp_file)
            
            logger.info(f"[MCP_BRIDGE] Task ID: {task_id} | File uploaded: {file.filename} -> {temp_file_path}")
        
        payload = TaskPayload(
            agent=agent,
            input=input,
            pdf_path=temp_file_path,
            input_type=input_type,
            retries=retries,
            fallback_model=fallback_model,
            tags=tags.split(",") if tags else []
        )
        
        result = await handle_task_request(payload)
        return result
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/query-kb")
async def query_knowledge_base(payload: QueryPayload):
    """Handle Gurukul queries to the Qdrant Vedabase."""
    task_id = payload.task_id or str(uuid.uuid4())
    start_time = datetime.now()
    logger.info(f"[MCP_BRIDGE] Query KB Task ID: {task_id} | Query: {payload.query[:50]}... | Filters: {payload.filters} | Tags: {payload.tags}")
    health_status["total_requests"] += 1

    try:
        task_context = {
            "task": "semantic_search",
            "keywords": ["knowledge_agent", "semantic_search", "vedabase", *payload.tags],
            "model": "knowledge_agent",
            "input_type": "text",
            "tags": payload.tags,
            "task_id": task_id
        }
        agent_id = agent_registry.find_agent(task_context)
        agent_config = agent_registry.get_agent_config(agent_id)
        
        if not agent_config or agent_id != "knowledge_agent":
            logger.error(f"[MCP_BRIDGE] Task {task_id}: KnowledgeAgent not found")
            raise HTTPException(status_code=404, detail="KnowledgeAgent not found")
        
        module_path = agent_config['module_path']
        class_name = agent_config['class_name']
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        agent = agent_class()
        
        result = agent.query(payload.query, payload.filters or {}, task_id)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        health_status["successful_requests"] += 1
        health_status["agent_status"][agent_id] = {"last_used": datetime.now().isoformat(), "status": "healthy"}

        # Log to MongoDB
        task_log_data = {
            "task_id": task_id,
            "agent": agent_id,
            "input": payload.query,
            "file_path": "",
            "output": result,
            "timestamp": datetime.now(),
            "input_type": "text",
            "tags": payload.tags,
            "processing_time": processing_time,
            "success": result.get('status', 500) == 200
        }
        await mongo_collection.insert_one(task_log_data)
        await mongo_logger.log_task_execution(task_log_data)

        # Add to agent memory
        memory_entry = {
            'task_id': task_id,
            'input': payload.query,
            'output': result,
            'input_type': "text",
            'tags': payload.tags,
            'model': "none",
            'status': result.get('status', 200),
            'response_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        agent_memory_handler.add_memory(agent_id, memory_entry)

        # RL logging
        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, payload.query, result, agent_id, "none", reward)
        rl_context.log_action(
            task_id=task_id,
            agent=agent_id,
            model="none",
            action="query_knowledge_base",
            reward=reward,
            metadata={"query": payload.query, "filters": payload.filters or {}, "tags": payload.tags}
        )

        return {
            "status": "success",
            "task_id": task_id,
            "agent_output": result
        }
    
    except Exception as e:
        logger.error(f"[MCP_BRIDGE] Error processing query-kb task {task_id}: {str(e)}")
        health_status["failed_requests"] += 1
        health_status["agent_status"][agent_id] = {"last_used": datetime.now().isoformat(), "status": f"error: {str(e)}"}
        error_output = {"error": f"Query processing failed: {str(e)}", "status": 500}
        reward = get_reward_from_output(error_output, task_id)
        replay_buffer.add_run(task_id, payload.query, error_output, agent_id, "none", reward)
        rl_context.log_action(
            task_id=task_id,
            agent=agent_id,
            model="none",
            action="query_knowledge_base_failed",
            reward=reward,
            metadata={"query": payload.query, "filters": payload.filters or {}, "tags": payload.tags}
        )
        return {
            "status": "error",
            "task_id": task_id,
            "agent_output": error_output
        }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        await mongo_client.admin.command('ping')
        mongodb_status = "healthy"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        mongodb_status = f"unhealthy: {str(e)}"

    try:
        available_agents = len(agent_registry.list_agents())
        agent_registry_status = "healthy"
    except Exception as e:
        logger.error(f"Agent registry health check failed: {str(e)}")
        available_agents = 0
        agent_registry_status = f"unhealthy: {str(e)}"

    uptime_seconds = (datetime.now() - health_status["startup_time"]).total_seconds()
    total_requests = health_status["total_requests"]
    success_rate = (health_status["successful_requests"] / total_requests * 100) if total_requests > 0 else 0

    health_data = {
        "status": "healthy" if mongodb_status == "healthy" and agent_registry_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime_seconds,
        "services": {
            "mongodb": mongodb_status,
            "agent_registry": agent_registry_status,
            "available_agents": available_agents
        },
        "metrics": {
            "total_requests": total_requests,
            "successful_requests": health_status["successful_requests"],
            "failed_requests": health_status["failed_requests"],
            "success_rate_percent": round(success_rate, 2)
        },
        "agent_status": health_status["agent_status"]
    }

    return health_data

@app.get("/config")
async def get_config():
    """Get current configuration."""
    try:
        return {
            "agents": agent_registry.list_agents(),
            "mongodb": {
                "database": MONGO_CONFIG["database"],
                "collection": MONGO_CONFIG["collection"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/reload")
async def reload_config():
    """Reload agent configuration dynamically."""
    try:
        importlib.reload(importlib.import_module('agents.agent_registry'))
        from agents.agent_registry import agent_registry as new_registry
        global agent_registry
        agent_registry = new_registry
        logger.info("Agent configuration reloaded successfully")
        return {
            "status": "success",
            "message": "Configuration reloaded",
            "agents": agent_registry.list_agents(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reloading config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")

@app.post("/handle_multi_task")
async def handle_multi_task(request: dict):
    """Handle multiple files/inputs asynchronously for improved performance."""
    try:
        files = request.get('files', [])
        agent = request.get('agent', 'edumentor_agent')
        task_type = request.get('task_type', 'summarize')
        tags = request.get('tags', [])

        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        logger.info(f"[MCP_BRIDGE] Multi-task request: {len(files)} files with agent {agent} and tags {tags}")

        tasks = []
        for file_info in files:
            payload = TaskPayload(
                agent=agent,
                input=file_info.get('data', ''),
                pdf_path=file_info.get('path', ''),
                input_type=file_info.get('type', 'text'),
                retries=3,
                fallback_model='edumentor_agent',
                tags=tags
            )
            tasks.append(handle_task_request(payload))

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        processed_results = []
        successful_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Multi-task file {i} failed: {str(result)}")
                processed_results.append({
                    "file_index": i,
                    "filename": files[i].get('path', f'file_{i}'),
                    "error": str(result),
                    "status": 500
                })