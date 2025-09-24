"""
Core API for BHIV System

This module provides API endpoints for the core orchestration layer,
allowing external systems to interact with the BHIV agent system.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from orchestration.core_orchestrator import execute_task, execute_sequence
from utils.logger import get_logger
import uuid

logger = get_logger(__name__)

app = FastAPI(
    title="BHIV Core API",
    description="API for the BHIV Core Orchestration Layer",
    version="1.0.0"
)

class TaskPayload(BaseModel):
    """Payload for executing a single task."""
    input: str
    agent: Optional[str] = None
    task_id: Optional[str] = None
    input_type: Optional[str] = "text"
    tags: Optional[List[str]] = []
    retries: Optional[int] = 3
    fallback_agent: Optional[str] = "edumentor_agent"

class TaskSequencePayload(BaseModel):
    """Payload for executing a sequence of tasks."""
    tasks: List[TaskPayload]

class TaskResponse(BaseModel):
    """Response model for task execution."""
    task_id: str
    agent_output: Dict[str, Any]
    status: str

class SequenceResponse(BaseModel):
    """Response model for sequence execution."""
    results: List[TaskResponse]

@app.post("/execute_task", response_model=TaskResponse)
async def execute_single_task(payload: TaskPayload):
    """Execute a single task using the core orchestrator."""
    try:
        logger.info(f"Received task execution request: {payload}")
        result = execute_task(payload.dict())
        return TaskResponse(**result)
    except Exception as e:
        logger.error(f"Error executing task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute_sequence", response_model=SequenceResponse)
async def execute_task_sequence(payload: TaskSequencePayload):
    """Execute a sequence of tasks using the core orchestrator."""
    try:
        logger.info(f"Received sequence execution request with {len(payload.tasks)} tasks")
        task_list = [task.dict() for task in payload.tasks]
        results = execute_sequence(task_list)
        return SequenceResponse(results=[TaskResponse(**result) for result in results])
    except Exception as e:
        logger.error(f"Error executing task sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "BHIV Core API",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "BHIV Core API",
        "version": "1.0.0",
        "endpoints": {
            "execute_task": "POST /execute_task - Execute a single task",
            "execute_sequence": "POST /execute_sequence - Execute a sequence of tasks",
            "health": "GET /health - Health check"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="BHIV Core API")
    parser.add_argument("--port", type=int, default=8003, help="Port to run the server on (default: 8003)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  BHIV CORE API")
    print("="*60)
    print(f" Server URL: http://{args.host}:{args.port}")
    print(f" API Documentation: http://{args.host}:{args.port}/docs")
    print("\n Endpoints:")
    print("   POST /execute_task - Execute a single task")
    print("   POST /execute_sequence - Execute a sequence of tasks")
    print("   GET /health - Health check")
    print("="*60)
    
    uvicorn.run(app, host=args.host, port=args.port)