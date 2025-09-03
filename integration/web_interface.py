#!/usr/bin/env python3
"""
Web Interface for BHIV Core - Bootstrap UI with real-time results and authentication.
"""

import os
import sys
import json
import uuid
import asyncio
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root (BHIV-Fouth-Installment-main) to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
import requests

# Simple logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from integration.nipun_adapter import get_nlos_by_subject, get_nlos_by_task_id
    logger.info("✅ Nipun adapter loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ Nipun adapter not available: {e}")
    def get_nlos_by_subject(*args, **kwargs): return []
    def get_nlos_by_task_id(*args, **kwargs): return None

import motor.motor_asyncio
try:
    from config.settings import MONGO_CONFIG, TIMEOUT_CONFIG
except ImportError:
    logger.warning("Config settings not available, using defaults")
    MONGO_CONFIG = {
        "uri": "mongodb://localhost:27017",
        "database": "bhiv_core",
        "collection": "task_logs"
    }
    TIMEOUT_CONFIG = {
        "default_timeout": 120,
        "image_processing_timeout": 180,
        "file_upload_timeout": 300
    }

# Initialize FastAPI app
app = FastAPI(title="BHIV Core Web Interface", version="1.0.0")

# Security setup
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simple user database (in production, use proper database)
USERS_DB = {
    "admin": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
    "user": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"   # secret
}

# Templates and static files
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("audio_output", exist_ok=True)  # For TTS audio files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory="audio_output"), name="audio")

# MongoDB client
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONFIG['uri'])
mongo_db = mongo_client[MONGO_CONFIG['database']]

# Task storage for real-time updates
active_tasks = {}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate user."""
    username = credentials.username
    if username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if not verify_password(credentials.password, USERS_DB[username]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return username

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: str = Depends(get_current_user)):
    """Home page with file upload interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": current_user,
        "title": "BHIV Core - AI Processing Pipeline"
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: str = Depends(get_current_user)):
    """Dashboard with recent tasks and analytics."""
    # Get recent tasks from MongoDB
    recent_tasks = await mongo_db.task_logs.find().sort("timestamp", -1).limit(10).to_list(10)
    
    # Get NLO statistics
    nlo_stats = await mongo_db.nlo_collection.aggregate([
        {"$group": {
            "_id": "$subject_tag",
            "count": {"$sum": 1},
            "avg_confidence": {"$avg": "$confidence"}
        }},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]).to_list(10)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": current_user,
        "recent_tasks": recent_tasks,
        "nlo_stats": nlo_stats,
        "title": "Dashboard - BHIV Core"
    })

@app.get("/voice", response_class=HTMLResponse)
async def voice_interface(request: Request, current_user: str = Depends(get_current_user)):
    """Voice interface page with real-time voice interaction."""
    return templates.TemplateResponse("voice_interface.html", {
        "request": request,
        "user": current_user,
        "title": "Voice Interface - BHIV Core"
    })

@app.post("/upload")
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    agent: str = Form("edumentor_agent"),
    task_description: str = Form(""),
    current_user: str = Depends(get_current_user)
):
    """Handle file uploads and process them."""
    task_id = str(uuid.uuid4())
    
    try:
        # Store task info for real-time updates
        active_tasks[task_id] = {
            "status": "processing",
            "files": [f.filename for f in files],
            "agent": agent,
            "user": current_user,
            "start_time": datetime.now(),
            "description": task_description
        }
        
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            temp_path = f"temp/{task_id}_{file.filename}"
            os.makedirs("temp", exist_ok=True)
            
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            temp_files.append({
                "path": temp_path,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content)
            })
        
        # Process files based on type
        results = []
        for temp_file in temp_files:
            # Determine input type from content type
            content_type = temp_file["content_type"] or ""
            if "pdf" in content_type:
                input_type = "pdf"
            elif "image" in content_type:
                input_type = "image"
            elif "audio" in content_type:
                input_type = "audio"
            else:
                input_type = "text"
            
            # Call MCP bridge API
            try:
                response = requests.post(
                    "http://localhost:8002/handle_task",
                    json={
                        "agent": agent,
                        "input": task_description,
                        "pdf_path": temp_file["path"],
                        "input_type": input_type
                    },
                    timeout=TIMEOUT_CONFIG.get('file_upload_timeout', 300)
                )
                response.raise_for_status()
                result = response.json()
                results.append({
                    "filename": temp_file["filename"],
                    "result": result,
                    "input_type": input_type
                })
            except Exception as e:
                logger.error(f"Error processing {temp_file['filename']}: {str(e)}")
                results.append({
                    "filename": temp_file["filename"],
                    "error": str(e),
                    "input_type": input_type
                })
        
        # Update task status
        active_tasks[task_id].update({
            "status": "completed",
            "results": results,
            "end_time": datetime.now(),
            "processing_time": (datetime.now() - active_tasks[task_id]["start_time"]).total_seconds()
        })
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file["path"])
            except:
                pass
        
        return JSONResponse({
            "task_id": task_id,
            "status": "completed",
            "results": results,
            "processing_time": active_tasks[task_id]["processing_time"]
        })
        
    except Exception as e:
        logger.error(f"Error in upload processing: {str(e)}")
        active_tasks[task_id] = {
            "status": "error",
            "error": str(e),
            "end_time": datetime.now()
        }
        return JSONResponse({
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        }, status_code=500)

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str, current_user: str = Depends(get_current_user)):
    """Get real-time task status."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_tasks[task_id].copy()
    
    # Convert datetime objects to strings for JSON serialization
    if "start_time" in task_info:
        task_info["start_time"] = task_info["start_time"].isoformat()
    if "end_time" in task_info:
        task_info["end_time"] = task_info["end_time"].isoformat()
    
    return JSONResponse(task_info)

@app.get("/download_nlo/{task_id}")
async def download_nlo(
    task_id: str,
    format: str = "json",
    current_user: str = Depends(get_current_user)
):
    """Download NLO data in specified format."""
    try:
        # Get NLO from MongoDB
        nlo = await get_nlos_by_task_id(task_id)
        
        if not nlo:
            raise HTTPException(status_code=404, detail="NLO not found")
        
        # Remove MongoDB ObjectId for JSON serialization
        if "_id" in nlo:
            del nlo["_id"]
        
        if format.lower() == "json":
            # Create temporary JSON file
            filename = f"nlo_{task_id}.json"
            filepath = f"temp/{filename}"
            os.makedirs("temp", exist_ok=True)
            
            with open(filepath, "w") as f:
                json.dump(nlo, f, indent=2, default=str)
            
            return FileResponse(
                filepath,
                media_type="application/json",
                filename=filename,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        elif format.lower() == "pdf":
            # For PDF generation, you would need a PDF library like reportlab
            # For now, return JSON with PDF content-type
            return JSONResponse(nlo, headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": f"attachment; filename=nlo_{task_id}.pdf"
            })
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        logger.error(f"Error downloading NLO: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/nlos")
async def get_nlos(
    subject: Optional[str] = None,
    limit: int = 10,
    current_user: str = Depends(get_current_user)
):
    """API endpoint to get NLOs by subject."""
    try:
        if subject:
            nlos = await get_nlos_by_subject(subject, limit)
        else:
            # Get recent NLOs
            cursor = mongo_db.nlo_collection.find().sort("timestamp", -1).limit(limit)
            nlos = await cursor.to_list(limit)
        
        # Clean up ObjectIds for JSON serialization
        for nlo in nlos:
            if "_id" in nlo:
                nlo["_id"] = str(nlo["_id"])
        
        return JSONResponse(nlos)
        
    except Exception as e:
        logger.error(f"Error getting NLOs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check MongoDB connection
        await mongo_client.admin.command('ping')
        
        # Check MCP bridge
        response = requests.get("http://localhost:8002/health", timeout=5)
        mcp_status = response.status_code == 200
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "mongodb": True,
                "mcp_bridge": mcp_status
            }
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status_code=503)

@app.post("/process_voice")
async def process_voice(
    audio: UploadFile = File(...),
    persona: str = Form("assistant"),
    language: str = Form("en"),
    current_user: str = Depends(get_current_user)
):
    """Process voice input through STT, agent processing, and TTS."""
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Save uploaded audio temporarily
        temp_audio_path = f"temp/voice_{task_id}.wav"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        logger.info(f"Voice processing started for task {task_id}")
        
        # Step 1: Speech-to-Text using AudioAgent
        try:
            from agents.audio_agent import AudioAgent
            audio_agent = AudioAgent()
            stt_result = audio_agent.process_audio(temp_audio_path, task_id)
            
            if stt_result.get("status") != 200:
                raise Exception(f"STT failed: {stt_result.get('error', 'Unknown error')}")
            
            transcription = stt_result.get("result", "")
            if not transcription.strip():
                raise Exception("No speech detected in audio")
            
            logger.info(f"STT successful: {transcription[:50]}...")
            
        except Exception as e:
            logger.error(f"STT processing failed: {e}")
            return JSONResponse({
                "status": 500,
                "error": f"Speech recognition failed: {str(e)}",
                "task_id": task_id
            })
        
        # Step 2: Process transcription through MCP Bridge
        try:
            response = requests.post(
                "http://localhost:8002/handle_task",
                json={
                    "agent": "text_agent",  # Use text agent for voice queries
                    "input": transcription,
                    "input_type": "text",
                    "task_id": task_id
                },
                timeout=30
            )
            response.raise_for_status()
            agent_result = response.json()
            
            response_text = agent_result.get("result", "I'm sorry, I couldn't process your request.")
            agent_used = agent_result.get("agent", "text_agent")
            
            logger.info(f"Agent processing successful: {agent_used}")
            
        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            response_text = "I'm sorry, I encountered an error while processing your request."
            agent_used = "fallback"
        
        # Step 3: Text-to-Speech using VoicePersonaAgent
        audio_file = None
        try:
            from agents.voice_persona_agent import VoicePersonaAgent
            tts_agent = VoicePersonaAgent()
            tts_result = await tts_agent.synthesize_speech(response_text, persona, language)
            
            if tts_result.get("status") == 200:
                audio_file = os.path.basename(tts_result.get("audio_file", ""))
                logger.info(f"TTS successful: {audio_file}")
            else:
                logger.warning(f"TTS failed: {tts_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"TTS processing failed: {e}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store conversation in MongoDB
        conversation_record = {
            "task_id": task_id,
            "user": current_user,
            "transcription": transcription,
            "response_text": response_text,
            "persona": persona,
            "language": language,
            "agent_used": agent_used,
            "audio_file": audio_file,
            "processing_time": processing_time,
            "timestamp": datetime.now(),
            "stt_method": stt_result.get("method", "unknown")
        }
        
        await mongo_db.voice_conversations.insert_one(conversation_record)
        
        return JSONResponse({
            "status": 200,
            "task_id": task_id,
            "transcription": transcription,
            "response_text": response_text,
            "audio_file": audio_file,
            "persona": persona,
            "language": language,
            "agent_used": agent_used,
            "processing_time": processing_time,
            "stt_method": stt_result.get("method", "unknown")
        })
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        return JSONResponse({
            "status": 500,
            "error": str(e),
            "task_id": task_id
        })
    
    finally:
        # Clean up temp audio file
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass

@app.post("/test_tts")
async def test_tts(
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """Test TTS functionality."""
    try:
        data = await request.json()
        text = data.get("text", "Hello, this is a test of the text-to-speech system.")
        persona = data.get("persona", "assistant")
        language = data.get("language", "en")
        
        from agents.voice_persona_agent import VoicePersonaAgent
        tts_agent = VoicePersonaAgent()
        result = await tts_agent.synthesize_speech(text, persona, language)
        
        if result.get("status") == 200:
            audio_file = os.path.basename(result.get("audio_file", ""))
            return JSONResponse({
                "status": 200,
                "audio_file": audio_file,
                "persona": persona,
                "language": language
            })
        else:
            return JSONResponse({
                "status": 500,
                "error": result.get("error", "TTS failed")
            })
            
    except Exception as e:
        return JSONResponse({
            "status": 500,
            "error": str(e)
        })

@app.get("/voice_conversations")
async def get_voice_conversations(
    limit: int = 20,
    current_user: str = Depends(get_current_user)
):
    """Get recent voice conversations."""
    try:
        conversations = await mongo_db.voice_conversations.find(
            {"user": current_user}
        ).sort("timestamp", -1).limit(limit).to_list(limit)
        
        # Convert ObjectId to string for JSON serialization
        for conv in conversations:
            if "_id" in conv:
                conv["_id"] = str(conv["_id"])
            if "timestamp" in conv:
                conv["timestamp"] = conv["timestamp"].isoformat()
        
        return JSONResponse({
            "status": 200,
            "conversations": conversations
        })
        
    except Exception as e:
        return JSONResponse({
            "status": 500,
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
