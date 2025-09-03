"""
Voice Persona Agent for BHIV Core - Multi-language TTS with voice personas
Supports Hindi and English with different voice personalities
"""

import os
import uuid
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import io
import tempfile
import logging

from utils.logger import get_logger

# Try to import TTS libraries
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Try to import advanced TTS (for future enhancement)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

logger = get_logger(__name__)

class VoicePersonaAgent:
    """Agent for converting text to speech with multiple voice personas and language support."""
    
    def __init__(self):
        self.personas = {
            "guru": {
                "name": "Guru Voice",
                "description": "Wise, calm, and authoritative voice for spiritual content",
                "languages": ["hi", "en"],
                "speed": 0.9,
                "pitch": 0.8
            },
            "teacher": {
                "name": "Teacher Voice", 
                "description": "Clear, patient, educational voice for learning content",
                "languages": ["hi", "en"],
                "speed": 1.0,
                "pitch": 1.0
            },
            "friend": {
                "name": "Friendly Voice",
                "description": "Warm, conversational voice for casual interactions",
                "languages": ["hi", "en"],
                "speed": 1.1,
                "pitch": 1.2
            },
            "assistant": {
                "name": "Assistant Voice",
                "description": "Professional, efficient voice for task completion",
                "languages": ["hi", "en"],
                "speed": 1.0,
                "pitch": 1.0
            }
        }
        
        # Initialize TTS engines
        self.tts_engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                logger.info("Pyttsx3 TTS engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize pyttsx3: {e}")
        
        # Create audio output directory
        self.audio_dir = Path("audio_output")
        self.audio_dir.mkdir(exist_ok=True)
        
        logger.info(f"VoicePersonaAgent initialized with {len(self.personas)} personas")

    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Hindi or English."""
        # Simple heuristic - count Devanagari characters
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return 'en'
        
        hindi_ratio = hindi_chars / total_chars
        return 'hi' if hindi_ratio > 0.3 else 'en'

    def generate_speech_gtts(self, text: str, language: str, persona: str) -> Optional[str]:
        """Generate speech using Google TTS."""
        if not GTTS_AVAILABLE:
            logger.error("gTTS not available")
            return None
            
        try:
            # Map personas to different speech parameters
            persona_config = self.personas.get(persona, self.personas["assistant"])
            
            # For gTTS, we can only control language, not voice characteristics
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Generate unique filename
            filename = f"speech_{uuid.uuid4().hex[:8]}_{persona}_{language}.mp3"
            filepath = self.audio_dir / filename
            
            # Save to file
            tts.save(str(filepath))
            logger.info(f"Generated speech file: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"gTTS speech generation failed: {e}")
            return None

    def generate_speech_pyttsx3(self, text: str, persona: str) -> Optional[str]:
        """Generate speech using pyttsx3 (offline)."""
        if not self.tts_engine:
            logger.error("Pyttsx3 engine not available")
            return None
            
        try:
            persona_config = self.personas.get(persona, self.personas["assistant"])
            
            # Configure voice properties based on persona
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to select appropriate voice
                selected_voice = voices[0]  # Default
                for voice in voices:
                    if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                        selected_voice = voice
                        break
                self.tts_engine.setProperty('voice', selected_voice.id)
            
            # Set speech rate and pitch based on persona
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', int(rate * persona_config['speed']))
            
            # Generate unique filename
            filename = f"speech_{uuid.uuid4().hex[:8]}_{persona}.wav"
            filepath = self.audio_dir / filename
            
            # Save to file
            self.tts_engine.save_to_file(text, str(filepath))
            self.tts_engine.runAndWait()
            
            logger.info(f"Generated speech file: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Pyttsx3 speech generation failed: {e}")
            return None

    async def synthesize_speech(self, text: str, persona: str = "assistant", language: str = None) -> Dict[str, Any]:
        """Main method to synthesize speech with given persona and language."""
        try:
            # Auto-detect language if not specified
            if language is None:
                language = self.detect_language(text)
            
            # Validate persona
            if persona not in self.personas:
                logger.warning(f"Unknown persona '{persona}', using 'assistant'")
                persona = "assistant"
            
            # Check if persona supports the language
            persona_config = self.personas[persona]
            if language not in persona_config["languages"]:
                logger.warning(f"Persona '{persona}' doesn't support language '{language}', using English")
                language = "en"
            
            logger.info(f"Synthesizing speech: persona='{persona}', language='{language}', text='{text[:50]}...'")
            
            # Try gTTS first (supports multiple languages)
            audio_file = None
            method_used = None
            
            if GTTS_AVAILABLE and language in ['hi', 'en']:
                audio_file = self.generate_speech_gtts(text, language, persona)
                method_used = "gtts"
            
            # Fallback to pyttsx3 for English only
            if audio_file is None and PYTTSX3_AVAILABLE and language == 'en':
                audio_file = self.generate_speech_pyttsx3(text, persona)
                method_used = "pyttsx3"
            
            if audio_file is None:
                error_msg = "All TTS methods failed"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "status": 500,
                    "audio_file": None
                }
            
            return {
                "status": 200,
                "audio_file": audio_file,
                "persona": persona,
                "language": language,
                "method": method_used,
                "text_length": len(text),
                "persona_config": persona_config
            }
            
        except Exception as e:
            error_msg = f"Speech synthesis failed: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "status": 500,
                "audio_file": None
            }

    def list_personas(self) -> Dict[str, Any]:
        """Return available voice personas."""
        return {
            "personas": self.personas,
            "available_languages": ["hi", "en"],
            "tts_engines": {
                "gtts": GTTS_AVAILABLE,
                "pyttsx3": PYTTSX3_AVAILABLE
            }
        }

    async def process_task(self, text: str, persona: str = "assistant", language: str = None, task_id: str = None) -> Dict[str, Any]:
        """Process a TTS task."""
        task_id = task_id or str(uuid.uuid4())
        logger.info(f"VoicePersonaAgent processing task {task_id}")
        
        result = await self.synthesize_speech(text, persona, language)
        result["task_id"] = task_id
        return result

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old generated audio files."""
        try:
            import time
            current_time = time.time()
            cleaned_count = 0
            
            for audio_file in self.audio_dir.glob("speech_*.mp3"):
                file_age = current_time - audio_file.stat().st_mtime
                if file_age > (max_age_hours * 3600):
                    audio_file.unlink()
                    cleaned_count += 1
            
            for audio_file in self.audio_dir.glob("speech_*.wav"):
                file_age = current_time - audio_file.stat().st_mtime
                if file_age > (max_age_hours * 3600):
                    audio_file.unlink()
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old audio files")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")

if __name__ == "__main__":
    # Test the voice persona agent
    async def test_agent():
        agent = VoicePersonaAgent()
        
        # Test English
        result = await agent.process_task(
            text="Hello, I am your AI assistant. How can I help you today?",
            persona="assistant",
            language="en"
        )
        print("English result:", result)
        
        # Test Hindi
        result = await agent.process_task(
            text="नमस्ते, मैं आपका एआई सहायक हूं। आज मैं आपकी कैसे सहायता कर सकता हूं?",
            persona="guru",
            language="hi"
        )
        print("Hindi result:", result)
        
        # List personas
        print("Available personas:", agent.list_personas())

    asyncio.run(test_agent())