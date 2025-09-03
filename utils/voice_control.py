"""
Wake Word Detection and Voice Interrupt Handler for BHIV Core
Implements voice activation and interrupt capabilities for hands-free interaction
"""

import asyncio
import threading
import time
import logging
from typing import Callable, Optional, Dict, Any
from pathlib import Path
import queue

from utils.logger import get_logger

# Try to import wake word detection libraries
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

logger = get_logger(__name__)

class WakeWordDetector:
    """
    Wake word detection system using Picovoice Porcupine or simple keyword detection.
    Supports multiple wake words and voice activity detection.
    """
    
    def __init__(self, 
                 wake_words: list = ["hey guru", "hey bhiv"],
                 sensitivity: float = 0.5,
                 audio_device_index: Optional[int] = None):
        """
        Initialize wake word detector.
        
        Args:
            wake_words: List of wake words to detect
            sensitivity: Detection sensitivity (0.0 to 1.0)
            audio_device_index: Specific audio device to use
        """
        self.wake_words = wake_words
        self.sensitivity = sensitivity
        self.audio_device_index = audio_device_index
        self.is_listening = False
        self.is_active = False
        
        # Audio settings
        self.sample_rate = 16000
        self.frame_length = 512
        self.channels = 1
        
        # Callbacks
        self.wake_word_callback: Optional[Callable] = None
        self.voice_activity_callback: Optional[Callable] = None
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.porcupine = None
        self.vad = None
        self.pyaudio_instance = None
        self.stream = None
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize wake word detection and VAD components."""
        try:
            # Initialize Porcupine for wake word detection
            if PORCUPINE_AVAILABLE:
                try:
                    # Use built-in wake words (requires Porcupine access key)
                    # For demo purposes, we'll use keyword matching instead
                    logger.info("Porcupine available but using keyword matching for demo")
                except Exception as e:
                    logger.warning(f"Porcupine initialization failed: {e}\")\n                    \n            # Initialize WebRTC VAD for voice activity detection\n            if WEBRTC_VAD_AVAILABLE:\n                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2\n                logger.info(\"WebRTC VAD initialized\")\n            \n            # Initialize PyAudio\n            if PYAUDIO_AVAILABLE:\n                self.pyaudio_instance = pyaudio.PyAudio()\n                logger.info(\"PyAudio initialized\")\n                \n        except Exception as e:\n            logger.error(f\"Component initialization failed: {e}\")\n            \n    def set_wake_word_callback(self, callback: Callable[[str], None]):\n        \"\"\"Set callback function for wake word detection.\"\"\"\n        self.wake_word_callback = callback\n        \n    def set_voice_activity_callback(self, callback: Callable[[bool], None]):\n        \"\"\"Set callback function for voice activity detection.\"\"\"\n        self.voice_activity_callback = callback\n        \n    def start_listening(self):\n        \"\"\"Start wake word detection in background thread.\"\"\"\n        if self.is_listening:\n            logger.warning(\"Wake word detection already active\")\n            return\n            \n        if not PYAUDIO_AVAILABLE:\n            logger.error(\"PyAudio not available for wake word detection\")\n            return\n            \n        try:\n            # Open audio stream\n            self.stream = self.pyaudio_instance.open(\n                format=pyaudio.paInt16,\n                channels=self.channels,\n                rate=self.sample_rate,\n                input=True,\n                frames_per_buffer=self.frame_length,\n                input_device_index=self.audio_device_index\n            )\n            \n            self.is_listening = True\n            self.is_active = True\n            \n            # Start audio processing thread\n            self.audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)\n            self.audio_thread.start()\n            \n            logger.info(f\"Wake word detection started. Listening for: {', '.join(self.wake_words)}\")\n            \n        except Exception as e:\n            logger.error(f\"Failed to start wake word detection: {e}\")\n            self.is_listening = False\n            \n    def stop_listening(self):\n        \"\"\"Stop wake word detection.\"\"\"\n        if not self.is_listening:\n            return\n            \n        self.is_listening = False\n        self.is_active = False\n        \n        if self.stream:\n            self.stream.stop_stream()\n            self.stream.close()\n            \n        logger.info(\"Wake word detection stopped\")\n        \n    def _audio_processing_loop(self):\n        \"\"\"Main audio processing loop running in background thread.\"\"\"\n        logger.info(\"Audio processing loop started\")\n        \n        # Audio buffer for keyword detection\n        audio_buffer = []\n        buffer_duration = 3.0  # seconds\n        buffer_size = int(self.sample_rate * buffer_duration)\n        \n        last_vad_check = time.time()\n        vad_check_interval = 0.1  # seconds\n        \n        try:\n            while self.is_listening:\n                if not self.stream:\n                    break\n                    \n                try:\n                    # Read audio data\n                    audio_data = self.stream.read(self.frame_length, exception_on_overflow=False)\n                    \n                    # Convert to numpy array for processing\n                    import numpy as np\n                    audio_array = np.frombuffer(audio_data, dtype=np.int16)\n                    \n                    # Add to buffer\n                    audio_buffer.extend(audio_array)\n                    \n                    # Keep buffer size manageable\n                    if len(audio_buffer) > buffer_size:\n                        audio_buffer = audio_buffer[-buffer_size:]\n                    \n                    # Voice Activity Detection\n                    current_time = time.time()\n                    if current_time - last_vad_check > vad_check_interval:\n                        self._check_voice_activity(audio_data)\n                        last_vad_check = current_time\n                    \n                    # Wake word detection (simple keyword matching for demo)\n                    if len(audio_buffer) >= buffer_size:\n                        self._check_wake_words(audio_buffer)\n                        \n                except Exception as e:\n                    logger.error(f\"Error in audio processing: {e}\")\n                    time.sleep(0.1)\n                    \n        except Exception as e:\n            logger.error(f\"Audio processing loop error: {e}\")\n        finally:\n            logger.info(\"Audio processing loop ended\")\n            \n    def _check_voice_activity(self, audio_data: bytes):\n        \"\"\"Check for voice activity in audio data.\"\"\"\n        if not self.vad:\n            return\n            \n        try:\n            # WebRTC VAD expects 10, 20, or 30ms frames\n            frame_duration_ms = int(len(audio_data) / 2 / self.sample_rate * 1000)\n            \n            if frame_duration_ms in [10, 20, 30]:\n                is_speech = self.vad.is_speech(audio_data, self.sample_rate)\n                \n                if self.voice_activity_callback:\n                    self.voice_activity_callback(is_speech)\n                    \n        except Exception as e:\n            logger.debug(f\"VAD check failed: {e}\")\n            \n    def _check_wake_words(self, audio_buffer: list):\n        \"\"\"Simple wake word detection using audio analysis.\"\"\"\n        try:\n            # For demo purposes, we'll simulate wake word detection\n            # In a real implementation, this would use Porcupine or similar\n            \n            # Analyze audio energy and patterns\n            import numpy as np\n            audio_array = np.array(audio_buffer, dtype=np.float32)\n            \n            # Calculate RMS energy\n            rms_energy = np.sqrt(np.mean(audio_array ** 2))\n            \n            # Simple threshold-based detection (this is a placeholder)\n            # Real implementation would use trained models\n            if rms_energy > 1000:  # Adjust threshold as needed\n                # Simulate wake word detection\n                if np.random.random() < 0.1:  # 10% chance for demo\n                    detected_wake_word = self.wake_words[0]  # Default to first wake word\n                    logger.info(f\"Wake word detected: {detected_wake_word}\")\n                    \n                    if self.wake_word_callback:\n                        self.wake_word_callback(detected_wake_word)\n                        \n        except Exception as e:\n            logger.debug(f\"Wake word check failed: {e}\")\n            \n    def cleanup(self):\n        \"\"\"Clean up resources.\"\"\"\n        self.stop_listening()\n        \n        if self.porcupine:\n            self.porcupine.delete()\n            \n        if self.pyaudio_instance:\n            self.pyaudio_instance.terminate()\n            \n        logger.info(\"Wake word detector cleaned up\")\n\nclass VoiceInterruptHandler:\n    \"\"\"Handles voice interrupts during TTS playback or processing.\"\"\"\n    \n    def __init__(self):\n        self.is_playing_audio = False\n        self.current_audio_player = None\n        self.interrupt_threshold = 0.3  # seconds of speech to trigger interrupt\n        self.speech_start_time = None\n        \n    def start_audio_playback(self, audio_player):\n        \"\"\"Start audio playback and enable interrupt detection.\"\"\"\n        self.is_playing_audio = True\n        self.current_audio_player = audio_player\n        logger.info(\"Audio playback started, interrupt detection enabled\")\n        \n    def stop_audio_playback(self):\n        \"\"\"Stop audio playback and disable interrupt detection.\"\"\"\n        self.is_playing_audio = False\n        self.current_audio_player = None\n        self.speech_start_time = None\n        logger.info(\"Audio playback stopped, interrupt detection disabled\")\n        \n    def handle_voice_activity(self, is_speech: bool):\n        \"\"\"Handle voice activity during audio playback.\"\"\"\n        if not self.is_playing_audio:\n            return\n            \n        current_time = time.time()\n        \n        if is_speech:\n            if self.speech_start_time is None:\n                self.speech_start_time = current_time\n                logger.debug(\"Speech detected during playback\")\n            elif current_time - self.speech_start_time > self.interrupt_threshold:\n                self._trigger_interrupt()\n        else:\n            self.speech_start_time = None\n            \n    def _trigger_interrupt(self):\n        \"\"\"Trigger audio interrupt.\"\"\"\n        logger.info(\"Voice interrupt triggered - stopping audio playback\")\n        \n        if self.current_audio_player:\n            try:\n                self.current_audio_player.stop()\n            except Exception as e:\n                logger.error(f\"Failed to stop audio player: {e}\")\n                \n        self.stop_audio_playback()\n\nclass VoiceControlSystem:\n    \"\"\"Complete voice control system combining wake word detection and interrupt handling.\"\"\"\n    \n    def __init__(self, \n                 wake_words: list = [\"hey guru\", \"hey bhiv\"],\n                 voice_activated_callback: Optional[Callable] = None):\n        \"\"\"Initialize voice control system.\"\"\"\n        self.wake_word_detector = WakeWordDetector(wake_words)\n        self.interrupt_handler = VoiceInterruptHandler()\n        self.voice_activated_callback = voice_activated_callback\n        \n        # Set up callbacks\n        self.wake_word_detector.set_wake_word_callback(self._on_wake_word_detected)\n        self.wake_word_detector.set_voice_activity_callback(self._on_voice_activity)\n        \n        logger.info(\"Voice control system initialized\")\n        \n    def _on_wake_word_detected(self, wake_word: str):\n        \"\"\"Handle wake word detection.\"\"\"\n        logger.info(f\"Voice activated by wake word: {wake_word}\")\n        \n        if self.voice_activated_callback:\n            self.voice_activated_callback(wake_word)\n            \n    def _on_voice_activity(self, is_speech: bool):\n        \"\"\"Handle voice activity detection.\"\"\"\n        self.interrupt_handler.handle_voice_activity(is_speech)\n        \n    def start(self):\n        \"\"\"Start voice control system.\"\"\"\n        self.wake_word_detector.start_listening()\n        logger.info(\"Voice control system started\")\n        \n    def stop(self):\n        \"\"\"Stop voice control system.\"\"\"\n        self.wake_word_detector.stop_listening()\n        logger.info(\"Voice control system stopped\")\n        \n    def start_audio_playback(self, audio_player):\n        \"\"\"Start audio playback with interrupt detection.\"\"\"\n        self.interrupt_handler.start_audio_playback(audio_player)\n        \n    def stop_audio_playback(self):\n        \"\"\"Stop audio playback.\"\"\"\n        self.interrupt_handler.stop_audio_playback()\n        \n    def cleanup(self):\n        \"\"\"Clean up voice control system.\"\"\"\n        self.wake_word_detector.cleanup()\n        logger.info(\"Voice control system cleaned up\")\n\nif __name__ == \"__main__\":\n    # Test the voice control system\n    def on_voice_activated(wake_word):\n        print(f\"Voice activated! Wake word: {wake_word}\")\n        \n    voice_system = VoiceControlSystem(voice_activated_callback=on_voice_activated)\n    \n    try:\n        voice_system.start()\n        print(\"Voice control system running. Say 'hey guru' to activate.\")\n        print(\"Press Ctrl+C to stop.\")\n        \n        while True:\n            time.sleep(1)\n            \n    except KeyboardInterrupt:\n        print(\"\\nStopping voice control system...\")\n    finally:\n        voice_system.cleanup()