import logging
from typing import Dict, Any
import uuid
import os
import speech_recognition as sr
from utils.logger import get_logger
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.replay_buffer import replay_buffer
from config.settings import MODEL_CONFIG
from agents.base_agent import BaseAgent

# Try to import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

logger = get_logger(__name__)

class AudioAgent(BaseAgent):
    """Agent for processing audio inputs using multiple fallback methods."""
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.model_config = MODEL_CONFIG.get("edumentor_agent", {})
        
        # Initialize Whisper if available
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}")
        
        # Try to initialize Wav2Vec2 if available
        self.wav2vec2_available = False
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2_available = True
            logger.info("Wav2Vec2 model loaded successfully")
        except Exception as e:
            logger.warning(f"Wav2Vec2 not available: {e}")

    def load_audio_with_fallback(self, audio_path: str):
        """Load audio using multiple backends as fallback."""
        # Method 1: Try soundfile
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_path)
            logger.info(f"Loaded audio with soundfile: {samplerate}Hz")
            return data, samplerate, "soundfile"
        except Exception as e:
            logger.warning(f"soundfile failed: {e}")
        
        # Method 2: Try librosa
        try:
            import librosa
            data, samplerate = librosa.load(audio_path, sr=None)
            logger.info(f"Loaded audio with librosa: {samplerate}Hz")
            return data, samplerate, "librosa"
        except Exception as e:
            logger.warning(f"librosa failed: {e}")
        
        # Method 3: Try torchaudio
        try:
            import torchaudio
            waveform, samplerate = torchaudio.load(audio_path)
            data = waveform.numpy().flatten()
            logger.info(f"Loaded audio with torchaudio: {samplerate}Hz")
            return data, samplerate, "torchaudio"
        except Exception as e:
            logger.warning(f"torchaudio failed: {e}")
        
        # Method 4: Try pydub
        try:
            from pydub import AudioSegment
            import numpy as np
            audio = AudioSegment.from_file(audio_path)
            data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                data = data.reshape((-1, 2)).mean(axis=1)  # Convert stereo to mono
            data = data / (2**15)  # Normalize
            samplerate = audio.frame_rate
            logger.info(f"Loaded audio with pydub: {samplerate}Hz")
            return data, samplerate, "pydub"
        except Exception as e:
            logger.warning(f"pydub failed: {e}")
        
        raise Exception("All audio loading methods failed")

    def transcribe_with_whisper(self, audio_path: str) -> str:
        """Transcribe using OpenAI Whisper model."""
        try:
            logger.info(f"Using Whisper for transcription: {audio_path}")
            result = self.whisper_model.transcribe(audio_path)
            transcription = result["text"].strip()
            logger.info(f"Whisper transcription successful: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"Whisper transcription failed: {type(e).__name__}: {str(e)}")
            raise

    def transcribe_with_wav2vec2(self, audio_path: str) -> str:
        """Transcribe using Wav2Vec2 model."""
        try:
            import torch
            import numpy as np
            from scipy import signal
            
            logger.info(f"Loading audio for Wav2Vec2: {audio_path}")
            
            # Load audio using fallback methods
            data, samplerate, method = self.load_audio_with_fallback(audio_path)
            logger.info(f"Audio loaded with {method}: {samplerate}Hz, {len(data)} samples")
            
            # Ensure we have enough audio data
            min_samples = 1600  # 0.1 seconds at 16kHz
            if len(data) < min_samples:
                logger.warning(f"Audio too short: {len(data)} samples, padding to {min_samples}")
                data = np.pad(data, (0, min_samples - len(data)), mode='constant')
            
            # Resample to 16kHz if needed
            if samplerate != 16000:
                logger.info(f"Resampling from {samplerate}Hz to 16000Hz")
                num_samples = int(len(data) * 16000 / samplerate)
                data = signal.resample(data, num_samples)
                samplerate = 16000
                logger.info(f"Resampled to {len(data)} samples")
            
            # Ensure data is float32 and normalized
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(data)) > 0:
                data = data / np.max(np.abs(data))
            
            logger.info(f"Processing with Wav2Vec2: {len(data)} samples")
            
            # Process with Wav2Vec2
            inputs = self.processor(data, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            logger.info(f"Wav2Vec2 transcription successful: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Wav2Vec2 traceback: {traceback.format_exc()}")
            raise

    def transcribe_with_speech_recognition(self, audio_path: str) -> str:
        """Transcribe using SpeechRecognition library."""
        logger.info(f"Using SpeechRecognition for: {audio_path}")
        try:
            # Convert to WAV if needed using pydub
            temp_wav_path = audio_path
            if not audio_path.lower().endswith('.wav'):
                try:
                    from pydub import AudioSegment
                    import tempfile
                    audio = AudioSegment.from_file(audio_path)
                    temp_wav_path = tempfile.mktemp(suffix='.wav')
                    audio.export(temp_wav_path, format="wav")
                    logger.info(f"Converted audio to WAV: {temp_wav_path}")
                except Exception as e:
                    logger.error(f"Failed to convert audio format: {e}")
                    raise Exception(f"Audio format conversion failed: {e}")
            
            logger.info(f"Processing WAV file: {temp_wav_path}")
            
            try:
                with sr.AudioFile(temp_wav_path) as source:
                    logger.info("Recording audio data...")
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self.recognizer.record(source)
                    logger.info(f"Audio data recorded, duration: {len(audio_data.frame_data)} bytes")
            except Exception as e:
                logger.error(f"Failed to read audio file: {e}")
                raise Exception(f"Audio file reading failed: {e}")
                
            # Try Google Speech Recognition
            try:
                logger.info("Attempting Google Speech Recognition...")
                transcription = self.recognizer.recognize_google(audio_data)
                logger.info(f"Google Speech Recognition successful: {transcription}")
                return transcription
            except sr.RequestError as e:
                logger.error(f"Google Speech Recognition request error: {e}")
                raise Exception(f"Google Speech Recognition service error: {e}")
            except sr.UnknownValueError as e:
                logger.error(f"Google Speech Recognition could not understand audio: {e}")
                raise Exception("Could not understand audio - no speech detected or audio quality too poor")
            except Exception as e:
                logger.error(f"Unexpected Google Speech Recognition error: {e}")
                raise Exception(f"Speech recognition error: {e}")
        
        except Exception as e:
            logger.error(f"Speech recognition failed with error: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        finally:
            # Clean up temp file if created
            if temp_wav_path != audio_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                    logger.info(f"Cleaned up temp file: {temp_wav_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

    def process_audio(self, audio_path: str, task_id: str) -> Dict[str, Any]:
        """Transcribe audio using available methods."""
        logger.info(f"Processing audio file: {audio_path}")
        
        try:
            if not os.path.exists(audio_path):
                error_msg = f"Audio file not found: {audio_path}"
                logger.error(error_msg)
                return {"error": error_msg, "status": 500, "keywords": []}
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            logger.info(f"Audio file size: {file_size} bytes")
            
            if file_size == 0:
                error_msg = "Audio file is empty"
                logger.error(error_msg)
                return {"error": error_msg, "status": 500, "keywords": []}
            
            transcription = None
            method_used = None
            
            # Try Whisper first if available (most accurate)
            if self.whisper_model:
                try:
                    logger.info("Attempting Whisper transcription...")
                    transcription = self.transcribe_with_whisper(audio_path)
                    method_used = "whisper"
                    logger.info(f"Whisper transcription successful: {transcription[:100]}...")
                except Exception as e:
                    logger.warning(f"Whisper failed: {e}")
            
            # Try Wav2Vec2 if Whisper failed
            if transcription is None and self.wav2vec2_available:
                try:
                    logger.info("Attempting Wav2Vec2 transcription...")
                    transcription = self.transcribe_with_wav2vec2(audio_path)
                    method_used = "wav2vec2"
                    logger.info(f"Wav2Vec2 transcription successful: {transcription[:100]}...")
                except Exception as e:
                    logger.warning(f"Wav2Vec2 failed: {e}")
            
            # Fallback to SpeechRecognition if other methods failed
            if transcription is None:
                try:
                    logger.info("Starting transcription with SpeechRecognition...")
                    transcription = self.transcribe_with_speech_recognition(audio_path)
                    method_used = "speech_recognition"
                    logger.info(f"SpeechRecognition successful: {transcription[:100]}...")
                except Exception as e:
                    error_msg = f"All speech recognition methods failed: {str(e)}"
                    logger.error(error_msg)
                    return {"error": error_msg, "status": 500, "keywords": []}
            
            if not transcription or transcription.strip() == "":
                error_msg = "No speech detected in audio"
                logger.warning(error_msg)
                return {"error": error_msg, "status": 500, "keywords": []}
            
            logger.info(f"Audio transcribed successfully using {method_used}: {transcription[:50]}...")
            return {
                "result": transcription,
                "method": method_used,
                "model": "audio_agent",
                "tokens_used": len(transcription.split()),
                "cost_estimate": 0.0,
                "status": 200,
                "keywords": ["audio", "transcription", method_used]
            }
            
        except Exception as e:
            error_msg = f"Unexpected error in process_audio: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": error_msg, "status": 500, "keywords": []}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard interface for AudioAgent.
        
        Args:
            input_data: Dictionary containing input parameters
                Required keys:
                    - input: The audio file path to process
                    - task_id: Unique identifier for the task (optional)
                Optional keys:
                    - input_type: Type of input (default: audio)
                    - model: Model to use for processing (default: edumentor_agent)
                    - tags: List of tags for the task
                    - live_feed: Live feed data (for streaming)
                    - retries: Number of retry attempts (default: 3)
        
        Returns:
            Dictionary with processing results in standardized format
        """
        task_id = input_data.get('task_id', str(uuid.uuid4()))
        input_path = input_data.get('input', '')
        input_type = input_data.get('input_type', 'audio')
        model = input_data.get('model', 'edumentor_agent')
        tags = input_data.get('tags', [])
        live_feed = input_data.get('live_feed', '')
        retries = input_data.get('retries', 3)
        
        logger.info(f"AudioAgent processing task {task_id}, input: {input_path}")
        result = self.process_audio(input_path, task_id)
        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, input_path, result, "audio_agent", model, reward)
        return result

if __name__ == "__main__":
    agent = AudioAgent()
    test_input = "test_audio.wav"
    
    # Test with new interface
    input_data = {
        "input": test_input,
        "input_type": "audio"
    }
    result = agent.run(input_data)
    print(result)