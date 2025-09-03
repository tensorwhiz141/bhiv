#!/usr/bin/env python3
"""Test script for AudioAgent debugging."""

import os
import sys
from agents.audio_agent import AudioAgent
from utils.logger import get_logger

logger = get_logger(__name__)

def test_audio_agent():
    """Test the AudioAgent directly."""
    audio_file = "test.wav"
    
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found")
        return
    
    print(f"Testing AudioAgent with file: {audio_file}")
    print(f"File size: {os.path.getsize(audio_file)} bytes")
    
    try:
        agent = AudioAgent()
        print("AudioAgent initialized successfully")
        
        result = agent.run(audio_file, input_type="audio")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error testing AudioAgent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_agent()