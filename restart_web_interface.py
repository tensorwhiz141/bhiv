#!/usr/bin/env python3
"""
Script to restart the web interface with updated timeout settings.
"""

import subprocess
import sys
import time
import os
import signal
import psutil

def find_and_kill_process(port=8003):
    """Find and kill any process running on the specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if f':{port}' in cmdline or f'port {port}' in cmdline:
                    print(f"Killing process {proc.info['pid']}: {proc.info['name']}")
                    proc.kill()
                    proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        print(f"Error finding processes: {e}")

def restart_web_interface():
    """Restart the web interface with new timeout settings."""
    print("🔄 Restarting BHIV Core Web Interface with Extended Timeouts...")
    
    # Kill existing processes
    print("🛑 Stopping existing web interface...")
    find_and_kill_process(8003)
    time.sleep(2)
    
    # Set environment variables for extended timeouts
    env = os.environ.copy()
    env.update({
        'DEFAULT_TIMEOUT': '120',
        'IMAGE_PROCESSING_TIMEOUT': '180',
        'AUDIO_PROCESSING_TIMEOUT': '240',
        'PDF_PROCESSING_TIMEOUT': '150',
        'LLM_TIMEOUT': '120',
        'FILE_UPLOAD_TIMEOUT': '300'
    })
    
    print("⚙️ Starting web interface with extended timeouts:")
    print("   - Default timeout: 120s")
    print("   - Image processing: 180s")
    print("   - Audio processing: 240s")
    print("   - PDF processing: 150s")
    print("   - File upload: 300s")
    
    try:
        # Start the web interface
        cmd = [sys.executable, "integration/web_interface.py"]
        process = subprocess.Popen(cmd, env=env)
        
        print(f"🚀 Web interface started with PID: {process.pid}")
        print("🌐 Dashboard available at: http://localhost:8003/dashboard")
        print("📊 Standalone dashboard: dashboard_standalone.html")
        print("\n✅ Extended timeouts applied successfully!")
        print("\nPress Ctrl+C to stop the web interface...")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping web interface...")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("✅ Web interface stopped.")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

if __name__ == "__main__":
    restart_web_interface()
