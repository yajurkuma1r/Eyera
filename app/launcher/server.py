"""
Eyera Global Launcher Server (FastAPI)
Acts as the central web controller, router, and process coordinator.
"""

import os
import sys
import webbrowser
import threading
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.launcher.process_manager import process_manager, PORTS

app = FastAPI(title="Eyera Global Launcher", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAUNCHER_HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")


class ModeRequest(BaseModel):
    mode: str


class VoiceCommandRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def get_home():
    """Serves the Eyera Home & Launcher frontend."""
    if os.path.exists(LAUNCHER_HTML_PATH):
        with open(LAUNCHER_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found in app/launcher</h3>"


@app.get("/api/status")
def get_status():
    """Returns current active mode, process status, and frontend URL."""
    return process_manager.get_status()


@app.post("/api/mode/start")
def start_mode(req: ModeRequest):
    """
    Enforces mode exclusivity: stops any active mode and starts requested mode.
    Valid modes: 'object_ai', 'navigation', 'vision_assistance'.
    """
    return process_manager.start_mode(req.mode)


@app.post("/api/mode/stop")
def stop_mode():
    """Stops any currently active mode and returns to Home."""
    return process_manager.stop_active_mode()


@app.post("/api/voice-command")
def handle_voice_command(req: VoiceCommandRequest):
    """
    Parses a voice command transcript and triggers the requested mode or stops it.
    Commands: 'trigger navigation', 'trigger object AI', 'trigger vision assistance', 'stop'.
    """
    return process_manager.handle_voice_command(req.text)


@app.post("/api/voice-listen")
def voice_listen_hardware():
    """
    Fallback endpoint that listens directly on the host machine's microphone
    using Eyera's SpeechService if Web Speech API is not available in browser.
    """
    try:
        from app.services.audio.speech_service import SpeechService
        stt = SpeechService()
        text = stt.listen()
        if text:
            result = process_manager.handle_voice_command(text)
            result["text"] = text
            return result
        return {
            "text": None,
            "command": "UNKNOWN",
            "message": "No speech recognized.",
            "state": process_manager.get_status()
        }
    except Exception as e:
        return {
            "error": str(e),
            "state": process_manager.get_status()
        }


def open_browser_later(url: str, delay: float = 1.5):
    time.sleep(delay)
    print(f"\n[Eyera Launcher] Opening Central Controller in browser: {url}")
    webbrowser.open(url)


def run_server(host: str = "0.0.0.0", port: int = PORTS["launcher"], auto_open: bool = True):
    """Runs the Eyera Launcher FastAPI server."""
    if auto_open:
        threading.Thread(target=open_browser_later, args=(f"http://localhost:{port}",), daemon=True).start()

    print("==================================================")
    print("        EYERA GLOBAL LAUNCHER & CONTROLLER        ")
    print("==================================================")
    print(f"Controller URL: http://localhost:{port}")
    print("Modes Available:")
    print("  1. Object AI (Approach Detector)")
    print("  2. Navigation (4-Screen Synced Hub)")
    print("  3. Vision Assistance (Multimodal AI & OCR)")
    print("==================================================")

    uvicorn.run("app.launcher.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    port = PORTS["launcher"]
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port=port)
