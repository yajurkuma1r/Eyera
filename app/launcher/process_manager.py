"""
Eyera Global Process Manager & Mode Controller

Enforces strict single-mode exclusivity across:
1. Object AI (Approach Detector - experiments/test_approach_detector.py)
2. Navigation (4-screen synced dashboard - app/services/navigation/app.py)
3. Vision Assistance (Multimodal AI & OCR - app/services/vision_assistance/server.py)

Does NOT modify any internal mode code. Operates purely as an external process
supervisor and launcher adapter.
"""

import os
import sys
import time
import socket
import logging
import threading
import subprocess
from typing import Optional, Dict, Any
import requests
import psutil

logger = logging.getLogger("EyeraLauncher")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s")

# Base project directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Ports configuration
PORTS = {
    "launcher": 5000,
    "navigation": 8000,
    "vision_assistance": 8001
}


def get_python_executable() -> str:
    """
    Finds the virtualenv python where all project dependencies
    (edge_tts, torch, ultralytics, supervision, etc.) are installed.
    """
    venv_python = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a TCP port is currently occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def kill_process_tree(proc: subprocess.Popen, timeout: float = 3.0):
    """
    Terminates a subprocess and all of its descendants safely on Windows.
    Frees locks on webcams and TCP sockets.
    """
    if proc is None:
        return

    pid = proc.pid
    logger.info(f"Stopping process tree for PID {pid}...")

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        parent.terminate()
        
        # Wait for graceful termination
        gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.warning(f"Error terminating process PID {pid} via psutil: {e}")
        try:
            proc.kill()
        except Exception:
            pass

    # Secondary Windows taskkill fallback for safety
    if sys.platform.startswith("win"):
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except Exception:
            pass


class ModeProcessManager:
    """
    Central Controller for mode lifecycle, exclusive execution,
    and voice command dispatch.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.active_mode: Optional[str] = None
        self.active_process: Optional[subprocess.Popen] = None
        self.mode_start_time: float = 0.0
        self.status_message: str = "Eyera Ready. Select a mode or speak a command."

        # Start background monitor thread for auto-detecting process exits
        self._monitor_thread = threading.Thread(target=self._process_monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _process_monitor_loop(self):
        """Monitors active child process. Resets state if process exits naturally."""
        while True:
            time.sleep(0.5)
            with self.lock:
                if self.active_process is not None:
                    ret_code = self.active_process.poll()
                    if ret_code is not None:
                        mode = self.active_mode
                        logger.info(f"Mode [{mode}] process exited with code {ret_code}.")
                        self.active_process = None
                        self.active_mode = None
                        self.status_message = f"Mode [{mode}] exited. Returned to Home."

    def get_status(self) -> Dict[str, Any]:
        """Returns the current launcher state."""
        with self.lock:
            running = self.active_process is not None and self.active_process.poll() is None
            
            frontend_url = None
            if running:
                if self.active_mode == "navigation":
                    frontend_url = f"http://localhost:{PORTS['navigation']}"
                elif self.active_mode == "vision_assistance":
                    frontend_url = f"http://localhost:{PORTS['vision_assistance']}"
                elif self.active_mode == "object_ai":
                    frontend_url = "opencv://desktop_window"

            return {
                "active_mode": self.active_mode if running else None,
                "is_running": running,
                "pid": self.active_process.pid if running else None,
                "frontend_url": frontend_url,
                "status_message": self.status_message,
                "uptime": round(time.time() - self.mode_start_time, 1) if running else 0
            }

    def stop_active_mode(self) -> Dict[str, Any]:
        """
        Stops whichever mode is currently running.
        Releases camera and audio resources immediately.
        """
        with self.lock:
            if self.active_process is not None and self.active_process.poll() is None:
                mode_name = self.active_mode
                logger.info(f"Stopping active mode: {mode_name}")
                self.status_message = f"Stopping {mode_name}..."
                
                kill_process_tree(self.active_process)
                self.active_process = None
                self.active_mode = None
                
                # Cooldown to allow Windows to completely release webcam and sockets
                time.sleep(1.0)
                self.status_message = f"Stopped {mode_name}. Eyera Home is active."
                logger.info(f"Active mode {mode_name} successfully stopped.")
            else:
                self.active_mode = None
                self.active_process = None
                self.status_message = "No active mode running. Eyera Home is active."

            return self.get_status()

    def start_mode(self, mode: str) -> Dict[str, Any]:
        """
        Starts the requested mode. Enforces mode exclusivity:
        any existing running mode is stopped first before launching the new one.
        """
        valid_modes = ["object_ai", "navigation", "vision_assistance"]
        if mode not in valid_modes:
            return {"error": f"Invalid mode '{mode}'. Expected one of {valid_modes}"}

        with self.lock:
            # If already running the requested mode, return existing state
            if self.active_mode == mode and self.active_process is not None and self.active_process.poll() is None:
                return self.get_status()

            # 1. Mode Exclusivity: Stop any currently running mode
            if self.active_process is not None and self.active_process.poll() is None:
                logger.info(f"Switching mode: stopping current [{self.active_mode}] before launching [{mode}]")
                self.stop_active_mode()

            # 2. Launch target mode
            self.status_message = f"Launching {mode}..."
            logger.info(f"Launching mode: {mode}")

            python_bin = get_python_executable()
            logger.info(f"Using Python executable: {python_bin}")

            env = os.environ.copy()
            # Ensure root is first in PYTHONPATH
            env["PYTHONPATH"] = f"{PROJECT_ROOT};{env.get('PYTHONPATH', '')}"

            if mode == "object_ai":
                # Approach Detector OpenCV window
                target_script = os.path.join(PROJECT_ROOT, "experiments", "test_approach_detector.py")
                cmd = [python_bin, "-u", target_script]
                
                # Do NOT use subprocess.PIPE without reading: on Windows, pipe buffer fills up and locks OpenCV!
                proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    env=env
                )
                self.active_process = proc
                self.active_mode = "object_ai"
                self.mode_start_time = time.time()
                self.status_message = "Object AI (Approach Detector) active in desktop window."
                logger.info(f"Object AI launched with PID {proc.pid}")

            elif mode == "navigation":
                # Uvicorn FastAPI server on port 8000
                port = PORTS["navigation"]
                cmd = [
                    python_bin,
                    "-m", "uvicorn",
                    "app.services.navigation.app:app",
                    "--host", "0.0.0.0",
                    "--port", str(port),
                    "--log-level", "info"
                ]
                
                proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    env=env
                )
                self.active_process = proc
                self.active_mode = "navigation"
                self.mode_start_time = time.time()
                
                # Wait for port to become responsive (up to 8s)
                ready = self._wait_for_service(f"http://localhost:{port}/api/config", timeout=8.0)
                if ready:
                    self.status_message = f"Navigation active on http://localhost:{port}"
                    logger.info(f"Navigation ready on port {port} (PID {proc.pid})")
                else:
                    logger.warning(f"Navigation started (PID {proc.pid}), still warming up...")
                    self.status_message = f"Navigation starting on http://localhost:{port}..."

            elif mode == "vision_assistance":
                # Vision Assistance HTTP server on port 8001
                port = PORTS["vision_assistance"]
                cmd = [
                    python_bin,
                    "-m", "app.services.vision_assistance.server",
                    str(port)
                ]
                
                proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    env=env
                )
                self.active_process = proc
                self.active_mode = "vision_assistance"
                self.mode_start_time = time.time()
                
                # Wait for port to become responsive (up to 8s)
                ready = self._wait_for_service(f"http://localhost:{port}/api/status", timeout=8.0)
                if ready:
                    self.status_message = f"Vision Assistance active on http://localhost:{port}"
                    logger.info(f"Vision Assistance ready on port {port} (PID {proc.pid})")
                else:
                    logger.warning(f"Vision Assistance started (PID {proc.pid}), warming up...")
                    self.status_message = f"Vision Assistance starting on http://localhost:{port}..."

            return self.get_status()

    def _wait_for_service(self, health_url: str, timeout: float = 8.0) -> bool:
        """Polls until the service endpoint responds or timeout is reached."""
        start = time.time()
        while time.time() - start < timeout:
            # Check if process died during startup
            if self.active_process is not None and self.active_process.poll() is not None:
                logger.error(f"Process died with code {self.active_process.poll()} during startup check.")
                return False
            try:
                res = requests.get(health_url, timeout=0.8)
                if res.status_code in [200, 302, 307]:
                    return True
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def handle_voice_command(self, raw_text: str) -> Dict[str, Any]:
        """
        Parses a voice transcript and executes mode switching or stopping.
        Supported commands at minimum:
        - trigger navigation / start navigation
        - trigger object AI / start object AI
        - trigger vision assistance / start vision assistance
        - stop
        """
        if not raw_text:
            return {"command": "UNKNOWN", "message": "No voice input received.", "state": self.get_status()}

        text = raw_text.lower().strip()
        # Strip trailing punctuation
        for ch in [".", ",", "!", "?", "'", '"']:
            text = text.replace(ch, "")

        logger.info(f"Voice Command Received: '{raw_text}' -> cleaned: '{text}'")

        # 1. STOP COMMAND
        if text == "stop" or "stop" in text.split() or any(term in text for term in [
            "stop mode", "stop navigation", "stop object ai", "stop vision", "exit", "quit", "return to home", "go home"
        ]):
            logger.info("Voice match: STOP")
            state = self.stop_active_mode()
            return {
                "command": "STOP",
                "action": "stopped_active_mode",
                "message": "Stopped active mode. Returned to Eyera Home.",
                "state": state
            }

        # 2. NAVIGATION COMMANDS
        if any(phrase in text for phrase in [
            "trigger navigation", "start navigation", "open navigation", "launch navigation",
            "navigation mode", "navigation"
        ]):
            logger.info("Voice match: START_NAVIGATION")
            state = self.start_mode("navigation")
            return {
                "command": "START_NAVIGATION",
                "action": "launched_navigation",
                "message": "Starting Navigation mode...",
                "state": state
            }

        # 3. OBJECT AI COMMANDS
        if any(phrase in text for phrase in [
            "trigger object ai", "start object ai", "open object ai", "launch object ai",
            "trigger object", "start object", "object ai mode", "object ai",
            "trigger approach detector", "start approach detector", "approach detector"
        ]):
            logger.info("Voice match: START_OBJECT_AI")
            state = self.start_mode("object_ai")
            return {
                "command": "START_OBJECT_AI",
                "action": "launched_object_ai",
                "message": "Starting Object AI (Approach Detector)...",
                "state": state
            }

        # 4. VISION ASSISTANCE COMMANDS
        if any(phrase in text for phrase in [
            "trigger vision assistance", "start vision assistance", "open vision assistance", "launch vision assistance",
            "trigger vision assistant", "start vision assistant", "open vision assistant", "launch vision assistant",
            "vision assistance mode", "vision assistance", "vision assistant", "vision assist"
        ]):
            logger.info("Voice match: START_VISION_ASSISTANCE")
            state = self.start_mode("vision_assistance")
            return {
                "command": "START_VISION_ASSISTANCE",
                "action": "launched_vision_assistance",
                "message": "Starting Vision Assistance mode...",
                "state": state
            }

        return {
            "command": "UNKNOWN",
            "action": "none",
            "message": f"Command not recognized: '{raw_text}'. Say 'trigger navigation', 'trigger object AI', 'trigger vision assistance', or 'stop'.",
            "state": self.get_status()
        }


# Global Singleton Process Manager instance
process_manager = ModeProcessManager()
