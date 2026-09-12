import os
import sys
import webbrowser
import threading
import time
import uvicorn
from dotenv import load_dotenv

# Calculate project root directory (three levels up from this script)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Add project root to sys.path so we can import 'app' module
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Change current working directory to project root so that all relative file paths
# (like .env and model files) resolve correctly.
os.chdir(root_dir)

# Load env variables from .env
load_dotenv()

# Suppress harmless WinError 10054 disconnection noise on Windows
if os.name == "nt":
    from functools import wraps
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        @wraps(_orig_call_connection_lost)
        def _silent_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _silent_call_connection_lost
    except Exception:
        pass

def open_browser():
    """
    Waits briefly for the server to start, then launches the default system browser
    to open the dashboard at http://localhost:8000.
    """
    time.sleep(2.0)
    print("\n[Launcher] Opening Eyera Navigation Hub in browser...")
    webbrowser.open("http://localhost:8000")

def main():
    print("==================================================")
    print("EYERA NEW VOICE-FIRST MULTI-SCREEN NAVIGATION MODE")
    print("==================================================")
    
    # Verify TomTom credentials
    api_key = os.getenv("TOMTOM_API_KEY", "")
    if not api_key:
        print("[WARNING] TOMTOM_API_KEY environment variable is not defined.")
        print("          Please define it in your .env file for real routing searches.")
        print("          Running fallback canvas-based simulation paths.")
        print("==================================================")
    
    # Start auto-opener thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the Uvicorn web server
    uvicorn.run("app.services.navigation.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
