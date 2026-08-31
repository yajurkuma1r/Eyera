import os
import sys
import webbrowser
import threading
import time
import uvicorn
from dotenv import load_dotenv

# Load env variables from .env before starting
load_dotenv()

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
