"""
Eyera Global Application Launcher
Main entrypoint to run the Eyera Central Controller & Mode Switcher.
"""

import os
import sys

# Ensure root is in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.chdir(root_dir)

# Ensure execution under project virtualenv if available
venv_python = os.path.join(root_dir, "venv", "Scripts", "python.exe")
if os.path.exists(venv_python):
    curr_exe = os.path.abspath(sys.executable).lower()
    target_exe = os.path.abspath(venv_python).lower()
    if curr_exe != target_exe:
        print(f"[Launcher] Switching to project virtual environment: {venv_python}")
        import subprocess
        result = subprocess.run([venv_python] + sys.argv, cwd=root_dir)
        sys.exit(result.returncode)

from app.launcher.server import run_server

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port=port)
