import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from app.services.vision_assistance.concurrency_manager import concurrency_controller
from app.services.vision_assistance.voice_controller import VoiceController

voice_controller = VoiceController()
stop_event = threading.Event()


def voice_loop():
    print("[VisionAssistantServer] Voice listening worker started.")
    while not stop_event.is_set():
        if concurrency_controller.is_listening_enabled and not concurrency_controller.is_busy:
            try:
                command, result = voice_controller.run_once()
                if command in ("EXIT", "DISABLED"):
                    if command == "EXIT":
                        concurrency_controller.disable_listening()
            except Exception as e:
                print(f"[VisionAssistantServer Worker Error] {e}")
        time.sleep(0.2)


class VisionAssistantHTTPRequestHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Suppress routine log output for status polling
        if "GET /api/status" in str(args[0]):
            return
        super().log_message(format, *args)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html_path = os.path.join(os.path.dirname(__file__), "index.html")
            if os.path.exists(html_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "index.html not found")
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_data = concurrency_controller.get_status_dict()
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        if self.path == "/api/enable-listen":
            new_state = concurrency_controller.toggle_listening()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            res = concurrency_controller.get_status_dict()
            res["message"] = f"Listening {'enabled' if new_state else 'disabled'}"
            self.wfile.write(json.dumps(res).encode("utf-8"))
        elif self.path == "/api/disable-listen":
            concurrency_controller.disable_listening()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            res = concurrency_controller.get_status_dict()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    server_address = (host, port)
    httpd = HTTPServer(server_address, VisionAssistantHTTPRequestHandler)
    print(f"\n[VisionAssistantServer] Server running at http://localhost:{port}")
    print("[VisionAssistantServer] Open http://localhost:8000 in your browser to use the 'Enable Listen' control.")

    # Enable listening by default when launching server
    concurrency_controller.enable_listening()

    worker_thread = threading.Thread(target=voice_loop, daemon=True)
    worker_thread.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[VisionAssistantServer] Shutting down server...")
        stop_event.set()
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    start_server(port=port)
