import asyncio
import base64
import cv2
import json
import math
import os
import numpy as np
import time
import threading
import requests
import webbrowser
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Eyera Navigation Mode", version="2.0.0")

# Import Eyera TTS and Vision Services
from app.services.audio.tts_service import TTSService
from app.services.vision.vision_service import VisionService

# Helper: calculate distance between two coordinates using Haversine
def get_haversine_distance(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Helper: Extract clean destination from voice inputs
def extract_destination(text: str) -> str:
    text = text.lower().strip()
    # Strip common ending punctuation
    for char in [".", ",", "?", "!"]:
        text = text.replace(char, "")
        
    prefixes = [
        "navigate me to the nearest",
        "navigate me to the",
        "navigate me to",
        "navigate to the nearest",
        "navigate to the",
        "navigate to",
        "take me to the nearest",
        "take me to the",
        "take me to",
        "find the nearest",
        "find the",
        "find nearest",
        "find a nearby",
        "find a",
        "find",
        "go to the nearest",
        "go to the",
        "go to",
        "search for the nearest",
        "search for the",
        "search for",
        "directions to the nearest",
        "directions to the",
        "directions to",
    ]
    for p in prefixes:
        if text.startswith(p):
            cleaned = text[len(p):].strip()
            if cleaned:
                return cleaned.title()
                
    return text.title()

# Class to hold single source of truth for the Navigation session
class NavigationSession:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Navigation parameters
        self.active = False
        self.destination = None
        self.origin = "Current Location"
        self.start_coords = [77.2295, 28.6129]  # Default Delhi India Gate [lon, lat]
        self.dest_coords = None
        
        # Route geometry & instructions
        self.route_geometry = []
        self.route_instructions = []
        self.total_distance = 0.0
        self.total_duration = 0.0
        
        # Current real-time states
        self.current_position = [77.2295, 28.6129]
        self.current_instruction = "Ready to navigate."
        self.distance_to_next = 0.0
        self.elapsed_time = 0.0
        self.start_time = 0.0
        self.speed_mps = 1.4  # Walking speed (~5 km/h)
        
        # Audio guidance track
        self.spoken_milestones = set()
        
        # Vision states
        self.warnings = []
        self.annotated_frame = ""
        self.vision_loaded = False
        self.vision_status = "Loading models..."
        self.camera_status = "Checking camera..."
        
        # Performance optimizations
        self.vision_mode = "smooth"  # "smooth" (high-FPS raw) or "overlay" (annotated)
        self.next_frame_to_process = None


# Instantiate global navigation session and services
session = NavigationSession()
tts_service = TTSService()

# Queue of speech messages to broadcast to connected websockets
pending_speech = []
lock_pending_speech = threading.Lock()

def tts_callback(message):
    with lock_pending_speech:
        pending_speech.append(message)

tts_service.on_speak_callback = tts_callback

# WebSocket tracking variables
connected_websockets = set()
lock_websockets = threading.Lock()

# -------------------------------------------------------------
# TomTom API helpers
# -------------------------------------------------------------
def tomtom_search(query: str, start_lat: float, start_lon: float) -> dict:
    """
    Search destination or nearby POIs using TomTom Fuzzy Search.
    Uses start coordinates as biasing centers.
    """
    api_key = os.getenv("TOMTOM_API_KEY", "").strip()
    if not api_key:
        print("[TomTom] API key missing. Generating mock search location.")
        return {
            "name": f"Mock {query.title()}",
            "lat": start_lat + 0.0015,
            "lon": start_lon + 0.0020
        }
        
    try:
        url = f"https://api.tomtom.com/search/2/search/{requests.utils.quote(query)}.json"
        params = {
            "key": api_key,
            "lat": start_lat,
            "lon": start_lon,
            "radius": 10000,
            "language": "en-GB",
            "limit": 5
        }
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        results = data.get("results", [])
        if not results:
            print(f"[TomTom] Fuzzy search returned zero results for: {query}")
            return None
            
        first = results[0]
        poi_name = first.get("poi", {}).get("name")
        address = first.get("address", {}).get("freeformAddress", "")
        
        if poi_name:
            name = f"{poi_name} ({address})" if address else poi_name
        else:
            name = address if address else query.title()
            
        pos = first.get("position", {})
        return {
            "name": name,
            "lat": pos.get("lat"),
            "lon": pos.get("lon")
        }
    except Exception as e:
        print(f"[TomTom Error] Search query failed ({e}). Generating mock destination.")
        return {
            "name": f"Mock {query.title()} (Search Failed)",
            "lat": start_lat + 0.0015,
            "lon": start_lon + 0.0020
        }

def get_fallback_route(start_lon: float, start_lat: float, dest_name="Destination") -> dict:
    """
    Creates a simulated route when TomTom API fails or is not configured.
    Walks northeast in a simple step pattern.
    """
    points_count = 12
    step_lon = 0.00015
    step_lat = 0.00012
    
    geometry = []
    for i in range(points_count):
        geometry.append([start_lon + i * step_lon, start_lat + i * step_lat])
        
    instructions = [
        {"instruction": "Proceed north-east on walking path.", "distance": 0.0, "duration": 0.0, "position": geometry[0]},
        {"instruction": "Walk straight for 40 meters.", "distance": 40.0, "duration": 28.0, "position": geometry[3]},
        {"instruction": "In 80 meters, turn right at the corner.", "distance": 80.0, "duration": 57.0, "position": geometry[7]},
        {"instruction": "Proceed for 30 meters to your destination.", "distance": 110.0, "duration": 78.0, "position": geometry[10]},
    ]
    
    total_dist = 0.0
    for i in range(1, len(geometry)):
        total_dist += get_haversine_distance(geometry[i-1], geometry[i])
        
    return {
        "geometry": geometry,
        "instructions": instructions,
        "total_distance": total_dist,
        "total_duration": total_dist / 1.4,
        "destination_name": dest_name
    }

def tomtom_routing(start_lat: float, start_lon: float, dest_lat: float, dest_lon: float) -> dict:
    """
    Generates walking directions and coordinate geometry using TomTom Pedestrian Route Planner.
    """
    api_key = os.getenv("TOMTOM_API_KEY", "").strip()
    if not api_key:
        print("[TomTom] API key missing. Generating fallback simulation route.")
        return get_fallback_route(start_lon, start_lat)
        
    try:
        locations = f"{start_lat},{start_lon}:{dest_lat},{dest_lon}"
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{locations}/json"
        params = {
            "key": api_key,
            "travelMode": "pedestrian",
            "instructionsType": "text",
            "language": "en-GB"
        }
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        
        if not data.get("routes"):
            raise Exception("No routes found in response payload.")
            
        route = data["routes"][0]
        points = route["legs"][0]["points"]
        geometry = [[pt["longitude"], pt["latitude"]] for pt in points]
        
        instructions = []
        raw_instructions = route["legs"][0].get("instructions", [])
        for inst in raw_instructions:
            instructions.append({
                "instruction": inst.get("message"),
                "distance": float(inst.get("routeOffsetInMeters", 0.0)),
                "duration": float(inst.get("travelTimeInSeconds", 0.0)),
                "position": [inst.get("point", {}).get("longitude"), inst.get("point", {}).get("latitude")]
            })
            
        summary = route["summary"]
        total_distance = summary.get("lengthInMeters", 0.0)
        total_duration = summary.get("travelTimeInSeconds", 0.0)
        
        return {
            "geometry": geometry,
            "instructions": instructions,
            "total_distance": total_distance,
            "total_duration": total_duration
        }
    except Exception as e:
        print(f"[TomTom Error] Routing failed ({e}). Falling back to simulated route.")
        return get_fallback_route(start_lon, start_lat)

# -------------------------------------------------------------
# Background Loops
# -------------------------------------------------------------
def run_vision_ml_loop(session_obj: NavigationSession, tts: TTSService):
    """
    Asynchronous ML Inference loop.
    Processes frame buffer with YOLO/MiDaS models in a separate background thread.
    Detections and warnings are produced here, preventing locks on websocket event loops.
    """
    while not session_obj.vision_loaded:
        time.sleep(0.5)
        
    vision_service = None
    if session_obj.vision_status == "Ready":
        try:
            vision_service = VisionService()
            print("[Vision ML Thread] VisionService pipeline loaded successfully.")
        except Exception as e:
            session_obj.vision_status = f"Error: {e}"
            print(f"[Vision ML Thread Error] VisionService init failed: {e}")
            
    while True:
        try:
            frame = None
            with session_obj.lock:
                if session_obj.next_frame_to_process is not None:
                    frame = session_obj.next_frame_to_process
                    session_obj.next_frame_to_process = None
                    
            if frame is None:
                time.sleep(0.02)
                continue
                
            warnings = []
            annotated = frame
            
            # Execute vision processing asynchronously
            if session_obj.vision_status == "Ready" and vision_service:
                try:
                    annotated, vision_warnings = vision_service.process_frame(frame)
                    warnings = vision_warnings
                except Exception as ex:
                    print(f"[Vision Pipeline Thread Error]: {ex}")
                    
            # Handle priority safety alerts speaking
            if warnings and session_obj.active:
                severity_map = {"approaching": 0, "very_close": 1, "ahead": 2}
                valid_warnings = [w for w in warnings if "type" in w and "message" in w]
                if valid_warnings:
                    valid_warnings.sort(key=lambda w: (severity_map.get(w["type"], 3), -w.get("depth", 0)))
                    critical_warning = valid_warnings[0]
                    tts.speak(critical_warning["message"], priority=1)
                    
            # Update session variables
            with session_obj.lock:
                session_obj.warnings = warnings
                if session_obj.vision_mode == "overlay":
                    # In overlay mode, encode annotated frames directly
                    _, buffer = cv2.imencode('.jpg', annotated)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    session_obj.annotated_frame = f"data:image/jpeg;base64,{frame_base64}"
                    
            time.sleep(0.01)
            
        except Exception as e:
            print(f"[Vision ML Loop Exception]: {e}")
            time.sleep(1)

def run_camera_loop(session_obj: NavigationSession, tts: TTSService):
    """
    Background worker that runs the camera feed.
    - If camera index 0 is open, it captures raw frames at high FPS (20-25 FPS).
    - If camera is absent, it renders a high-tech mock scanning radar with target alerts.
    - Dispatches frames to the background ML thread for warning evaluations.
    """
    cap = cv2.VideoCapture(0)
    camera_active = cap.isOpened()
    
    if camera_active:
        session_obj.camera_status = "Connected"
        print("[Navigation System] Camera index 0 connected. Live stream active.")
    else:
        session_obj.camera_status = "Simulated"
        print("[Navigation System] Camera index 0 unavailable. Starting radar simulation.")
        cap.release()
        
    last_process_push = 0.0
    
    while True:
        try:
            current_time = time.time()
            frame = None
            
            if camera_active:
                ret, raw_frame = cap.read()
                if ret:
                    # Resize to 480x360 to dramatically increase frame speed
                    frame = cv2.resize(raw_frame, (480, 360))
                else:
                    camera_active = False
                    session_obj.camera_status = "Simulated"
                    cap.release()
                    
            if frame is None:
                # Generate custom radar scanner feed
                frame = generate_mock_frame(session_obj, current_time)
                if session_obj.active:
                    sim_warnings = get_simulated_warnings(session_obj)
                    with session_obj.lock:
                        session_obj.warnings = sim_warnings
                else:
                    with session_obj.lock:
                        session_obj.warnings = []
                        
                # Encode radar frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                with session_obj.lock:
                    session_obj.annotated_frame = f"data:image/jpeg;base64,{frame_base64}"
            else:
                # 1. Push frame reference to background ML processor
                push_interval = 0.35 if session_obj.vision_mode == "smooth" else 0.15
                if current_time - last_process_push > push_interval:
                    with session_obj.lock:
                        if session_obj.next_frame_to_process is None:
                            session_obj.next_frame_to_process = frame
                            last_process_push = current_time
                            
                # 2. If in smooth mode, compile and update base64 stream at 25 FPS
                if session_obj.vision_mode == "smooth":
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    with session_obj.lock:
                        session_obj.annotated_frame = f"data:image/jpeg;base64,{frame_base64}"
                        
            time.sleep(0.04)  # ~25 FPS loop pace
            
        except Exception as e:
            print(f"[Camera Background Loop Error]: {e}")
            time.sleep(1)
            
    if camera_active:
        cap.release()


def generate_mock_frame(session_obj: NavigationSession, current_time: float):
    """
    Renders a stunning high-tech radar overlay on a canvas frame.
    Displays target rings, scanning line, status labels, and mock warning indicator boxes.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (18, 14, 11)  # Premium deep carbon color
    
    # Draw radar search elements
    center = (320, 240)
    cv2.circle(frame, center, 180, (30, 48, 30), 1)
    cv2.circle(frame, center, 120, (20, 32, 20), 1)
    cv2.circle(frame, center, 60, (10, 16, 10), 1)
    
    # Sweep line
    angle = (current_time * 90) % 360
    rad = math.radians(angle)
    x = int(320 + 180 * math.cos(rad))
    y = int(240 - 180 * math.sin(rad))
    cv2.line(frame, center, (x, y), (0, 210, 0), 2)
    
    # Render indicators for active warnings
    if session_obj.active and session_obj.warnings:
        for w in session_obj.warnings:
            w_type = w.get("type", "ahead")
            obj = w.get("object", "Obstacle")
            
            offset = int(math.sin(current_time * 4) * 15)
            if w_type == "approaching":
                size = int(60 + (current_time * 12) % 60)
                color = (0, 150, 255)  # Orange
                cv2.rectangle(frame, (320 - size//2 + offset, 240 - size//2), (320 + size//2 + offset, 240 + size//2), color, 2)
                cv2.putText(frame, f"{obj.upper()} APPROACHING", (320 - size//2 + offset, 240 - size//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            elif w_type == "very_close":
                size = 140
                color = (0, 0, 255)  # Red
                cv2.rectangle(frame, (320 - size//2, 240 - size//2), (320 + size//2, 240 + size//2), color, 3)
                cv2.putText(frame, "CRITICAL: VERY CLOSE", (320 - size//2, 240 - size//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                size = 80
                color = (0, 255, 255)  # Yellow
                cv2.rectangle(frame, (320 - size//2 + offset, 240 - size//2), (320 + size//2 + offset, 240 + size//2), color, 2)
                cv2.putText(frame, f"{obj} Ahead", (320 - size//2 + offset, 240 - size//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
                
    # Frame overlays
    cv2.putText(frame, "EYERA HUD RADAR (MOCK CAM)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)
    cv2.putText(frame, f"GPS: {session_obj.current_position[1]:.5f}, {session_obj.current_position[0]:.5f}", (20, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1)
    status_msg = f"CAMERA: {session_obj.camera_status} | VISION PIPELINE: {session_obj.vision_status}"
    cv2.putText(frame, status_msg, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0) if session_obj.camera_status=="Connected" else (0, 160, 240), 1)
    
    return frame

def get_simulated_warnings(session_obj: NavigationSession):
    """
    Yields mock warnings mapping to simulated walk time.
    Keeps the Screen 4 interface lively and tests priority preemption.
    """
    t = int(session_obj.elapsed_time) % 65
    
    if 5 <= t <= 12:
        return [{
            "type": "approaching",
            "object": "Car",
            "depth": 340.0 + (t - 5) * 15,
            "message": "Car approaching"
        }]
    elif 20 <= t <= 27:
        return [{
            "type": "ahead",
            "object": "Pole",
            "depth": 315.0,
            "message": "Pole ahead"
        }]
    elif 38 <= t <= 45:
        return [{
            "type": "approaching",
            "object": "Person",
            "depth": 330.0 + (t - 38) * 12,
            "message": "Person approaching"
        }]
    elif 52 <= t <= 56:
        return [{
            "type": "very_close",
            "object": "Obstacle",
            "depth": 520.0,
            "message": "Obstacle very close"
        }]
    return []

def run_navigation_simulation(session_obj: NavigationSession, tts: TTSService):
    """
    Simulates walking navigation. Moves the coordinates along the geometry,
    evaluates instruction triggers, and plays walking directions via TTSService (Priority 2).
    """
    session_obj.start_time = time.time()
    session_obj.elapsed_time = 0.0
    session_obj.spoken_milestones = set()
    
    # Speak activation
    tts.speak(f"Starting navigation to {session_obj.destination}.", priority=2)
    
    if session_obj.route_instructions:
        first_instruction = session_obj.route_instructions[0]["instruction"]
        tts.speak(first_instruction, priority=2)
        session_obj.current_instruction = first_instruction
        
    last_tick = time.time()
    
    while session_obj.active:
        time.sleep(0.1)
        now = time.time()
        dt = now - last_tick
        last_tick = now
        
        with session_obj.lock:
            if not session_obj.active:
                break
                
            session_obj.elapsed_time += dt
            distance_walked = session_obj.elapsed_time * session_obj.speed_mps
            
            geom = session_obj.route_geometry
            if not geom:
                session_obj.active = False
                break
                
            # Compute route legs
            total_route_distance = 0.0
            leg_distances = [0.0]
            for i in range(1, len(geom)):
                d_seg = get_haversine_distance(geom[i-1], geom[i])
                total_route_distance += d_seg
                leg_distances.append(total_route_distance)
                
            if distance_walked >= total_route_distance:
                # Arrived at destination
                session_obj.current_position = geom[-1]
                session_obj.current_instruction = "You have reached your destination."
                session_obj.distance_to_next = 0.0
                session_obj.active = False
                tts.speak("You have reached your destination.", priority=2)
                break
                
            # Find interpolated position
            curr_pos = geom[0]
            for i in range(1, len(leg_distances)):
                if distance_walked <= leg_distances[i]:
                    d_prev = leg_distances[i-1]
                    d_curr = leg_distances[i]
                    segment_len = d_curr - d_prev
                    t = (distance_walked - d_prev) / segment_len if segment_len > 0 else 0.0
                    
                    lon_prev, lat_prev = geom[i-1]
                    lon_curr, lat_curr = geom[i]
                    
                    lon = lon_prev + t * (lon_curr - lon_prev)
                    lat = lat_prev + t * (lat_curr - lat_prev)
                    curr_pos = [lon, lat]
                    break
            session_obj.current_position = curr_pos
            
            # Match current navigation instruction based on progress
            insts = session_obj.route_instructions
            if insts:
                k = 0
                for idx, inst in enumerate(insts):
                    if inst["distance"] <= distance_walked:
                        k = idx
                    else:
                        break
                        
                session_obj.current_instruction = insts[k]["instruction"]
                
                # Check upcoming directions prompts
                if k + 1 < len(insts):
                    next_inst = insts[k+1]
                    dist_to_next = next_inst["distance"] - distance_walked
                    session_obj.distance_to_next = dist_to_next
                    next_msg = next_inst["instruction"]
                    
                    # 1. Warning milestone: ~20 meters before maneuver
                    if 18.0 <= dist_to_next <= 22.0:
                        mid = f"approach_{k+1}"
                        if mid not in session_obj.spoken_milestones:
                            tts.speak(f"In 20 metres, {next_msg}", priority=2)
                            session_obj.spoken_milestones.add(mid)
                            
                    # 2. Action milestone: ~3 meters before maneuver
                    elif 1.0 <= dist_to_next <= 4.0:
                        mid = f"turn_{k+1}"
                        if mid not in session_obj.spoken_milestones:
                            tts.speak(f"{next_msg} now.", priority=2)
                            session_obj.spoken_milestones.add(mid)
                else:
                    # Final arrival alert
                    dist_to_next = total_route_distance - distance_walked
                    session_obj.distance_to_next = dist_to_next
                    if 1.0 <= dist_to_next <= 4.0:
                        mid = "arriving_soon"
                        if mid not in session_obj.spoken_milestones:
                            tts.speak("Destination ahead.", priority=2)
                            session_obj.spoken_milestones.add(mid)
            else:
                session_obj.current_instruction = "Proceed straight toward destination."
                session_obj.distance_to_next = total_route_distance - distance_walked

def start_navigation_session(destination: str, origin: str, start_lat: float, start_lon: float):
    """
    Orchestrates the startup of a new navigation query:
    1. Geocodes destination using TomTom.
    2. Computes the Pedestrian Route from TomTom APIs.
    3. Runs the walk simulator in a background thread.
    """
    global nav_thread
    
    with session.lock:
        if session.active:
            print("[Navigation System] Stopping active session...")
            session.active = False
            time.sleep(0.2)
            
        session.active = True
        session.destination = destination
        session.origin = origin
        session.start_coords = [start_lon, start_lat]
        session.current_position = [start_lon, start_lat]
        
    print(f"[Navigation System] Fetching route: '{origin}' -> '{destination}'")
    dest_data = tomtom_search(destination, start_lat, start_lon)
    
    if dest_data:
        session.dest_coords = [dest_data["lon"], dest_data["lat"]]
        session.destination = dest_data["name"]
        
        # Calculate pedestrian routes
        route_data = tomtom_routing(start_lat, start_lon, dest_data["lat"], dest_data["lon"])
        session.route_geometry = route_data["geometry"]
        session.route_instructions = route_data["instructions"]
        session.total_distance = route_data["total_distance"]
        session.total_duration = route_data["total_duration"]
    else:
        # Fallback if geocoding fails
        print("[Navigation System] Destination not found. Routing with simulated markers.")
        session.dest_coords = [start_lon + 0.0018, start_lat + 0.0014]
        route_data = get_fallback_route(start_lon, start_lat, destination)
        session.route_geometry = route_data["geometry"]
        session.route_instructions = route_data["instructions"]
        session.total_distance = route_data["total_distance"]
        session.total_duration = route_data["total_duration"]
        
    # Launch navigation thread
    nav_thread = threading.Thread(
        target=run_navigation_simulation,
        args=(session, tts_service),
        daemon=True
    )
    nav_thread.start()

# -------------------------------------------------------------
# FastAPI HTTP Endpoints
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def get_index():
    """
    Serves the primary synchronized 4-screen dashboard interface.
    """
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html is missing inside navigation service folder.</h3>"

@app.get("/api/config")
def get_config():
    """
    Exposes setup metadata (like TomTom API key token and model status).
    """
    return {
        "tomtom_key": os.getenv("TOMTOM_API_KEY", "").strip(),
        "vision_status": session.vision_status,
        "camera_status": session.camera_status
    }

@app.get("/api/state")
def get_state():
    """
    REST endpoint to poll current session variables.
    """
    with session.lock:
        return {
            "active": session.active,
            "destination": session.destination,
            "origin": session.origin,
            "start_coords": session.start_coords,
            "dest_coords": session.dest_coords,
            "current_position": session.current_position,
            "current_instruction": session.current_instruction,
            "distance_to_next": session.distance_to_next,
            "elapsed_time": session.elapsed_time,
            "total_distance": session.total_distance,
            "total_duration": session.total_duration,
            "warnings": session.warnings,
            "vision_status": session.vision_status,
            "camera_status": session.camera_status
        }

# -------------------------------------------------------------
# WebSocket Controllers
# -------------------------------------------------------------
def get_serialized_state():
    """
    Compiles state dictionary including base64 frame stream.
    """
    with session.lock:
        return {
            "active": session.active,
            "destination": session.destination,
            "origin": session.origin,
            "start_coords": session.start_coords,
            "dest_coords": session.dest_coords,
            "route": session.route_geometry,
            "current_position": session.current_position,
            "current_instruction": session.current_instruction,
            "distance_to_next": session.distance_to_next,
            "elapsed_time": session.elapsed_time,
            "total_distance": session.total_distance,
            "total_duration": session.total_duration,
            "warnings": session.warnings,
            "frame": session.annotated_frame,
            "vision_status": session.vision_status,
            "camera_status": session.camera_status,
            "vision_mode": session.vision_mode
        }


@app.websocket("/ws/navigation")
async def websocket_navigation(websocket: WebSocket):
    """
    Synchronizes screens over a single shared WebSocket channel.
    Accepts voice transcript commands and buttons triggers.
    """
    await websocket.accept()
    with lock_websockets:
        connected_websockets.add(websocket)
        
    print(f"[WebSocket] Client connected. Active views: {len(connected_websockets)}")
    
    try:
        # Instantly push current state on connect
        await websocket.send_json(get_serialized_state())
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action")
            
            if action == "start_nav":
                dest = message.get("destination", "India Gate")
                orig = message.get("origin", "Current Location")
                lat = float(message.get("start_lat", 28.6129))
                lon = float(message.get("start_lon", 77.2295))
                start_navigation_session(dest, orig, lat, lon)
                
            elif action == "stop_nav":
                with session.lock:
                    session.active = False
                tts_service.speak("Navigation stopped.", priority=2)
                
            elif action == "voice_input":
                speech = message.get("text", "")
                if speech:
                    dest = extract_destination(speech)
                    print(f"[WebSocket Voice Command] Heard: '{speech}' -> Clean Destination: '{dest}'")
                    lat = float(message.get("start_lat", 28.6129))
                    lon = float(message.get("start_lon", 77.2295))
                    start_navigation_session(dest, "Voice Input", lat, lon)
                    
            elif action == "speak":
                text = message.get("text", "")
                tts_service.speak(text, priority=2)
                
            elif action == "set_vision_mode":
                mode = message.get("mode", "smooth")
                with session.lock:
                    session.vision_mode = mode
                print(f"[WebSocket] Vision Mode toggled to: {mode}")

            elif action == "set_start_coords":
                lat = float(message.get("lat", 28.6129))
                lon = float(message.get("lon", 77.2295))
                with session.lock:
                    session.start_coords = [lon, lat]
                    session.current_position = [lon, lat]
                print(f"[WebSocket] Start Coordinates updated to: {lat}, {lon}")


                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Handle Error]: {e}")
    finally:
        with lock_websockets:
            connected_websockets.discard(websocket)
        print(f"[WebSocket] Client disconnected. Active views remaining: {len(connected_websockets)}")

async def websocket_broadcast_task():
    """
    Independent state broadcaster.
    Streams base64 camera matrices & route status to all connected screens at 10 FPS.
    Also broadcasts voice cues.
    Runs on the main event loop to ensure thread safety with WebSockets.
    """
    global pending_speech
    while True:
        await asyncio.sleep(0.10)  # Throttled to 10 FPS
        
        # Retrieve and clear pending speech messages (independent of socket existence)
        to_speak = []
        with lock_pending_speech:
            if pending_speech:
                to_speak = list(pending_speech)
                pending_speech.clear()
        
        if not connected_websockets:
            continue
            
        # Send any speech messages first
        if to_speak:
            with lock_websockets:
                sockets = list(connected_websockets)
            for msg in to_speak:
                for ws in sockets:
                    try:
                        await ws.send_json({"type": "speech", "text": msg})
                    except Exception:
                        pass
        
        # Send normal state frame
        state = get_serialized_state()
        with lock_websockets:
            sockets = list(connected_websockets)
            
        for ws in sockets:
            try:
                await ws.send_json(state)
            except Exception:
                pass

# -------------------------------------------------------------
# App startup events
# -------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    # Speak greeting
    tts_service.speak("Navigation mode activated.", priority=2)
    tts_service.speak("Where would you like to go?", priority=2)
    
    # Load vision models in background thread so startup isn't blocked
    def load_models_task():
        try:
            print("[Startup] Instantiating VisionService models in background...")
            session.vision_status = "Loading models..."
            # Invoking this triggers YOLO/MiDaS weight loading
            vs = VisionService()
            session.vision_loaded = True
            session.vision_status = "Ready"
            print("[Startup] VisionService models successfully loaded.")
        except Exception as e:
            session.vision_status = f"Error: {e}"
            print(f"[Startup Error] Failed to load VisionService weights: {e}")
            
    threading.Thread(target=load_models_task, daemon=True).start()
    
    # Start the Camera loop
    camera_thread = threading.Thread(
        target=run_camera_loop,
        args=(session, tts_service),
        daemon=True
    )
    camera_thread.start()
    
    # Start the Vision ML loop
    ml_thread = threading.Thread(
        target=run_vision_ml_loop,
        args=(session, tts_service),
        daemon=True
    )
    ml_thread.start()
    
    # Start the WebSocket broadcast loop on the main event loop
    asyncio.create_task(websocket_broadcast_task())
    print("[Startup] Camera, Asynchronous ML threads, and WebSockets broadcaster launched successfully.")


# Run using standard uvicorn entrypoint if executed directly
if __name__ == "__main__":
    webbrowser.open("http://localhost:8000")
    uvicorn.run("app.services.navigation.app:app", host="0.0.0.0", port=8000, reload=False)
