"""
Eyera Vision Assistance Live Runner

Live Vision Assistant Pipeline:
1. User speaks -> Speech-to-Text
2. Intent / Command Detection -> CommandService
3. Fusion Layer -> Determines required visual capabilities
4. Live Camera -> Captures CURRENT live webcam frame
5. Vision Processing -> Selective YOLO / MiDaS / Tesseract OCR on live frame
6. Structured Visual Output -> Factual observations (no hardcoding)
7. Fusion Layer -> Formats factual visual context + original query
8. LLM -> Generates dynamic natural response
9. Edge TTS -> Audio output through earpiece
"""

import sys
import time
from app.services.vision_assistance import (
    AssistantService,
    STTService,
    CommandService,
    CameraService,
    FusionService,
)
from app.services.vision_assistance.concurrency_manager import concurrency_controller


def run_live_vision_assistant():
    print("=" * 65)
    print("        EYERA LIVE VISION ASSISTANT RUNNER             ")
    print("  (Speech/STT + Live Camera + YOLO/MiDaS/OCR + LLM)    ")
    print("=" * 65)

    print("\n[Init] Initializing Live Vision Assistance Services...")
    assistant = AssistantService()
    stt = STTService()
    command_service = CommandService()
    camera = CameraService()
    fusion = FusionService()
    concurrency_controller.enable_listening()

    print("\n[Ready] Eyera Live Vision Assistant Active.")
    print("Speak or type any natural vision command:")
    print("  - 'Read the menu' / 'What does this sign say?'")
    print("  - 'What is in front of me?' / 'Where is the chair?'")
    print("  - 'Is it safe to cross?' / 'Describe my surroundings'")
    print("  - 'What time is it?' / 'What's the date?' / 'What day is it?'")
    print("  - 'Is there any festival today?'")
    print("  - Type 'exit' or 'quit' to stop\n")

    while True:
        try:
            print("-" * 65)
            with concurrency_controller.acquire_operation("CLI_PIPELINE"):
                # 1. 🎤 User Speaks -> Speech-to-Text
                query = stt.listen()
                if not query:
                    continue

            clean_query = query.strip()
            if clean_query.lower() in ("exit", "quit"):
                print("[Eyera] Shutting down Live Vision Assistant. Goodbye!")
                break

            print(f"\n[STT] User command: {clean_query}")

            # 2. 🎯 Intent / Command Detection
            command = command_service.process_command(clean_query)
            print(f"[COMMAND] Intent: {command}")

            # 3. 🧠 Fusion Layer Determines Required Visual Capabilities
            reqs = fusion.determine_requirements(command=command, user_query=clean_query)
            need_ocr = reqs.get("need_ocr", False)
            need_objects = reqs.get("need_objects", True)
            need_depth = reqs.get("need_depth", True)
            print(f"[FUSION] Required capability: {reqs.get('capability')}")

            # 3b. Non-vision commands (time/date/day/festival/general knowledge/unknown)
            # never touch the camera - answer directly from clock/calendar or LLM general knowledge.
            if not need_ocr and not need_objects and not need_depth:
                print("[FUSION] No camera required for this query.")
                start_time = time.time()
                response = assistant.process_live_query(
                    user_query=clean_query,
                    command=command,
                    speak=True
                )
                elapsed = time.time() - start_time
                print(f"[ASSISTANT] Response ({elapsed:.2f}s): \"{response.text}\"")
                print("[TTS] Generating speech...")
                print("[TTS] Audio playback started\n")
                continue

            # 4. 📷 Live Camera: Capture CURRENT Camera Frame
            print("[CAMERA] Capturing current frame...")
            frame = camera.get_current_frame()
            if frame is None:
                print("[CAMERA] Notice: Live camera stream not accessible.")

            # 5. 👁️ Vision Processing on LIVE Frame
            if need_ocr:
                print("[OCR] Processing frame...")
            if need_objects:
                print("[YOLO/MiDaS] Detecting objects and estimating depth...")

            visual_data = assistant.vision.process_live_frame(
                frame=frame,
                need_ocr=need_ocr,
                need_objects=need_objects,
                need_depth=need_depth
            )

            if need_ocr:
                detected_text = visual_data.get("text", "")
                if detected_text:
                    print(f"[OCR] Detected text: {detected_text}")
                else:
                    print("[OCR] Detected text: [None]")

            if need_objects:
                objs = visual_data.get("objects", [])
                if objs:
                    obj_summary = [f"{o.get('label')} ({o.get('position')}, {o.get('distance')})" for o in objs]
                    print(f"[YOLO/MiDaS] Objects detected: {obj_summary}")
                else:
                    print("[YOLO/MiDaS] Objects detected: [None]")

            # 6. 🔗 Fusion Layer: Send Factual Context to LLM
            print("[FUSION] Sending visual context to LLM...")
            visual_context = fusion.fuse(clean_query, visual_data, command=command)

            # 7. 🤖 LLM Response Generation
            start_time = time.time()
            response = assistant.process(
                user_query=clean_query,
                visual_context=visual_context,
                speak=True
            )
            elapsed = time.time() - start_time
            print(f"[LLM] Response generated ({elapsed:.2f}s): \"{response.text}\"")

            # 8. 🔊 Edge TTS Audio Output
            print("[TTS] Generating speech...")
            print("[TTS] Audio playback started\n")

        except KeyboardInterrupt:
            print("\n[Eyera] Session terminated by user.")
            break
        except Exception as e:
            print(f"\n[Pipeline Error] {e}")


if __name__ == "__main__":
    run_live_vision_assistant()
