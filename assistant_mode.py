"""
Eyera AI Assistant + Vision Fusion Master Orchestrator

Ties together:
1. User Voice Command (STTService / Mic)
2. Live Scene Perception (SceneCaptureService / OCR + YOLO)
3. Multimodal Relevance Filtering (FusionService)
4. LLM Scene Understanding (LLMService)
5. Spoken Feedback (TTSService / Edge TTS)
"""

import sys
import time
from app.services.assistant.assistant_service import AssistantService
from app.services.audio.stt_service import STTService
from app.services.vision.scene_capture import SceneCaptureService


def run_assistant_pipeline():
    print("=" * 60)
    print("      EYERA AI ASSISTANT + VISION FUSION RUNNER      ")
    print("=" * 60)

    # Initialize services
    print("\n[Init] Initializing Assistant, Fusion, and Audio services...")
    assistant = AssistantService()
    stt = STTService()
    scene_capture = SceneCaptureService()

    # Configuration defaults
    current_preset = "menu"
    use_live_camera = False

    print("\n[Ready] Eyera Assistant Mode Active.")
    print("Available Commands:")
    print("  - Type or speak your query (e.g., 'Read the menu', 'What is in front of me?')")
    print("  - 'preset <menu|street|office|bus>' to switch simulated camera scene")
    print("  - 'camera' to toggle live webcam vs preset scenes")
    print("  - 'exit' or 'quit' to stop\n")

    while True:
        try:
            print("-" * 60)
            mode_indicator = "Live Camera" if use_live_camera else f"Preset: '{current_preset}'"
            print(f"[Mode: {mode_indicator}]")

            # 1. Capture Voice or Text Input
            query = stt.listen()

            if not query:
                continue

            clean_query = query.strip()
            if clean_query.lower() in ("exit", "quit"):
                print("[Eyera] Shutting down Assistant mode. Goodbye!")
                break

            # Handle control commands
            if clean_query.lower().startswith("preset "):
                preset_name = clean_query.split(" ", 1)[1].strip()
                if preset_name in scene_capture.list_presets():
                    current_preset = preset_name
                    use_live_camera = False
                    print(f"[Eyera] Switched to scene preset: '{current_preset}'")
                else:
                    print(f"[Eyera] Unknown preset. Choose from: {scene_capture.list_presets()}")
                continue

            if clean_query.lower() == "camera":
                use_live_camera = not use_live_camera
                status = "Live Webcam" if use_live_camera else f"Preset: '{current_preset}'"
                print(f"[Eyera] Camera mode toggled to: {status}")
                continue

            # 2. Capture Vision & OCR Perception Data
            print("\n[Pipeline Step 1] Capturing visual perception...")
            if use_live_camera:
                visual_data = scene_capture.capture_live()
            else:
                visual_data = scene_capture.get_preset(current_preset)

            print(f" -> Scene: {visual_data.get('scene_name', 'Unknown')}")
            if visual_data.get("ocr_text"):
                ocr_preview = visual_data["ocr_text"].replace('\n', ' ')[:80]
                print(f" -> OCR Extracted: \"{ocr_preview}...\"")
            if visual_data.get("detected_objects"):
                print(f" -> Objects: {[o.get('label') for o in visual_data['detected_objects']]}")

            # 3. Multimodal Fusion & Relevance Extraction
            print("\n[Pipeline Step 2] Running Multimodal Fusion relevance filter...")
            intent = assistant.fusion.classify_intent(clean_query)
            filtered_context = assistant.fusion.fuse(clean_query, visual_data)
            print(f" -> Detected Intent: {intent}")
            print(f" -> Filtered Context Preview:\n{filtered_context}\n")

            # 4. LLM Processing & Edge TTS Generation
            print("[Pipeline Step 3] Generating spoken response via Assistant & LLM...")
            start_time = time.time()
            response = assistant.process_with_fusion(
                user_query=clean_query,
                raw_vision_data=visual_data,
                speak=True
            )
            elapsed = time.time() - start_time

            print("\n" + "=" * 40)
            print(f"EYERA RESPONSE ({elapsed:.2f}s):")
            print(f"\"{response.text}\"")
            print("=" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n[Eyera] Session terminated by user.")
            break
        except Exception as e:
            print(f"\n[Pipeline Error] {e}")


if __name__ == "__main__":
    run_assistant_pipeline()
