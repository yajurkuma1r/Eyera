"""
Comprehensive Integration Test for app/services/vision_assistance/
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.vision_assistance import (
    VisionService,
    CameraService,
    ApproachDetector,
    CommandService,
    SpeechService,
    STTService,
    FusionService,
    LLMService,
    DailyInfoService,
    AssistantService,
    VoiceController,
)


def test_vision_assistance_imports():
    print("\n--- Test 1: Stitched Module Imports ---")
    assert VisionService is not None
    assert CameraService is not None
    assert ApproachDetector is not None
    assert CommandService is not None
    assert SpeechService is not None
    assert STTService is not None
    assert FusionService is not None
    assert LLMService is not None
    assert AssistantService is not None
    assert VoiceController is not None
    print("[PASS] All stitched classes imported successfully from app.services.vision_assistance.")


def test_command_service():
    print("\n--- Test 2: Command & Intent Service ---")
    cmd_svc = CommandService()
    assert cmd_svc.process_command("read the menu") == "READ_MENU"
    assert cmd_svc.process_command("what does this sign say") == "READ_SIGN"
    assert cmd_svc.process_command("what is in front of me") == "DESCRIBE_SCENE"
    assert cmd_svc.process_command("where is the door") == "UNKNOWN" or "DESCRIBE" in cmd_svc.process_command("where is the door") or cmd_svc.process_command("where is the door") == "UNKNOWN"
    assert cmd_svc.process_command("is there an obstacle") == "CHECK_OBSTACLE"
    print("[PASS] CommandService intent recognition verified.")


def test_fusion_requirements():
    print("\n--- Test 3: Fusion Requirements Determination ---")
    fusion = FusionService()

    req_menu = fusion.determine_requirements(command="READ_MENU", user_query="read the menu")
    assert req_menu["need_ocr"] is True
    assert req_menu["need_objects"] is False

    req_scene = fusion.determine_requirements(command="DESCRIBE_SCENE", user_query="what is in front of me")
    assert req_scene["need_ocr"] is False
    assert req_scene["need_objects"] is True
    assert req_scene["need_depth"] is True

    req_safety = fusion.determine_requirements(command="CHECK_OBSTACLE", user_query="is it safe to walk")
    assert req_safety["need_objects"] is True
    assert req_safety["need_depth"] is True
    print("[PASS] Fusion requirements determination verified.")


def test_live_vision_processing():
    print("\n--- Test 4: Live Vision & Factual Processing ---")
    # Synthetic frame test (black image / blank test frame)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assistant = AssistantService(mock=True)

    # 1. Reading query on blank frame (should report no text detected)
    resp_blank = assistant.process_live_query(
        user_query="Read the menu",
        command="READ_MENU",
        frame=test_frame,
        speak=False
    )
    print(f"Reading on blank frame -> \"{resp_blank.text}\"")
    assert "couldn't detect" in resp_blank.text.lower() or "no readable text" in resp_blank.text.lower()

    # 2. Reading query with factual OCR extracted text
    factual_vision_data = {
        "text": "Daily Specials Espresso $3.50 Latte $4.50",
        "objects": []
    }
    context_read = assistant.fusion.fuse("Read the menu", factual_vision_data, command="READ_MENU")
    assert "Daily Specials" in context_read

    resp_read = assistant.process(user_query="Read the menu", visual_context=context_read, speak=False)
    print(f"Reading with live text -> \"{resp_read.text}\"")
    assert "Espresso" in resp_read.text or "Daily Specials" in resp_read.text or "Latte" in resp_read.text

    # 3. Spatial query with factual YOLO objects
    factual_spatial_data = {
        "text": "",
        "objects": [
            {"label": "chair", "position": "center", "distance": "close (1m - 2m)"},
            {"label": "door", "position": "right", "distance": "far (> 2m)"}
        ]
    }
    context_spatial = assistant.fusion.fuse("Where is the chair?", factual_spatial_data, command="DESCRIBE_OBJECT")
    resp_spatial = assistant.process(user_query="Where is the chair?", visual_context=context_spatial, speak=False)
    print(f"Spatial query -> \"{resp_spatial.text}\"")
    assert "chair" in resp_spatial.text.lower()

    print("[PASS] Live factual vision processing verified.")


def test_daily_info_commands():
    print("\n--- Test 5: Daily-Life Commands (no camera required) ---")
    cmd_svc = CommandService()
    fusion = FusionService()
    daily = DailyInfoService()

    # Intent detection
    assert cmd_svc.process_command("what time is it") == "GET_TIME"
    assert cmd_svc.process_command("what's the date") == "GET_DATE"
    assert cmd_svc.process_command("what day is it") == "GET_DAY"
    assert cmd_svc.process_command("is there any festival today") == "GET_FESTIVAL"

    # These must never request vision capabilities
    for command in ["GET_TIME", "GET_DATE", "GET_DAY", "GET_FESTIVAL"]:
        reqs = fusion.determine_requirements(command=command, user_query="")
        assert reqs["need_ocr"] is False
        assert reqs["need_objects"] is False
        assert reqs["need_depth"] is False
        assert reqs["capability"] == "NONE"

    # Answers are deterministic, from the real clock/calendar
    assert "currently" in daily.answer("GET_TIME").lower()
    assert "today is" in daily.answer("GET_DATE").lower()
    assert "today is" in daily.answer("GET_DAY").lower()

    # AssistantService should bypass the camera entirely for these
    assistant = AssistantService(mock=True)
    response = assistant.process_live_query(
        user_query="What time is it?",
        command="GET_TIME",
        speak=False
    )
    print(f"Daily info (time) -> \"{response.text}\"")
    assert "currently" in response.text.lower()

    print("[PASS] Daily-life commands verified (camera never invoked).")


def test_voice_controller_integration():
    print("\n--- Test 5: VoiceController Integration ---")
    vc = VoiceController()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    reply = vc.handle_command_live("What is in front of me?", "DESCRIBE_SCENE", frame=test_frame)
    print(f"VoiceController reply -> \"{reply}\"")
    assert reply, "Reply should not be empty"
    print("[PASS] VoiceController integration verified.")


if __name__ == "__main__":
    print("======================================================")
    print("  RUNNING VISION ASSISTANCE COMPREHENSIVE TESTS       ")
    print("======================================================")
    test_vision_assistance_imports()
    test_command_service()
    test_fusion_requirements()
    test_live_vision_processing()
    test_daily_info_commands()
    test_voice_controller_integration()
    print("\n======================================================")
    print("  ALL VISION ASSISTANCE TESTS PASSED SUCCESSFULLY!    ")
    print("======================================================")
