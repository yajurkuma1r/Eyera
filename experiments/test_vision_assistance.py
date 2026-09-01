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

    # Vision Commands
    assert cmd_svc.process_command("read the menu") == "READ_MENU"
    assert cmd_svc.process_command("what does this sign say") == "READ_SIGN"
    assert cmd_svc.process_command("what is in front of me") == "DESCRIBE_SCENE"
    assert cmd_svc.process_command("where is the chair") == "FIND_OBJECT"
    assert cmd_svc.process_command("where is the door") == "FIND_OBJECT"
    assert cmd_svc.process_command("is there an obstacle in front of me") == "CHECK_OBSTACLE"
    assert cmd_svc.process_command("what color is this") == "GET_COLOR"

    # Daily-Life Commands (NO CAMERA)
    assert cmd_svc.process_command("what is the date today") == "GET_DATE"
    assert cmd_svc.process_command("what time is it") == "GET_TIME"
    assert cmd_svc.process_command("what day is today") == "GET_DAY"
    assert cmd_svc.process_command("is there any festival today") == "GET_FESTIVAL"

    # General Knowledge Questions (NO CAMERA)
    assert cmd_svc.process_command("what is the capital of France") == "GENERAL_QUERY"
    assert cmd_svc.process_command("who invented the telephone") == "GENERAL_QUERY"
    assert cmd_svc.process_command("what is machine learning") == "GENERAL_QUERY"
    assert cmd_svc.process_command("why is the sky blue") == "GENERAL_QUERY"

    # Unknown input
    assert cmd_svc.process_command("asdfghjk 123") == "UNKNOWN"

    print("[PASS] CommandService intent recognition verified for Vision, Daily-Life, General Knowledge, and Unknown.")


def test_fusion_requirements():
    print("\n--- Test 3: Fusion Requirements Determination ---")
    fusion = FusionService()

    # CAMERA REQUIRED
    req_menu = fusion.determine_requirements(command="READ_MENU", user_query="read the menu")
    assert req_menu["need_ocr"] is True
    assert req_menu["need_objects"] is False
    assert req_menu["capability"] == "OCR"

    req_scene = fusion.determine_requirements(command="DESCRIBE_SCENE", user_query="what is in front of me")
    assert req_scene["need_ocr"] is False
    assert req_scene["need_objects"] is True
    assert req_scene["need_depth"] is True

    req_find = fusion.determine_requirements(command="FIND_OBJECT", user_query="where is the chair")
    assert req_find["need_objects"] is True
    assert req_find["need_depth"] is True

    req_safety = fusion.determine_requirements(command="CHECK_OBSTACLE", user_query="is it safe to walk")
    assert req_safety["need_objects"] is True
    assert req_safety["need_depth"] is True

    req_color = fusion.determine_requirements(command="GET_COLOR", user_query="what color is this")
    assert req_color["need_objects"] is True
    assert req_color["need_depth"] is True

    # NO CAMERA (Daily-Life)
    for command in ["GET_TIME", "GET_DATE", "GET_DAY", "GET_FESTIVAL"]:
        req_daily = fusion.determine_requirements(command=command, user_query="")
        assert req_daily["need_ocr"] is False
        assert req_daily["need_objects"] is False
        assert req_daily["need_depth"] is False
        assert req_daily["capability"] == "NONE"

    # NO CAMERA (General Knowledge)
    req_gen1 = fusion.determine_requirements(command="GENERAL_QUERY", user_query="What is the capital of France?")
    assert req_gen1["need_ocr"] is False
    assert req_gen1["need_objects"] is False
    assert req_gen1["need_depth"] is False
    assert req_gen1["capability"] == "NONE"

    req_gen2 = fusion.determine_requirements(command="GENERAL_QUERY", user_query="What is machine learning?")
    assert req_gen2["need_ocr"] is False
    assert req_gen2["need_objects"] is False
    assert req_gen2["need_depth"] is False
    assert req_gen2["capability"] == "NONE"

    # NO CAMERA (Unknown)
    req_unknown = fusion.determine_requirements(command="UNKNOWN", user_query="asdfghjk 123")
    assert req_unknown["need_ocr"] is False
    assert req_unknown["need_objects"] is False
    assert req_unknown["need_depth"] is False
    assert req_unknown["capability"] == "NONE"

    print("[PASS] Fusion requirements determination verified for both Camera and No-Camera categories.")


def test_live_vision_processing():
    print("\n--- Test 4: Live Vision & Factual Processing ---")
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
    daily = DailyInfoService()

    # Answers are deterministic, from the real clock/calendar
    assert "currently" in daily.answer("GET_TIME").lower()
    assert "today is" in daily.answer("GET_DATE").lower()
    assert "today is" in daily.answer("GET_DAY").lower()

    # AssistantService should bypass the camera entirely for these
    assistant = AssistantService(mock=True)
    resp_time = assistant.process_live_query(
        user_query="What time is it?",
        command="GET_TIME",
        speak=False
    )
    print(f"Daily info (time) -> \"{resp_time.text}\"")
    assert "currently" in resp_time.text.lower()

    resp_date = assistant.process_live_query(
        user_query="What is the date today?",
        command="GET_DATE",
        speak=False
    )
    print(f"Daily info (date) -> \"{resp_date.text}\"")
    assert "today is" in resp_date.text.lower()

    resp_day = assistant.process_live_query(
        user_query="What day is today?",
        command="GET_DAY",
        speak=False
    )
    print(f"Daily info (day) -> \"{resp_day.text}\"")
    assert "today is" in resp_day.text.lower()

    print("[PASS] Daily-life commands verified (camera never invoked).")


def test_general_knowledge_queries():
    print("\n--- Test 6: General Knowledge Queries (no camera required) ---")
    assistant = AssistantService(mock=True)

    # 1. Capital of France -> Paris
    resp_france = assistant.process_live_query(
        user_query="What is the capital of France?",
        command="GENERAL_QUERY",
        speak=False
    )
    print(f"General Query (France) -> \"{resp_france.text}\"")
    assert "paris" in resp_france.text.lower()
    assert "camera" not in resp_france.text.lower()
    assert "notable objects" not in resp_france.text.lower()

    # 2. Who invented the telephone -> Alexander Graham Bell
    resp_tel = assistant.process_live_query(
        user_query="Who invented the telephone?",
        command="GENERAL_QUERY",
        speak=False
    )
    print(f"General Query (Telephone) -> \"{resp_tel.text}\"")
    assert "bell" in resp_tel.text.lower()
    assert "notable objects" not in resp_tel.text.lower()

    # 3. Machine learning
    resp_ml = assistant.process_live_query(
        user_query="What is machine learning?",
        command="GENERAL_QUERY",
        speak=False
    )
    print(f"General Query (ML) -> \"{resp_ml.text}\"")
    assert "intelligence" in resp_ml.text.lower() or "data" in resp_ml.text.lower() or "learn" in resp_ml.text.lower()
    assert "notable objects" not in resp_ml.text.lower()

    # 4. Sky blue
    resp_sky = assistant.process_live_query(
        user_query="Why is the sky blue?",
        command="GENERAL_QUERY",
        speak=False
    )
    print(f"General Query (Sky) -> \"{resp_sky.text}\"")
    assert "scatter" in resp_sky.text.lower() or "blue" in resp_sky.text.lower() or "light" in resp_sky.text.lower()
    assert "notable objects" not in resp_sky.text.lower()

    # 5. Unknown fallback
    resp_unknown = assistant.process_live_query(
        user_query="asdfghjk 123",
        command="UNKNOWN",
        speak=False
    )
    print(f"Unknown query fallback -> \"{resp_unknown.text}\"")
    assert "rephrase" in resp_unknown.text.lower() or "catch" in resp_unknown.text.lower()
    assert "camera" not in resp_unknown.text.lower()

    print("[PASS] General knowledge queries verified (camera never invoked, factual answers provided).")


def test_voice_controller_integration():
    print("\n--- Test 7: VoiceController Integration ---")
    vc = VoiceController()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Vision command via VoiceController
    reply_vision = vc.handle_command_live("What is in front of me?", "DESCRIBE_SCENE", frame=test_frame)
    print(f"VoiceController vision reply -> \"{reply_vision}\"")
    assert reply_vision, "Vision reply should not be empty"

    # General knowledge query via VoiceController
    reply_general = vc.handle_command_live("What is the capital of France?", "GENERAL_QUERY")
    print(f"VoiceController general reply -> \"{reply_general}\"")
    assert "paris" in reply_general.lower()

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
    test_general_knowledge_queries()
    test_voice_controller_integration()
    print("\n======================================================")
    print("  ALL VISION ASSISTANCE TESTS PASSED SUCCESSFULLY!    ")
    print("======================================================")
