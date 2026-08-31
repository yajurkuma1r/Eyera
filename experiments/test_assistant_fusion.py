"""
Automated Test for Eyera AI Assistant + Vision Fusion Pipeline
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.fusion.fusion_service import FusionService
from app.services.vision.scene_capture import ScenePreset, SceneCaptureService
from app.services.assistant.assistant_service import AssistantService


def test_fusion_intent_classification():
    print("\n--- Test 1: Intent Classification ---")
    fusion = FusionService()

    tests = [
        ("Read the menu for me", "READING"),
        ("What does that sign say?", "READING"),
        ("Where is the chair?", "OBJECT_SEARCH"),
        ("What do you see in front of me?", "OBJECT_SEARCH"),
        ("Is it safe to cross the street?", "SAFETY"),
        ("Hello, who are you?", "GENERAL")
    ]

    for query, expected in tests:
        intent = fusion.classify_intent(query)
        print(f"Query: \"{query}\" -> Intent: {intent} (Expected: {expected})")
        assert intent == expected, f"Failed for query '{query}': got {intent}, expected {expected}"

    print("[PASS] Intent classification tests passed.")


def test_fusion_relevance_filtering():
    print("\n--- Test 2: Fusion Context Filtering ---")
    fusion = FusionService()

    # Reading scenario
    query_read = "Read the menu"
    context_read = fusion.fuse(query_read, ScenePreset.CAFE_MENU)
    print(f"Reading Context:\n{context_read}\n")
    assert "[Detected Text in Scene]" in context_read
    assert "Espresso" in context_read

    # Spatial / Object scenario
    query_objects = "Where is the door?"
    context_objects = fusion.fuse(query_objects, ScenePreset.OFFICE_ROOM)
    print(f"Object Context:\n{context_objects}\n")
    assert "[Detected Objects & Positions]" in context_objects
    assert "door" in context_objects

    # Safety scenario
    query_safety = "Is it safe to walk?"
    context_safety = fusion.fuse(query_safety, ScenePreset.STREET_CROSSING)
    print(f"Safety Context:\n{context_safety}\n")
    assert "[CRITICAL SAFETY WARNINGS]" in context_safety
    assert "approaching" in context_safety.lower()

    print("[PASS] Fusion context filtering tests passed.")


def test_end_to_end_assistant_pipeline():
    print("\n--- Test 3: End-to-End Multimodal Assistant Pipeline ---")
    assistant = AssistantService(mock=True)

    # 1. Test Menu Reading
    resp_menu = assistant.process_with_fusion(
        user_query="Read the specials on the menu",
        raw_vision_data=ScenePreset.CAFE_MENU,
        speak=False
    )
    print(f"Menu Response: \"{resp_menu.text}\"")
    assert resp_menu.text, "Response text should not be empty"
    assert resp_menu.should_speak is False or resp_menu.should_speak is True

    # 2. Test Obstacle Navigation
    resp_street = assistant.process_with_fusion(
        user_query="Is anything approaching me?",
        raw_vision_data=ScenePreset.STREET_CROSSING,
        speak=False
    )
    print(f"Safety Response: \"{resp_street.text}\"")
    assert resp_street.text, "Response text should not be empty"
    assert "caution" in resp_street.text.lower() or "approaching" in resp_street.text.lower()

    print("[PASS] End-to-end Assistant pipeline tests passed.")


if __name__ == "__main__":
    print("==============================================")
    print("  RUNNING ASSISTANT + VISION FUSION TESTS     ")
    print("==============================================")
    test_fusion_intent_classification()
    test_fusion_relevance_filtering()
    test_end_to_end_assistant_pipeline()
    print("\n==============================================")
    print("  ALL TESTS PASSED SUCCESSFULLY!              ")
    print("==============================================")
