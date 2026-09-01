import numpy as np
from typing import Any, Dict, Optional, Union
from app.models.assistant_response import AssistantResponse
from app.services.vision_assistance.llm_service import LLMService
from app.services.audio.tts_service import TTSService
from app.services.vision_assistance.fusion_service import FusionService
from app.services.vision_assistance.vision_service import VisionService
from app.services.vision_assistance.camera_service import CameraService
from app.services.vision_assistance.daily_info_service import DailyInfoService


class AssistantService:
    """
    Main Live AI Assistant Service for Eyera (Vision Assistance Module).

    Coordinates:
    1. Intent-based vision capability determination (FusionService)
    2. Live camera frame capture (CameraService)
    3. Selective Vision & OCR perception (VisionService)
    4. Multimodal context fusion (FusionService)
    5. Factual LLM reasoning (LLMService)
    6. Real-time spoken output (TTSService)
    """

    def __init__(self, model: str = "gpt-4o-mini", mock: bool = False):
        self.llm = LLMService(model=model, mock=mock)
        self.tts = TTSService()
        self.fusion = FusionService()
        self.vision = VisionService()
        self.camera = CameraService()
        self.daily_info = DailyInfoService()

    def process(
        self,
        user_query: str,
        visual_context: str = "",
        speak: bool = True
    ) -> AssistantResponse:
        """
        Processes a query with an already formatted visual context string.
        """
        response_text = self.llm.generate_response(
            user_query=user_query,
            visual_context=visual_context
        )

        response = AssistantResponse(
            text=response_text,
            priority="NORMAL",
            should_speak=speak
        )

        if response.should_speak:
            self.tts.speak(response.text)

        return response

    def process_live_query(
        self,
        user_query: str,
        command: str = "",
        frame: Optional[np.ndarray] = None,
        speak: bool = True
    ) -> AssistantResponse:
        """
        Executes the 100% LIVE vision assistant pipeline:
        1. Determines needed vision capabilities from user command.
        2. Captures live camera frame if not provided.
        3. Runs YOLO / MiDaS / OCR selectively on the live frame.
        4. Fuses factual visual data into LLM prompt.
        5. Generates dynamic natural response and speaks it.

        Daily-life commands (time / date / day / festival) are answered
        directly from the device clock and calendar - the camera is
        never activated for them, since they carry no visual content.
        """
        if command in self.fusion.NO_VISION_COMMANDS:
            response_text = self.daily_info.answer(command)
            response = AssistantResponse(
                text=response_text,
                priority="NORMAL",
                should_speak=speak
            )
            if response.should_speak:
                self.tts.speak(response.text)
            return response

        # Step 1: Determine required capabilities
        reqs = self.fusion.determine_requirements(command=command, user_query=user_query)
        need_ocr = reqs.get("need_ocr", False)
        need_objects = reqs.get("need_objects", True)
        need_depth = reqs.get("need_depth", True)

        # Step 2: Capture live camera frame if not provided
        if frame is None:
            frame = self.camera.get_current_frame()

        # Step 3: Run vision models on live frame
        visual_data = self.vision.process_live_frame(
            frame=frame,
            need_ocr=need_ocr,
            need_objects=need_objects,
            need_depth=need_depth
        )

        # Step 4: Fuse factual context
        visual_context = self.fusion.fuse(
            user_query=user_query,
            visual_data=visual_data,
            command=command
        )

        # Step 5 & 6: LLM reasoning and speech
        return self.process(
            user_query=user_query,
            visual_context=visual_context,
            speak=speak
        )


def main():
    assistant = AssistantService()
    print("================================")
    print("   EYERA LIVE AI ASSISTANT")
    print("================================")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            continue

        try:
            response = assistant.process_live_query(user_query=user_query)
            print(f"Eyera: {response.text}")
        except Exception as error:
            print(f"[ERROR] {error}")


if __name__ == "__main__":
    main()
