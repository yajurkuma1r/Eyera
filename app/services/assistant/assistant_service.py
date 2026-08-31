from typing import Any, Dict, Optional, Union
from app.models.assistant_response import AssistantResponse
from app.services.assistant.llm_service import LLMService
from app.services.audio.tts_service import TTSService
from app.services.fusion.fusion_service import FusionService


class AssistantService:
    """
    Main AI Assistant service for Eyera.

    Takes user voice requests and visual perception data,
    applies the Fusion Service to extract relevant context,
    queries the LLM, and speaks the response via Edge TTS.
    """

    def __init__(self, model: str = "gpt-4o-mini", mock: bool = False):
        self.llm = LLMService(model=model, mock=mock)
        self.tts = TTSService()
        self.fusion = FusionService()

    def process(
        self,
        user_query: str,
        visual_context: str = "",
        speak: bool = True
    ) -> AssistantResponse:
        """
        Process with pre-formatted visual context string.
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

    def process_with_fusion(
        self,
        user_query: str,
        raw_vision_data: Optional[Union[Dict[str, Any], str]] = None,
        speak: bool = True
    ) -> AssistantResponse:
        """
        Processes raw multimodal vision perception data (OCR + YOLO + depth)
        through the Fusion Service before passing to LLM and TTS.
        """
        visual_context = self.fusion.fuse(
            user_query=user_query,
            visual_data=raw_vision_data
        )

        return self.process(
            user_query=user_query,
            visual_context=visual_context,
            speak=speak
        )


def main():
    assistant = AssistantService()

    print("================================")
    print("       EYERA AI ASSISTANT")
    print("================================")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() == "exit":
            break

        if not user_query:
            continue

        try:
            response = assistant.process(user_query=user_query)
            print(f"Eyera: {response.text}")
        except Exception as error:
            print(f"[ERROR] {error}")


if __name__ == "__main__":
    main()
