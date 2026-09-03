from typing import Optional
import numpy as np
from app.services.audio.speech_service import SpeechService
from app.services.audio.command_service import CommandService
from app.services.audio.tts_service import TTSService
from app.services.vision_assistance.assistant_service import AssistantService


class VoiceController:
    """
    Voice Controller for Eyera.
    Listens for user speech, detects intent via CommandService,
    and executes the live vision assistant pipeline.
    """

    def __init__(self, assistant_service: Optional[AssistantService] = None):
        self.speech_service = SpeechService()
        self.command_service = CommandService()
        self.tts_service = TTSService()
        self.assistant_service = assistant_service or AssistantService()

    def listen_for_command(self) -> tuple[str, str]:
        """
        Listens to the user's voice and extracts both transcribed text and command label.
        Returns: (raw_text, command_label)
        """
        text = self.speech_service.listen()
        if not text:
            return "", "UNKNOWN"

        command = self.command_service.process_command(text)
        print(f"[VoiceController] Heard: \"{text}\" -> Command: [{command}]")
        return text, command

    def speak(self, message: str):
        self.tts_service.speak(message)

    def handle_command_live(self, text: str, command: str, frame: Optional[np.ndarray] = None) -> str:
        """
        Executes live vision understanding and LLM reasoning for the detected command.
        """
        response = self.assistant_service.process_live_query(
            user_query=text,
            command=command,
            frame=frame,
            speak=True
        )
        return response.text

    def run_once(self, frame: Optional[np.ndarray] = None):
        """
        Executes a single live voice-in -> vision -> assistant -> TTS cycle.
        """
        text, command = self.listen_for_command()
        if not text:
            return "UNKNOWN", "No speech detected"

        response_text = self.handle_command_live(text, command, frame=frame)
        return command, response_text