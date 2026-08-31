from app.services.audio.speech_service import SpeechService
from app.services.audio.command_service import CommandService


speech_service = SpeechService()
command_service = CommandService()

print("Eyera Voice Command Test")
print("Say a command...")

text = speech_service.listen()

if text:
    command = command_service.process_command(text)
    print(f"[Command] {command}")
else:
    print("[Command] No command detected.")