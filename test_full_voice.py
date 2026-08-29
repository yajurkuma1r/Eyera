from app.services.audio.speech_service import SpeechService
from app.services.audio.command_service import CommandService
from app.services.audio.tts_service import TTSService


speech_service = SpeechService()
command_service = CommandService()
tts_service = TTSService()

print("Eyera Voice Assistant Test")
print("Say a command...")

text = speech_service.listen()

if text:
    command = command_service.process_command(text)

    print(f"[Command] {command}")

    responses = {
        "READ_MENU": "Okay, I will read the menu.",
        "DESCRIBE_OBJECT": "Okay, I will describe the object.",
        "DESCRIBE_SCENE": "Okay, I will describe what is in front of you.",
        "START_NAVIGATION": "Navigation started.",
        "STOP_NAVIGATION": "Navigation stopped.",
        "UNKNOWN": "Sorry, I did not understand that command."
    }

    response = responses.get(command, responses["UNKNOWN"])

    tts_service.speak(response)

else:
    tts_service.speak("Sorry, I could not hear you.")