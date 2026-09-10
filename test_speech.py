from app.services.audio.speech_service import SpeechService


speech = SpeechService()

print("Say something...")

result = speech.listen()

print("Final result:", result)