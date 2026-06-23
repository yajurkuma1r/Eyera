from app.services.audio.tts_service import TTSService
import time

tts = TTSService()

while True:
    tts.speak("Person approaching")
    time.sleep(5)