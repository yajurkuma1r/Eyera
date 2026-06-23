import pyttsx3
import time

while True:

    engine = pyttsx3.init()

    print("SPEAKING")

    engine.say("Person approaching")
    engine.runAndWait()
    engine.stop()

    time.sleep(3)