import time
import pyttsx3

engine = pyttsx3.init()

while True:
    print("SPEAKING")
    engine.say("person approaching")
    engine.runAndWait()
    time.sleep(3)