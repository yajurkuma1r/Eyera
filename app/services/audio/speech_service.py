import speech_recognition as sr


class SpeechService:

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen(self):
        """
        Listen to the microphone and convert speech to text.
        Returns the recognized text or None if speech could not be understood.
        """

        with sr.Microphone() as source:

            print("[Speech] Listening...")

            self.recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=5
                )

                print("[Speech] Processing...")

                text = self.recognizer.recognize_google(audio)

                print(f"[Speech] You said: {text}")

                return text.lower()

            except sr.WaitTimeoutError:
                print("[Speech] No speech detected.")
                return None

            except sr.UnknownValueError:
                print("[Speech] Could not understand the speech.")
                return None

            except sr.RequestError as e:
                print(f"[Speech] Speech recognition service error: {e}")
                return None