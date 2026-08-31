import sys
from typing import Optional


class STTService:
    """
    Speech-to-Text (STT) service for capturing voice queries from the user's microphone.
    Provides automatic fallback to keyboard input if no microphone or STT package is available.
    """

    def __init__(self, energy_threshold: int = 300):
        self.energy_threshold = energy_threshold
        self._recognizer = None
        self._microphone = None
        self._mic_available = False

        self._init_speech_recognition()

    def _init_speech_recognition(self):
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self.energy_threshold
            self._recognizer.dynamic_energy_threshold = True

            # Test microphone access
            with sr.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self._mic_available = True
                print("[STTService] Microphone initialized and calibrated.")
        except ImportError:
            print("[STTService] speech_recognition not installed. Falling back to keyboard input.")
            self._mic_available = False
        except Exception as e:
            print(f"[STTService] Microphone unavailable ({e}). Falling back to keyboard input.")
            self._mic_available = False

    def is_microphone_available(self) -> bool:
        """
        Returns True if a working microphone is detected.
        """
        return self._mic_available

    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> str:
        """
        Listens to the microphone and transcribes voice to text.
        Falls back to terminal input if voice recognition fails or is unavailable.
        """
        if self._mic_available:
            try:
                import speech_recognition as sr
                with sr.Microphone() as source:
                    print("\n[STT] Listening... (Speak into microphone)")
                    audio = self._recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit
                    )
                    print("[STT] Processing audio...")
                    text = self._recognizer.recognize_google(audio)
                    print(f"[STT] Heard: \"{text}\"")
                    return text.strip()
            except sr.WaitTimeoutError:
                print("[STT] No speech detected (timeout).")
                return ""
            except sr.UnknownValueError:
                print("[STT] Could not understand audio.")
                return ""
            except Exception as e:
                print(f"[STT] Speech recognition error ({e}). Using prompt fallback.")

        # Fallback to interactive prompt
        try:
            user_input = input("You (Voice/Text): ").strip()
            return user_input
        except EOFError:
            return ""
