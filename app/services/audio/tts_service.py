import asyncio
import edge_tts
import os
import platform
import subprocess
import time


class TTSService:

    def __init__(self):
        self.last_spoken = {}
        self.cooldown_seconds = 3

    def speak(self, message):

        current_time = time.time()

        if message in self.last_spoken:

            elapsed = current_time - self.last_spoken[message]

            if elapsed < self.cooldown_seconds:
                return

        print(f"[TTS] {message}")

        try:
            asyncio.run(self._generate_and_play(message))
        except Exception as e:
            # Do not fabricate success - log clearly if TTS fails, try pyttsx3 offline fallback.
            print(f"[TTS] Online generation/play failed: {e}. Trying offline pyttsx3 fallback.")
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
            except Exception as fallback_err:
                print(f"[TTS] Failed to generate/play speech with fallback: {fallback_err}")
                return

        self.last_spoken[message] = current_time

    async def _generate_and_play(self, message):

        communicate = edge_tts.Communicate(
            message,
            voice="en-US-AriaNeural"
        )

        await communicate.save("tts.mp3")

        self._play_audio("tts.mp3")

    def _play_audio(self, path):
        """
        Plays the generated audio file. Uses the appropriate player for
        the current OS since 'start' (the previous implementation) only
        works on Windows and would silently fail to play on macOS/Linux.
        """
        system = platform.system()
        try:
            if system == "Windows":
                abs_path = os.path.abspath(path)
                try:
                    import ctypes
                    winmm = ctypes.windll.winmm
                    winmm.mciSendStringW("close eyera_tts", None, 0, 0)
                    open_cmd = f'open "{abs_path}" type mpegvideo alias eyera_tts'
                    if winmm.mciSendStringW(open_cmd, None, 0, 0) == 0:
                        winmm.mciSendStringW("play eyera_tts wait", None, 0, 0)
                        winmm.mciSendStringW("close eyera_tts", None, 0, 0)
                        return
                except Exception as mci_err:
                    print(f"[TTS] winmm playback notice: {mci_err}")
                os.startfile(abs_path)
                time.sleep(2.0)
            elif system == "Darwin":
                subprocess.run(["afplay", path], check=True)
            else:
                # Linux: try common players in order of availability.
                for player in (["mpg123", path], ["ffplay", "-nodisp", "-autoexit", path], ["aplay", path]):
                    try:
                        subprocess.run(player, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue
                raise RuntimeError("No supported audio player found (tried mpg123, ffplay, aplay).")
        except Exception as e:
            print(f"[TTS] Audio playback error: {e}")
            raise