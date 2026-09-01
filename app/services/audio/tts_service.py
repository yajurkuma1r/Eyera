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
            # Do not fabricate success - log clearly if TTS fails.
            print(f"[TTS] Failed to generate/play speech: {e}")
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
                os.startfile(path)
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