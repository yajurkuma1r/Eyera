import asyncio
import edge_tts
import os
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

        asyncio.run(self._generate_and_play(message))

        self.last_spoken[message] = current_time

    async def _generate_and_play(self, message):

        communicate = edge_tts.Communicate(
            message,
            voice="en-US-AriaNeural"
        )

        await communicate.save("tts.mp3")

        os.system("start tts.mp3")