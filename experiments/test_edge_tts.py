import asyncio
import edge_tts
import os

TEXT = "Person approaching"

async def main():
    communicate = edge_tts.Communicate(
        TEXT,
        voice="en-US-AriaNeural"
    )

    await communicate.save("tts.mp3")

    os.system("start tts.mp3")

asyncio.run(main())