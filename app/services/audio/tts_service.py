import asyncio
import edge_tts
import os
import time
import queue
import threading
import pythoncom
import tempfile


class TTSService:
    def __init__(self):
        self.last_spoken = {}
        self.cooldown_seconds = 3
        self.speech_queue = queue.PriorityQueue()
        self.lock = threading.RLock()
        
        # Priority preemption state
        self.active_player = None
        self.active_priority = None
        self.active_message = None
        self.interrupted_message = None
        
        self._counter = 0
        
        # Callback to broadcast speech messages to other modules/websockets
        self.on_speak_callback = None
        
        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def speak(self, message, priority=2, force_queue=False):
        """
        Speak a message.
        priority=1 for safety warnings (High priority).
        priority=2 for normal navigation (Normal priority).
        force_queue=True bypasses the message cooldown check (e.g. for re-queueing).
        """
        current_time = time.time()
        
        with self.lock:
            # Cooldown check: prevent spamming the exact same message too quickly
            if not force_queue and message in self.last_spoken:
                elapsed = current_time - self.last_spoken[message]
                if elapsed < self.cooldown_seconds:
                    return
            self.last_spoken[message] = current_time

            # Preemption check: if safety warning (priority 1) comes in while nav instruction is playing
            if priority == 1 and self.active_priority == 2:
                if self.active_player is not None:
                    try:
                        print(f"[TTS] Preempting current normal speech: '{self.active_message}'")
                        self.active_player.controls.stop()
                        self.interrupted_message = self.active_message
                    except Exception as e:
                        print(f"[TTS] Failed to stop current player: {e}")

            # Put message in priority queue
            # (priority, counter, message) ensures FIFO for same priority
            self._counter += 1
            self.speech_queue.put((priority, self._counter, message))
            print(f"[TTS] Queued (Priority {priority}): {message}")

        # Call registered speak callback if any
        if self.on_speak_callback:
            try:
                self.on_speak_callback(message)
            except Exception as e:
                print(f"[TTS Callback Error] {e}")

    def _speech_worker(self):
        pythoncom.CoInitialize()
        while True:
            priority, _, message = self.speech_queue.get()
            try:
                # Set active metadata
                with self.lock:
                    self.active_priority = priority
                    self.active_message = message

                # Create unique temp file to avoid permission conflicts on disk
                temp_dir = tempfile.gettempdir()
                temp_filename = os.path.join(temp_dir, f"tts_{threading.get_ident()}_{int(time.time() * 1000)}.mp3")
                
                play_success = False
                try:
                    # Generate audio file
                    asyncio.run(self._generate_audio(message, temp_filename))
                    # Play audio file (COM-based with fallback)
                    play_success = self._play_audio_com(temp_filename)
                except Exception as gen_err:
                    print(f"[TTS Generation / Playback Error] {gen_err}, trying offline fallback.")
                    play_success = False
                
                if not play_success:
                    self._play_audio_fallback(temp_filename, message)

                # Clean up temp file
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"[TTS Error] in speech worker: {e}")
            finally:
                with self.lock:
                    self.active_player = None
                    self.active_priority = None
                    self.active_message = None
                
                self.speech_queue.task_done()
                
                # If we interrupted a normal message, re-queue it so it can continue afterward
                msg_to_requeue = None
                with self.lock:
                    if self.interrupted_message is not None:
                        msg_to_requeue = self.interrupted_message
                        self.interrupted_message = None
                        
                if msg_to_requeue is not None:
                    # Re-queue with normal priority (will play after safety warnings)
                    # We use force_queue=True to bypass cooldown for the interrupted instruction
                    self.speak(msg_to_requeue, priority=2, force_queue=True)

    async def _generate_audio(self, message, filename):
        communicate = edge_tts.Communicate(
            message,
            voice="en-US-AriaNeural"
        )
        await communicate.save(filename)

    def _play_audio_com(self, filename):
        try:
            import win32com.client
            player = win32com.client.Dispatch("WMPlayer.OCX.7")
            player.settings.volume = 100
            player.URL = os.path.abspath(filename)
            player.controls.play()
            
            with self.lock:
                self.active_player = player
                
            # Wait for playback to transition and start playing
            time.sleep(0.3)
            # playState: 3 = playing, 9 = transitioning, 2 = paused, 1 = stopped, 8 = ended
            while player.playState in [3, 9]:
                time.sleep(0.1)
                
            return True
        except Exception as e:
            # COM unavailable or failed
            return False

    def _play_audio_fallback(self, filename, message):
        print(f"[TTS Fallback] Playing speech offline using SAPI: '{message}'")
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(message)
            return True
        except Exception as e:
            print(f"[TTS Fallback Error] SAPI failed: {e}")
            os.system(f"start {filename}")
            word_count = len(message.split())
            duration = max(1.5, word_count * 0.45 + 0.8)
            time.sleep(duration)
            return False