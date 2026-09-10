import threading
from contextlib import contextmanager
from typing import Optional


class ConcurrencyController:
    """
    Thread-safe concurrency and state controller for Eyera Vision Assistant.
    Ensures:
    - Only one operation (listening, processing, TTS playback) runs at any time.
    - New voice inputs are blocked while an operation is still processing or playing audio.
    - System cannot get permanently stuck if an exception occurs (using try...finally lock cleanup).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._is_busy = False
        self._is_listening_enabled = False
        self._current_state = "IDLE"

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._is_busy

    @property
    def is_listening_enabled(self) -> bool:
        with self._lock:
            return self._is_listening_enabled

    @property
    def current_state(self) -> str:
        with self._lock:
            return self._current_state

    def enable_listening(self) -> bool:
        with self._lock:
            self._is_listening_enabled = True
            return True

    def disable_listening(self) -> bool:
        with self._lock:
            self._is_listening_enabled = False
            return False

    def toggle_listening(self) -> bool:
        with self._lock:
            self._is_listening_enabled = not self._is_listening_enabled
            return self._is_listening_enabled

    def set_listening_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self._is_listening_enabled = enabled
            return self._is_listening_enabled

    @contextmanager
    def acquire_operation(self, state_label: str = "PROCESSING"):
        """
        Context manager to acquire operational lock safely.
        Guarantees that state is restored to IDLE and is_busy to False
        even if an exception occurs inside the operation block.
        """
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError("Operation already in progress.")

        if self._is_busy:
            self._lock.release()
            raise RuntimeError("Operation already in progress.")

        try:
            self._is_busy = True
            self._current_state = state_label
            yield
        finally:
            self._is_busy = False
            self._current_state = "IDLE"
            self._lock.release()

    def get_status_dict(self) -> dict:
        with self._lock:
            return {
                "listening_enabled": self._is_listening_enabled,
                "is_busy": self._is_busy,
                "state": self._current_state
            }


# Shared singleton instance
concurrency_controller = ConcurrencyController()
