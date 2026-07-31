"""Non-blocking keyboard input for controlling recording and gripper."""

import sys
import threading
import termios
import tty


class KeyListener:
    """Reads single characters from stdin in a background thread."""

    def __init__(self):
        self.keys_pressed = set()
        self._stop = False
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True

    def pop(self, key: str) -> bool:
        """Check and consume a key press."""
        if key in self.keys_pressed:
            self.keys_pressed.discard(key)
            return True
        return False

    def _listen(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop:
                ch = sys.stdin.read(1).lower()
                self.keys_pressed.add(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
