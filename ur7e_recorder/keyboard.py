"""Non-blocking keyboard input for controlling recording and gripper."""

import select
import sys
import termios
import threading
import tty


class KeyListener:
    """Reads single characters from stdin in a background thread."""

    def __init__(self):
        self.keys_pressed = set()
        self._stop = False
        self._fd = sys.stdin.fileno()
        self._old_settings = None
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        # Save/restore terminal settings from the main thread so the
        # terminal is guaranteed to be sane again after stop(), even if
        # the background thread is parked in a blocking read.
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def pop(self, key: str) -> bool:
        """Check and consume a key press."""
        if key in self.keys_pressed:
            self.keys_pressed.discard(key)
            return True
        return False

    def _listen(self):
        while not self._stop:
            # Poll with a timeout instead of blocking forever in
            # sys.stdin.read(1), so the loop notices _stop promptly.
            ready, _, _ = select.select([self._fd], [], [], 0.2)
            if not ready:
                continue
            ch = sys.stdin.read(1).lower()
            self.keys_pressed.add(ch)
