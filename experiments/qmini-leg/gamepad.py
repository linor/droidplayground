"""
gamepad.py -- non-blocking Logitech F710 (XInput mode, switch on "X") reader
via the Linux joystick API (/dev/input/js*), no extra Python packages.

The xpad driver's joystick mapping for an XInput pad:
  axes:    0 left X (+right)   1 left Y (+down)   2 LT   3 right X (+right)
           4 right Y (+down)   5 RT               6 d-pad X   7 d-pad Y
  buttons: 0 A  1 B  2 X  3 Y  4 LB  5 RB  6 back  7 start  8 guide  9 LS  10 RS
Check yours with `jstest /dev/input/js0` if a mapping looks off.

Each event is struct js_event {u32 time_ms; s16 value; u8 type; u8 number}.
The device is opened O_NONBLOCK and drained every poll(); if it disappears
(out of range, battery) poll() reports disconnected and keeps retrying the
open, so the caller can fall back to a zero command.
"""
from __future__ import annotations

import errno, glob, os, struct, time

JS_EVENT_BUTTON, JS_EVENT_AXIS, JS_EVENT_INIT = 0x01, 0x02, 0x80
_FMT = "IhBB"
_SIZE = struct.calcsize(_FMT)

AXIS_LX, AXIS_LY, AXIS_LT, AXIS_RX, AXIS_RY, AXIS_RT = 0, 1, 2, 3, 4, 5
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_LB, BTN_RB, BTN_BACK, BTN_START = 0, 1, 2, 3, 4, 5, 6, 7


class Gamepad:
    def __init__(self, device: str | None = None, retry_s: float = 1.0):
        self.device_arg = device
        self.retry_s = retry_s
        self.fd = None
        self.path = None
        self.axes: dict[int, float] = {}
        self.buttons: dict[int, bool] = {}
        self._pressed_edges: set[int] = set()
        self._last_try = 0.0
        self._open()

    @property
    def connected(self) -> bool:
        return self.fd is not None

    def _open(self):
        self._last_try = time.monotonic()
        paths = [self.device_arg] if self.device_arg else sorted(glob.glob("/dev/input/js*"))
        for p in paths:
            try:
                self.fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
                self.path = p
                self.axes.clear(); self.buttons.clear(); self._pressed_edges.clear()
                return True
            except OSError:
                continue
        self.fd = None
        return False

    def _close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = None
        self.axes.clear(); self.buttons.clear()

    def poll(self) -> bool:
        """Drain pending events; returns connected. Never blocks."""
        if self.fd is None:
            if time.monotonic() - self._last_try >= self.retry_s:
                self._open()
            if self.fd is None:
                return False
        while True:
            try:
                data = os.read(self.fd, _SIZE * 64)
            except BlockingIOError:
                break
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                self._close()            # ENODEV: unplugged / out of range
                return False
            if not data:
                break
            for i in range(0, len(data) - _SIZE + 1, _SIZE):
                _t, value, etype, number = struct.unpack_from(_FMT, data, i)
                kind = etype & ~JS_EVENT_INIT
                if kind == JS_EVENT_AXIS:
                    self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
                elif kind == JS_EVENT_BUTTON:
                    pressed = bool(value)
                    if pressed and not self.buttons.get(number, False) and not (etype & JS_EVENT_INIT):
                        self._pressed_edges.add(number)
                    self.buttons[number] = pressed
        return True

    def axis(self, n: int) -> float:
        return self.axes.get(n, 0.0)

    def pressed(self, n: int) -> bool:
        """True once per press (edge), since the last call for that button."""
        if n in self._pressed_edges:
            self._pressed_edges.discard(n)
            return True
        return False


def deadzone(x: float, dz: float) -> float:
    """1D deadzone, rescaled so the output still reaches +-1."""
    a = abs(x)
    return 0.0 if a < dz else (a - dz) / (1.0 - dz) * (1.0 if x > 0 else -1.0)


def deadzone2(x: float, y: float, dz: float) -> tuple[float, float]:
    """Radial deadzone for a stick (same as qmini-gait/web/input_gamepad.js)."""
    m = (x * x + y * y) ** 0.5
    if m < dz:
        return 0.0, 0.0
    s = (m - dz) / (1.0 - dz) / m
    return max(-1.0, min(1.0, x * s)), max(-1.0, min(1.0, y * s))


if __name__ == "__main__":      # quick check on the robot: python3 gamepad.py
    pad = Gamepad()
    print("device:", pad.path or "none found (is the F710 receiver plugged in, switch on X?)")
    try:
        while True:
            pad.poll()
            lx, ly = deadzone2(pad.axis(AXIS_LX), pad.axis(AXIS_LY), 0.12)
            print(f"\rconnected {pad.connected}  left ({lx:+.2f}, {ly:+.2f})  right X {deadzone(pad.axis(AXIS_RX), 0.14):+.2f}  "
                  f"buttons {[n for n, b in sorted(pad.buttons.items()) if b]}        ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
