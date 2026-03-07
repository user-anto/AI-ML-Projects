import math
import random
from dataclasses import dataclass, field
from typing import Optional

import cv2

from prompts import ROAST_MODES, mode_label, normalize_roast_mode


WHEEL_LABELS_SHORT = {
    "normal": "Normal",
    "sports-commentary": "Sports",
    "mean-girl": "Mean",
    "corporate-review": "Corp",
}


@dataclass
class ModeWheel:
    modes: list[str] = field(default_factory=lambda: list(ROAST_MODES))
    selected_mode: str = "normal"
    active: bool = False
    _start_ts: float = 0.0
    _end_ts: float = 0.0
    _next_tick_ts: float = 0.0
    _highlight_index: int = 0

    def __post_init__(self) -> None:
        if not self.modes:
            self.modes = list(ROAST_MODES)
        self.modes = [normalize_roast_mode(m) for m in self.modes]
        self.selected_mode = normalize_roast_mode(self.selected_mode)
        if self.selected_mode not in self.modes:
            self.selected_mode = self.modes[0]
        self._highlight_index = self.modes.index(self.selected_mode)

    @property
    def highlighted_mode(self) -> str:
        return self.modes[self._highlight_index]

    def spin(self, now: float, duration_seconds: float) -> None:
        duration = max(0.4, duration_seconds)
        self.active = True
        self._start_ts = now
        self._end_ts = now + duration
        self._next_tick_ts = now
        self._highlight_index = self.modes.index(self.selected_mode)

    def update(self, now: float) -> Optional[str]:
        if not self.active:
            return None

        if now >= self._next_tick_ts:
            spin_window = self._end_ts - self._start_ts
            progress = 0.0
            if spin_window > 0:
                progress = min(1.0, max(0.0, (now - self._start_ts) / spin_window))

            self._highlight_index = (self._highlight_index + random.randint(1, 2)) % len(
                self.modes
            )
            interval = 0.05 + (progress**2) * 0.22
            self._next_tick_ts = now + interval

        if now >= self._end_ts:
            self.active = False
            self.selected_mode = self.highlighted_mode
            return self.selected_mode

        return None


def draw_mode_wheel(frame, wheel: ModeWheel) -> None:
    height, width = frame.shape[:2]
    radius = max(72, min(height, width) // 7)
    center = (width - radius - 24, radius + 24)
    mode_count = len(wheel.modes)

    for idx, mode in enumerate(wheel.modes):
        start_angle = int((idx * 360) / mode_count)
        end_angle = int(((idx + 1) * 360) / mode_count)
        is_highlight = wheel.active and idx == wheel._highlight_index
        is_selected = (not wheel.active) and mode == wheel.selected_mode

        color = (72, 72, 72)
        if is_highlight:
            color = (0, 170, 255)
        elif is_selected:
            color = (0, 165, 80)

        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            -90,
            start_angle,
            end_angle,
            color,
            -1,
        )
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            -90,
            start_angle,
            end_angle,
            (30, 30, 30),
            2,
        )

        angle_mid = math.radians(-90 + ((idx + 0.5) * 360 / mode_count))
        text_x = int(center[0] + math.cos(angle_mid) * radius * 0.62) - 28
        text_y = int(center[1] + math.sin(angle_mid) * radius * 0.62) + 6
        cv2.putText(
            frame,
            WHEEL_LABELS_SHORT.get(mode, mode[:6]),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    cv2.circle(frame, center, int(radius * 0.34), (20, 20, 20), -1)
    center_text = "SPIN" if wheel.active else "LOCK"
    cv2.putText(
        frame,
        center_text,
        (center[0] - 26, center[1] + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    caption = "Mode wheel: spinning..." if wheel.active else f"Mode: {mode_label(wheel.selected_mode)}"
    cv2.putText(
        frame,
        caption,
        (center[0] - radius, center[1] + radius + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
