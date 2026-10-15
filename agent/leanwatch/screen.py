"""Screenshots and coordinate mapping.

The model sees a downscaled screenshot. Its coordinates are in that image's
pixels, so every click is mapped back to the screen's own coordinate system
(the one pyautogui uses). Basing the scale on the logical screen size handles
HiDPI and Retina displays, where the raw capture has 2x more pixels.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass


def fit_size(width: int, height: int, long_edge: int, max_pixels: int) -> tuple[int, int]:
    """Largest size with the same aspect ratio that fits both limits. Never upscales."""
    scale = min(1.0, long_edge / max(width, height), math.sqrt(max_pixels / (width * height)))
    return max(1, int(width * scale)), max(1, int(height * scale))


@dataclass(frozen=True)
class Geometry:
    """Logical screen size (mouse coordinates) and the size the model sees."""

    screen_w: int
    screen_h: int
    shot_w: int
    shot_h: int

    @classmethod
    def for_screen(cls, screen_w: int, screen_h: int, long_edge: int = 1366, max_pixels: int = 1_100_000) -> "Geometry":
        shot_w, shot_h = fit_size(screen_w, screen_h, long_edge, max_pixels)
        return cls(screen_w, screen_h, shot_w, shot_h)

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Model (screenshot) coordinates -> real mouse coordinates, clamped on screen."""
        sx = round(x * self.screen_w / self.shot_w)
        sy = round(y * self.screen_h / self.shot_h)
        return min(max(sx, 0), self.screen_w - 1), min(max(sy, 0), self.screen_h - 1)

    def to_shot(self, x: float, y: float) -> tuple[int, int]:
        """Real mouse coordinates -> model (screenshot) coordinates."""
        return round(x * self.shot_w / self.screen_w), round(y * self.shot_h / self.screen_h)

    def fraction(self, x: float, y: float) -> tuple[float, float]:
        return x / self.screen_w, y / self.screen_h


def png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def jpeg_data_url(image, width: int = 720, quality: int = 62) -> str:
    """Small preview for the web page (never sent to the model)."""
    from PIL import Image

    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


def image_block(b64_png: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_png}}


class ScreenCapture:
    """Captures the primary monitor with mss and scales it for the model.

    `shield` (optional, from the desktop bubble) has a cloak() context manager that keeps
    LeanWatch's own windows out of the picture, so the model never sees or clicks them.
    """

    def __init__(self, geometry: Geometry, shield=None):
        self.geometry = geometry
        self.shield = shield
        self._sct = None
        self.last_image = None  # last scaled screenshot, for previews

    def _grab_raw(self):
        from PIL import Image
        import mss

        if self._sct is None:
            self._sct = mss.mss()
        mon = self._sct.monitors[1]
        if self.shield is not None:
            with self.shield.cloak():
                raw = self._sct.grab(mon)
        else:
            raw = self._sct.grab(mon)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def screenshot(self):
        """Return the screenshot resized to the model's size (a PIL image)."""
        from PIL import Image

        g = self.geometry
        img = self._grab_raw().resize((g.shot_w, g.shot_h), Image.LANCZOS)
        self.last_image = img
        return img

    def zoom(self, region: list[float]):
        """Crop a region given in screenshot coordinates at full resolution."""
        from PIL import Image

        g = self.geometry
        x0, y0, x1, y1 = region
        if x1 <= x0 or y1 <= y0:
            raise ValueError("zoom region must have x1 > x0 and y1 > y0")
        raw = self._grab_raw()
        kx, ky = raw.width / g.shot_w, raw.height / g.shot_h
        box = (
            max(0, round(x0 * kx)), max(0, round(y0 * ky)),
            min(raw.width, round(x1 * kx)), min(raw.height, round(y1 * ky)),
        )
        crop = raw.crop(box)
        # Fit the crop into the screenshot's box: small regions are enlarged so
        # details are legible, and the result never exceeds the model's limits.
        s = min(g.shot_w / crop.width, g.shot_h / crop.height)
        return crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
