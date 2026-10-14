import pytest
from PIL import Image

from leanwatch.screen import Geometry, ScreenCapture, fit_size, jpeg_data_url


def test_fit_size_never_upscales():
    assert fit_size(800, 600, 1366, 1_100_000) == (800, 600)


@pytest.mark.parametrize("w,h", [(1920, 1080), (2560, 1440), (3840, 2160), (1440, 900), (1080, 1920)])
def test_fit_size_respects_both_limits_and_aspect(w, h):
    sw, sh = fit_size(w, h, 1366, 1_100_000)
    assert max(sw, sh) <= 1366
    assert sw * sh <= 1_100_000
    assert abs(sw / sh - w / h) < 0.01


def test_coordinates_round_trip():
    g = Geometry.for_screen(1920, 1080)
    for x, y in [(0, 0), (960, 540), (1919, 1079), (123, 456)]:
        sx, sy = g.to_shot(x, y)
        bx, by = g.to_screen(sx, sy)
        assert abs(bx - x) <= 2 and abs(by - y) <= 2


def test_to_screen_clamps_inside_the_display():
    g = Geometry.for_screen(1920, 1080)
    assert g.to_screen(-50, -50) == (0, 0)
    assert g.to_screen(10_000, 10_000) == (1919, 1079)


def test_hidpi_raw_capture_maps_through_logical_size():
    # A Retina screen: pyautogui reports 1440x900 but mss captures 2880x1800.
    g = Geometry.for_screen(1440, 900)
    cap = ScreenCapture(g)
    cap._grab_raw = lambda: Image.new("RGB", (2880, 1800), "white")
    shot = cap.screenshot()
    assert shot.size == (g.shot_w, g.shot_h)
    # The centre of the screenshot is the centre of the logical screen.
    assert g.to_screen(g.shot_w / 2, g.shot_h / 2) == (720, 450)


def test_zoom_crops_the_right_region_and_fits_limits():
    g = Geometry.for_screen(1920, 1080)
    raw = Image.new("RGB", (1920, 1080), "black")
    raw.paste((255, 0, 0), (0, 0, 960, 540))  # top-left quarter is red
    cap = ScreenCapture(g)
    cap._grab_raw = lambda: raw
    z = cap.zoom([0, 0, g.shot_w / 4, g.shot_h / 4])
    assert z.width <= g.shot_w and z.height <= g.shot_h
    assert z.getpixel((z.width // 2, z.height // 2)) == (255, 0, 0)
    with pytest.raises(ValueError):
        cap.zoom([10, 10, 5, 20])


def test_preview_is_a_small_jpeg_data_url():
    url = jpeg_data_url(Image.new("RGB", (1366, 768), "blue"), width=320)
    assert url.startswith("data:image/jpeg;base64,")
