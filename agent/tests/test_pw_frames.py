"""The real screen reader helper (leanwatch/pw_frames.py) talking to LeanWatch's PipeWireFrames, with a
stand-in for GStreamer (a fake `gi` package written to a temporary folder): both sides of the protocol."""

import os
import sys
import time
from pathlib import Path

import pytest

from leanwatch import wayland

FAKE_GST = '''
"""Just enough of GStreamer for pw_frames.py. The scenario is the pipeline's path= (the stream node)."""
import re
import threading
import time

W, H, STRIDE = 3, 2, 16
ROW0 = bytes([0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0]) + b"\\xab" * 4
ROW1 = bytes([255, 255, 255, 0, 0, 0, 0, 0, 128, 128, 128, 0]) + b"\\xcd" * 4


class State:
    PLAYING, NULL = "PLAYING", "NULL"


class StateChangeReturn:
    SUCCESS, FAILURE, ASYNC = "SUCCESS", "FAILURE", "ASYNC"


class MapFlags:
    READ = 1


def init(argv):
    pass


class _Structure:
    def get_value(self, key):
        return {"width": W, "height": H}[key]


class _Caps:
    def get_structure(self, i):
        return _Structure()


class _Buffer:
    def __init__(self, data):
        self.data = data

    def map(self, flags):
        return True, type("Info", (), {"data": self.data})()

    def unmap(self, info):
        pass


class _Sample:
    def __init__(self, data):
        self.data = data

    def get_caps(self):
        return _Caps()

    def get_buffer(self):
        return _Buffer(self.data)


class _Sink:
    def __init__(self, scenario):
        self.scenario = scenario
        self.pulls = 0

    def emit(self, signal):
        assert signal == "pull-sample"
        self.pulls += 1
        if self.scenario == "ended":
            return None
        if self.pulls == 1:
            if self.scenario == "late":
                time.sleep(0.15)
            return _Sample(ROW0 + ROW1)
        while True:                       # no more changes on the screen: no more frames
            time.sleep(60)


class _Pipeline:
    def __init__(self, scenario, fd):
        self.scenario, self.fd = scenario, fd
        self.sink = _Sink(scenario)

    def get_by_name(self, name):
        assert name == "sink"
        return self.sink

    def set_state(self, state):
        if state == State.PLAYING and self.scenario == "fail-start":
            return StateChangeReturn.FAILURE
        return StateChangeReturn.SUCCESS


def parse_launch(description):
    assert "appsink name=sink" in description and "format=BGRx" in description
    fd = int(re.search(r"pipewiresrc fd=(\\d+)", description).group(1))
    import os
    os.fstat(fd)                          # the PipeWire connection was passed to this process
    return _Pipeline(re.search(r"path=(\\S+)", description).group(1), fd)
'''


@pytest.fixture
def real_helper(tmp_path):
    """A 'system Python' whose gi.repository.Gst is the fake above."""
    pkg = tmp_path / "fakegi" / "gi"
    (pkg / "repository").mkdir(parents=True)
    (pkg / "__init__.py").write_text("def require_version(name, version):\n    assert name == 'Gst'\n")
    (pkg / "repository" / "__init__.py").write_text("")
    (pkg / "repository" / "Gst.py").write_text(FAKE_GST)
    python = tmp_path / "system-python"
    python.write_text(f"#!/bin/sh\nPYTHONPATH={tmp_path / 'fakegi'} exec {sys.executable} \"$@\"\n")
    python.chmod(0o755)
    made = []

    def start(scenario):
        remote = type("Remote", (), {})()
        remote.node = scenario
        fd, other = os.pipe()
        remote.pipewire_fd = lambda: fd
        try:
            frames = wayland.PipeWireFrames(remote, str(python))
        finally:
            os.close(other)
        made.append(frames)
        return frames

    yield start
    for frames in made:
        frames.close()


def test_helper_is_standalone():
    source = Path(wayland._HELPER).read_text()
    assert "leanwatch" not in source.split('"""', 2)[2]       # nothing from LeanWatch after the docstring
    assert wayland._HELPER.name == "pw_frames.py" and wayland._HELPER.exists()


def first_frame(frames):
    deadline = time.monotonic() + 5
    while True:                          # the first frame arrives on the helper's own thread
        try:
            return frames.grab()
        except wayland.PortalError:
            assert time.monotonic() < deadline
            time.sleep(0.01)


def test_a_frame_from_the_real_helper(real_helper):
    frames = real_helper("one-frame")
    img = first_frame(frames)
    assert img.size == (3, 2)
    assert list(img.tobytes()) == [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 128, 128, 128]
    img = frames.grab(since=time.monotonic() - 60)            # that frame is newer than a minute ago
    assert img.size == (3, 2)
    frames.close()
    assert frames.proc.returncode == 0


def test_since_waits_then_sends_the_newest_frame(real_helper):
    frames = real_helper("one-frame")
    first_frame(frames)
    started = time.monotonic()
    img = frames.grab(since=time.monotonic(), wait=0.15)      # nothing newer comes: the newest after `wait`
    assert img.size == (3, 2)
    assert 0.12 <= time.monotonic() - started < 3


def test_no_frame_yet_then_the_nudge_brings_one(real_helper):
    frames = real_helper("late")                              # the first frame comes 0.15 s after start
    nudges = []
    img = frames.grab(nudge=lambda: nudges.append(1))
    assert nudges == [1] and img.size == (3, 2)


@pytest.mark.parametrize("scenario,reason", [("ended", "the stream ended"), ("fail-start", "could not start the stream")])
def test_a_stream_that_fails_says_why(scenario, reason, real_helper):
    frames = real_helper(scenario)
    with pytest.raises(wayland.PortalError, match=f"none {reason}"):
        frames.grab(nudge=lambda: None)
