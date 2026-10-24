"""Screen frames from a PipeWire screen-cast stream, for LeanWatch on Wayland.

Run by leanwatch/wayland.py with the system's own Python (it has GStreamer's bindings; LeanWatch's
Python, e.g. Anaconda's, usually doesn't):  python3 pw_frames.py <pipewire fd> <stream node id>

Protocol on stdin/stdout (one request at a time):
  -> "ready\n" once the stream runs
  <- "grab <seconds> <since>\n"   wait up to <seconds> for a frame taken after <since> (a time.monotonic()
     value; -1 means after this request), then send the newest
  -> "frame <width> <height> <stride>\n" followed by stride*height bytes of BGRx pixels,
     or "none\n" if no frame came yet (the desktop only sends one when something on screen changes)
  <- "quit\n"
Standalone on purpose: it must not import anything from LeanWatch.
"""

import sys
import threading
import time


def main() -> int:
    import signal

    # Ctrl+C in LeanWatch's terminal reaches this helper too. LeanWatch ends it (its stdin closes), so a Ctrl+C
    # here would only print a confusing traceback.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    fd, node = int(sys.argv[1]), sys.argv[2]
    pipeline = Gst.parse_launch(
        f"pipewiresrc fd={fd} path={node} always-copy=true do-timestamp=true ! "
        "videoconvert ! video/x-raw,format=BGRx ! "
        "appsink name=sink max-buffers=1 drop=true sync=false emit-signals=false")
    sink = pipeline.get_by_name("sink")
    latest = {"sample": None, "seq": 0, "time": 0.0}
    cond = threading.Condition()
    failed = []

    def pull():
        while True:
            sample = sink.emit("pull-sample")         # blocks until a frame (or end of stream)
            if sample is None:
                with cond:
                    failed.append("the stream ended")
                    cond.notify_all()
                return
            with cond:
                latest.update(sample=sample, seq=latest["seq"] + 1, time=time.monotonic())
                cond.notify_all()

    def start():
        # Blocks until the desktop agrees on a format, which can wait for the first change on screen.
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            with cond:
                failed.append("could not start the stream")
                cond.notify_all()
            return
        pull()

    threading.Thread(target=start, daemon=True).start()
    out = sys.stdout.buffer
    out.write(b"ready\n")
    out.flush()

    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == "quit":
            break
        if parts[0] != "grab":
            continue
        wait = float(parts[1]) if len(parts) > 1 else 0.0
        since = float(parts[2]) if len(parts) > 2 else -1.0
        if since < 0:
            since = time.monotonic()
        with cond:
            if wait > 0:      # the screen only sends a frame when something changed: don't wait forever
                cond.wait_for(lambda: latest["time"] > since or failed, timeout=wait)
            sample = latest["sample"]
        if sample is None:
            out.write(f"none {failed[0] if failed else ''}\n".encode())
            out.flush()
            continue
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            out.write(b"error could not read the frame\n")
            out.flush()
            continue
        try:
            data = bytes(info.data)
        finally:
            buf.unmap(info)
        stride = len(data) // h if h else 0
        out.write(f"frame {w} {h} {stride}\n".encode())
        out.write(data[:stride * h])
        out.flush()
    pipeline.set_state(Gst.State.NULL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
