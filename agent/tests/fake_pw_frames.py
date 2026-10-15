"""Stand-in for leanwatch/pw_frames.py in the unit tests: the same stdin/stdout protocol, no GStreamer.

    python fake_pw_frames.py <pipewire fd> <mode>

Modes (passed where pw_frames.py gets the stream's node id):
  frames       answer every grab with a 3 x 2 frame whose rows are padded (stride 16 > 3 * 4)
  none-once    answer the first grab with "none ...", then frames
  none         always answer "none ..."
  exit         say "ready", then exit at once
  silent-exit  say "ready", read one request, exit without answering
  broken       answer with a frame header and only part of the pixels, then exit
  not-ready    say "error no stream" instead of "ready", then exit
  slow-first   like frames, but answer the first grab only after 0.25 s; the red of the first pixel
               of every frame is the request's <since> (so a test can tell whose answer it got)

Every request line is appended to the file named by LEANWATCH_FAKE_LOG (first line: whether the
file descriptor was passed to this process), so the tests can check what LeanWatch asked for.
Standalone (standard library only), like the real helper.
"""

import os
import sys
import time

W, H, STRIDE = 3, 2, 16
# BGRx pixels, then 4 bytes of row padding that must never show up in the picture.
ROW0 = bytes([0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0]) + b"\xab\xab\xab\xab"        # red, green, blue
ROW1 = bytes([255, 255, 255, 0, 0, 0, 0, 0, 128, 128, 128, 0]) + b"\xcd\xcd\xcd\xcd"   # white, black, grey
FRAME = ROW0 + ROW1


def main() -> int:
    fd, mode = int(sys.argv[1]), sys.argv[2]
    log_path = os.environ.get("LEANWATCH_FAKE_LOG")

    def note(text: str) -> None:
        if log_path:
            with open(log_path, "a") as f:
                f.write(text + "\n")

    try:
        os.fstat(fd)
        note("fd-ok")
    except OSError:
        note("fd-missing")
    out = sys.stdout.buffer
    if mode == "not-ready":
        out.write(b"error no stream\n")
        out.flush()
        return 0
    out.write(b"ready\n")
    out.flush()
    if mode == "exit":
        return 0
    count = 0
    for line in sys.stdin:
        note(line.strip())
        parts = line.split()
        if not parts or parts[0] == "quit":
            break
        if parts[0] != "grab":
            continue
        count += 1
        if mode == "silent-exit":
            return 0
        if mode == "none" or (mode == "none-once" and count == 1):
            out.write(b"none no frame yet\n")
            out.flush()
            continue
        frame = FRAME
        if mode == "slow-first":
            if count == 1:
                time.sleep(0.25)
            since = float(parts[2]) if len(parts) > 2 else -1.0
            frame = bytearray(FRAME)
            frame[2] = max(0, int(since)) % 256
            frame = bytes(frame)
        out.write(f"frame {W} {H} {STRIDE}\n".encode())
        if mode == "broken":
            out.write(frame[:10])
            out.flush()
            return 0
        out.write(frame)
        out.flush()
    note("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
