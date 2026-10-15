"""A stand-in for LocalModel that the model process loads in milliseconds (tests only).

The child process imports it as "fake_model:FakeModel": spawn gives the child this folder on sys.path.
"""

import os
import signal
import sys
import time

from leanwatch.local_model import ModelAnswer


class FakeModel:
    def __init__(self, model_id, status=None, load_s=0.0, fail=None, die_on_load=False, die_on_answer=False,
                 steps=0, step_s=0.01, ignore_stop=False, say=None, **options):
        if status:
            status(f"Loading {model_id} (fake) in 16-bit.")
        if say:
            print(say, file=sys.stderr, flush=True)
        if die_on_load:
            os.kill(os.getpid(), signal.SIGKILL)
        if fail == "torchvision":
            # What transformers' lazy import shows when torchvision's compiled operators are missing.
            try:
                raise RuntimeError("operator torchvision::nms does not exist")
            except RuntimeError as err:
                raise ModuleNotFoundError("Could not import module 'AutoModelForImageTextToText'. "
                                          "Are this object's requirements defined correctly?") from err
        time.sleep(load_s)
        self.model_id = model_id
        self.coords = "rel1000"
        self.info = {"dtype": "bfloat16", "pid": os.getpid(), "options": sorted(options),
                     "modules": sorted(m for m in sys.modules if m.split(".")[0] in ("leanwatch", "PySide6", "pyautogui"))}
        self.die_on_answer = die_on_answer
        self.steps, self.step_s, self.ignore_stop = steps, step_s, ignore_stop

    def answer(self, messages, images, stop=None):
        if self.die_on_answer:
            os.kill(os.getpid(), signal.SIGKILL)       # what the kernel's out-of-memory killer does
        done = 0
        while done < self.steps and (self.ignore_stop or not (stop and stop.is_set())):
            time.sleep(self.step_s)                    # one "token" at a time, like generate()
            done += 1
        task = messages[-1]["content"][-1]["text"]
        return ModelAnswer(f"Action: wait() task={task!r} tokens={done} images={[im.size for im in images]}", 1000, 800)
