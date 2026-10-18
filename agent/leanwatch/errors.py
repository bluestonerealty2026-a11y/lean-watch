"""The real reason the model could not load, in plain words, with the fix when we know it.

The real error is often hidden: transformers turns a failed import into "Could not import module
'AutoModelForImageTextToText'. Are this object's requirements defined correctly?" and chains the actual
error underneath. root_cause() digs it out. No torch here, so the bubble's process can use it too.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"

# torch release -> the torchvision release built for it. Each pair is the Requires-Dist of that torchvision's
# wheels on PyPI (torchvision 0.20.1 requires torch==2.5.1, ...; 0.29.1 requires torch>=2.14.0).
TORCHVISION_FOR_TORCH = {
    "2.0.1": "0.15.2", "2.1.0": "0.16.0", "2.1.1": "0.16.1", "2.1.2": "0.16.2",
    "2.2.0": "0.17.0", "2.2.1": "0.17.1", "2.2.2": "0.17.2", "2.3.0": "0.18.0", "2.3.1": "0.18.1",
    "2.4.0": "0.19.0", "2.4.1": "0.19.1", "2.5.0": "0.20.0", "2.5.1": "0.20.1", "2.6.0": "0.21.0",
    "2.7.0": "0.22.0", "2.7.1": "0.22.1", "2.8.0": "0.23.0", "2.9.0": "0.24.0", "2.9.1": "0.24.1",
    "2.10.0": "0.25.0", "2.11.0": "0.26.0", "2.12.0": "0.27.0", "2.12.1": "0.27.1", "2.13.0": "0.28.0",
    "2.14.0": "0.29.0", "2.14.1": "0.29.1",
}
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Import names whose pip package is called something else.
PIP_NAMES = {
    "cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn", "skimage": "scikit-image",
    "google": "protobuf", "attr": "attrs", "bs4": "beautifulsoup4", "dateutil": "python-dateutil",
    "qwen_vl_utils": "qwen-vl-utils", "flash_attn": "flash-attn", "Crypto": "pycryptodome", "jwt": "PyJWT",
    "serial": "pyserial", "Xlib": "python-xlib", "PySide6": "PySide6-Essentials",
}
# What requirements.txt installs: if one of these is missing, install them all.
OWN_PACKAGES = {"torch", "torchvision", "transformers", "accelerate", "peft", "PIL", "numpy", "safetensors",
                "tokenizers", "huggingface_hub", "psutil", "PySide6", "mss", "pyautogui", "aiohttp", "yaml"}

NUMPY_ABI = ("_ARRAY_API not found", "compiled using NumPy 1.x", "numpy.dtype size changed",
             "numpy.core.multiarray failed to import", "numpy.core._multiarray_umath", "Numpy is not available",
             "compiled against API version", "compiled against ABI version")
OUT_OF_MEMORY = ("MemoryError", "can't allocate memory", "Cannot allocate memory", "DefaultCPUAllocator",
                 "not enough memory", "out of memory", "OutOfMemoryError")
NO_NETWORK = ("Temporary failure in name resolution", "Failed to establish a new connection", "Max retries exceeded",
              "couldn't connect to", "ConnectTimeout", "ConnectError", "Network is unreachable",
              "Cannot find the requested files in the disk cache", "LocalEntryNotFoundError")


class Stopped(Exception):
    """The user pressed Stop."""


class ModelError(RuntimeError):
    """The model could not load or answer. The message is ready to show and the traceback is already printed."""


def chain(exc: BaseException) -> list[BaseException]:
    """The exception and the ones under it, outermost first, as Python prints them ("The above exception...")."""
    out = [exc]
    while True:
        e = out[-1]
        nxt = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
        if nxt is None or any(nxt is seen for seen in out):
            return out
        out.append(nxt)


def root_cause(exc: BaseException) -> BaseException:
    """The deepest exception in the chain: usually the real reason."""
    return chain(exc)[-1]


def one_line(text, limit: int = 240) -> str:
    """The first meaningful line of an error message, shortened for the bubble."""
    lines = [s.strip() for s in str(text).splitlines() if s.strip()]
    text = lines[0] if lines else ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def details_hint() -> str:
    """Where the full traceback went: the terminal, or the log file when LeanWatch started at login."""
    name = getattr(sys.stderr, "name", None)
    if isinstance(name, str) and name and not name.startswith("<"):
        home = str(Path.home())
        return f"The full error is in {'~' + name[len(home):] if name.startswith(home + os.sep) else name}."
    return "The full error is printed in the terminal."


def requirements_arg() -> str:
    req = str(REQUIREMENTS)
    return f'"{req}"' if " " in req else req


def in_conda() -> bool:
    return bool(os.environ.get("CONDA_PREFIX")) or (Path(sys.prefix) / "conda-meta").exists()


def clean_env_command() -> str:
    """A fresh environment with only what LeanWatch needs: the surest fix for a mixed-up install."""
    req = requirements_arg()
    # Without an NVIDIA GPU, torch's CPU build is a ~200 MB download instead of several GB of CUDA libraries.
    cpu = ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && "
           if sys.platform.startswith("linux") and not os.path.exists("/proc/driver/nvidia") else "")
    if in_conda():
        # PYTHONNOUSERSITE: packages from "pip install --user" (~/.local) stay out of it.
        return ("conda create -n leanwatch python=3.12 -y && conda activate leanwatch && "
                "conda env config vars set PYTHONNOUSERSITE=1 && conda activate leanwatch && "
                f"{cpu}pip install -r {req}")
    if os.name == "nt":
        return f"py -m venv %USERPROFILE%\\leanwatch-env && %USERPROFILE%\\leanwatch-env\\Scripts\\pip install -r {req}"
    return f"python3 -m venv ~/leanwatch-env && ~/leanwatch-env/bin/pip install -r {req}"


def installed(dist: str) -> str | None:
    """A package's installed version, read without importing it."""
    try:
        return importlib.metadata.version(dist)
    except Exception:
        return None


def _cpu_only_torch(version: str) -> bool:
    """On Linux a CPU-only torch needs torchvision's CPU build: the one on PyPI links to CUDA libraries."""
    if not sys.platform.startswith("linux"):
        return False
    if "+cpu" in version:
        return True
    try:
        spec = importlib.util.find_spec("torch")
        text = (Path(next(iter(spec.submodule_search_locations))) / "version.py").read_text()
        return re.search(r"^cuda\b[^=\n]*=\s*None", text, re.M) is not None
    except Exception:
        return False


def torchvision_fix() -> str:
    torch_v, tv_v = installed("torch"), installed("torchvision")
    if not torch_v:
        return f"torch is missing. Fix: pip install -r {requirements_arg()}"
    want = TORCHVISION_FOR_TORCH.get(torch_v.split("+")[0])
    index = f" --index-url {CPU_INDEX}" if _cpu_only_torch(torch_v) else ""
    if want and (tv_v or "").split("+")[0] != want:
        return f"torch {torch_v} and torchvision {tv_v or '(missing)'} don't match. Fix: pip install torchvision=={want}{index}"
    if want:
        return (f"torchvision {tv_v}'s compiled part doesn't load with torch {torch_v}. "
                f"Fix: pip install --force-reinstall --no-deps torchvision=={want}{index}")
    return (f"torch {torch_v} and torchvision {tv_v} don't match. Install the torchvision made for torch {torch_v} "
            "(see https://pytorch.org/get-started/previous-versions/).")


def pip_name(module: str) -> str:
    return PIP_NAMES.get(module, module.replace("_", "-"))


def _missing_module_fix(name: str) -> str:
    top = name.split(".")[0]
    if top.startswith("_") and top.lstrip("_") in getattr(sys, "stdlib_module_names", ()):
        return (f"This Python was built without its {top.lstrip('_')} module. "
                "Use a Python from python.org, your Linux distribution or conda.")
    try:
        present = top != name and importlib.util.find_spec(top) is not None
    except Exception:
        present = False
    if present:
        return f"{top} is installed incompletely ({name} is missing). Fix: pip install --force-reinstall {pip_name(top)}"
    if top in OWN_PACKAGES:
        return f"Fix: pip install -r {requirements_arg()}"
    return f"Fix: pip install {pip_name(top)}"


def known_fix(kind: str, message: str, model_id: str | None = None, cpu_dtype: str = "auto") -> tuple[str | None, bool]:
    """(the fix for a known problem or None, whether it is an install problem that a clean environment solves)."""
    text = f"{kind}: {message}"
    if re.search(r"torchvision::\w+ does not exist|custom C\+\+ ops|'torchvision' object has no attribute", text) or (
            "torchvision" in text and re.search(r"undefined symbol|cannot open shared object", text)):
        return torchvision_fix(), True
    if any(s in text for s in NUMPY_ABI):
        np_v = installed("numpy") or "?"
        fix = 'pip install "numpy<2"' if np_v.split(".")[0].isdigit() and int(np_v.split(".")[0]) >= 2 \
            else "pip install --upgrade numpy"
        return f"A package here was built for a different NumPy (NumPy {np_v} is installed). Fix: {fix}", True
    if re.search(r"module 'lib' has no attribute '\w+'", text):
        # pyOpenSSL older than the cryptography package it runs on. accelerate reaches it through boto3
        # (when boto3 is installed), so a broken pyOpenSSL stops the model from loading.
        return ("pyOpenSSL is too old for the installed cryptography package (accelerate loads it through boto3). "
                "Fix: pip install --upgrade pyopenssl"), True
    if "may only be initialized once per interpreter process" in text:
        return ("This follows an earlier failed import in the same program (see the first error above): "
                "fix that one and restart LeanWatch."), True
    if "partially initialized module" in text or "circular import" in text:
        return ("A package was left half-loaded by an earlier failed try, or it is installed badly. "
                "Restart LeanWatch; if this comes back, run: python -m leanwatch --check"), True
    m = re.search(r"No module named '([^']+)'", text)
    if m:
        return _missing_module_fix(m.group(1)), True
    m = re.search(r"cannot import name '([^']+)' from '([^']+)'", text)
    if m:
        return (f"{m.group(2).split('.')[0]} doesn't fit the other installed packages (too old or too new). "
                f"Fix: pip install --upgrade -r {requirements_arg()}"), True
    if re.search(r"undefined symbol|cannot open shared object|DLL load failed|Library not loaded", text):
        return "A compiled library doesn't match the rest of the install.", True
    if any(s in text for s in OUT_OF_MEMORY):
        if "CUDA" in text or "GPU" in text:
            return "The graphics card ran out of memory. Close other apps that use it, or try --4bit or --small.", False
        from .memory import memory_advice, ram_gb

        ram = ram_gb()
        has = f" (it has {ram:.0f} GB)" if ram else ""
        return f"The computer ran out of memory{has}. {memory_advice(model_id or '', cpu_dtype)}", False
    if "No space left on device" in text or "Errno 28" in text:
        return ("The disk is full. The first start downloads the model (several GB) to ~/.cache/huggingface: "
                "free some space and try again."), False
    if any(s in text for s in NO_NETWORK):
        return ("LeanWatch couldn't download the model. Check your internet connection: "
                "the first start downloads it from huggingface.co."), False
    if "is not a local folder and is not a valid model identifier" in text or "Repository Not Found" in text:
        return "Hugging Face has no model with that name. Check --model (or LEANWATCH_LOCAL_MODEL).", False
    return None, False


def explain_load_error(exc: BaseException, model_id: str | None = None, cpu_dtype: str = "auto",
                       doing: str = "load the model") -> str:
    """A short message for the bubble, one point per line: the real error, the fix if we know one, where the details are.

    Commands stand at the end of their line with no full stop, so they can be copied as they are.
    """
    root = root_cause(exc)
    fix, install = None, False
    for e in reversed(chain(exc)):          # the deepest error with a known fix wins
        try:
            fix, install = known_fix(type(e).__name__, str(e), model_id, cpu_dtype)
        except Exception:                   # explaining must never hide the error itself
            fix, install = None, False
        if fix:
            break
    first = f"Could not {doing}: {type(root).__name__}: {one_line(root) or '(no message)'}"
    lines = [first if first.endswith((".", "?", "!", "…")) else first + "."]
    if fix:
        lines.append(fix)
    if install:
        lines.append(f"If that doesn't help, a clean environment fixes most install problems: {clean_env_command()}")
    lines.append(details_hint())
    return "\n".join(lines)
