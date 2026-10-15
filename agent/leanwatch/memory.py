"""How much memory this computer has, how much a model needs, and which weight precision fits on a CPU.

No torch here: the bubble's process, the model's process and --check all use it.
"""

from __future__ import annotations

import re

# Parameter counts of the models LeanWatch suggests. Others are read from their name ("UI-TARS-7B-DPO").
MODELS = {
    "ByteDance-Seed/UI-TARS-1.5-7B": 8.3e9,
    "ByteDance-Seed/UI-TARS-2B-SFT": 2.2e9,
}
SMALL_MODEL = "ByteDance-Seed/UI-TARS-2B-SFT"
CPU_DTYPES = ("auto", "float32", "bfloat16")
# On top of the weights: the screenshot's activations, the libraries themselves and the operating system.
HEADROOM_GB = {"float32": 3.5, "bfloat16": 1.5}
BYTES = {"float32": 4, "bfloat16": 2}


def ram_gb() -> float | None:
    """Total memory in GB (10^9 bytes), or None if it can't be read."""
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        return _meminfo("MemTotal")


def available_gb() -> float | None:
    """Memory free for a new program right now (without pushing others into swap)."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return _meminfo("MemAvailable")


def _meminfo(key: str) -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    return None


def param_count(model_id: str) -> float | None:
    """Parameters of a model: from the table, else from a size in its name ("2B", "7B", "3.5B"), else None."""
    if model_id in MODELS:
        return MODELS[model_id]
    sizes = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z0-9])", model_id)
    return max(float(s) for s in sizes) * 1e9 if sizes else None


def need_gb(params: float, dtype: str) -> float:
    """Memory a model needs on a CPU with this weight precision."""
    return params * BYTES[dtype] / 1e9 + HEADROOM_GB[dtype]


def short_name(model_id: str) -> str:
    return model_id.rstrip("/\\").replace("\\", "/").split("/")[-1]


def choose_cpu_dtype(model_id: str, setting: str = "auto", ram: float | None = None) -> tuple[str, str]:
    """Weight precision for a model on the CPU, and one line saying which and why.

    "auto" uses 32-bit weights only when they fit with room to spare; otherwise 16-bit (bfloat16) weights
    take half the memory. UI-TARS is published in bfloat16, and the arithmetic stays 32-bit
    (local_model.float32_arithmetic), so the answers are the same, only about 10% slower.
    """
    if setting not in CPU_DTYPES:
        raise ValueError(f"Unknown CPU precision {setting!r}: use auto, float32 or bfloat16 "
                         "(--cpu-dtype or LEANWATCH_CPU_DTYPE).")
    if setting == "float32":
        return "float32", "Loading the model in 32-bit (float32), as set with --cpu-dtype."
    if setting == "bfloat16":
        return "bfloat16", "Loading the model in 16-bit (bfloat16), as set with --cpu-dtype: it needs half the memory."
    params = param_count(model_id)
    ram = ram_gb() if ram is None else ram
    name = short_name(model_id)
    if params is None:
        return "bfloat16", (f"Loading {name} in 16-bit (bfloat16) to save memory: I can't tell its size. "
                            "With plenty of memory, --cpu-dtype float32 is a little more precise.")
    need32 = need_gb(params, "float32")
    if ram is None:
        return "bfloat16", (f"Loading {name} in 16-bit (bfloat16): I can't tell how much memory this computer has, "
                            f"and 32-bit would need about {need32:.0f} GB.")
    if ram >= need32:
        return "float32", (f"Loading {name} in 32-bit (float32): this computer's {ram:.0f} GB of memory "
                           f"has room for it (about {need32:.0f} GB).")
    return "bfloat16", (f"Loading {name} in 16-bit (bfloat16) to fit in this computer's {ram:.0f} GB of memory "
                        f"(about {need_gb(params, 'bfloat16'):.0f} GB; 32-bit would need about {need32:.0f} GB).")


def memory_advice(model_id: str, cpu_dtype: str = "auto") -> str:
    """What to try after the model ran out of memory, most helpful first."""
    dtype = choose_cpu_dtype(model_id, cpu_dtype)[0]
    params = param_count(model_id)
    tips = ["close big apps such as your web browser"]
    if params is None or params > MODELS[SMALL_MODEL]:
        tips.append("use the small model (--small)")
    if dtype != "bfloat16":
        tips.append("load it in 16-bit (--cpu-dtype bfloat16)")
    text = ", ".join(tips[:-1]) + (" or " if len(tips) > 1 else "") + tips[-1]
    if params:
        bits = "16-bit" if dtype == "bfloat16" else "32-bit"
        text += f". {short_name(model_id)} in {bits} needs about {need_gb(params, dtype):.0f} GB on a CPU"
    return text[0].upper() + text[1:] + "."
