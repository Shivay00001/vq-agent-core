"""Local system introspection tools (Linux/Termux: /proc based)."""
import datetime
import json
import os

import needle


@needle.tool
def get_time() -> str:
    """Return the current local date and time."""
    return json.dumps({"now": datetime.datetime.now().isoformat(timespec="seconds")})


@needle.tool
def sysinfo() -> str:
    """Return THIS device's hardware stats: CPU count, load average, memory.

    Use for questions about the local machine/phone/computer itself —
    e.g. "how loaded is this device", "show CPU and RAM", "system load".
    Works on Linux and Termux (reads /proc). No external calls.
    """
    info = {"cpu_count": os.cpu_count()}
    try:
        info["load_avg_1_5_15"] = list(os.getloadavg())
    except OSError:
        info["load_avg_1_5_15"] = None
    mem = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    k, v = line.split(":")
                    mem[k.strip()] = v.strip()
    except OSError:
        pass
    info["memory"] = mem
    return json.dumps(info)
