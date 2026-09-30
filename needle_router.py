"""Needle 2 action router with two backends.

CTypesBackend: in-process `needle.Needle` (pip package). Works on
  Linux/macOS/Windows where the engine .so auto-downloads.
ServeBackend:  talks HTTP to a `needle --serve` endpoint. This is the
  Termux/Android path: the pip auto-fetch grabs a glibc linux-aarch64
  .so which cannot load on Android's bionic libc, so on phones we run
  the official android-arm64 `needle` CLI binary as a server instead.

Both backends return the same envelope: function_calls + confidence.
`python needle_router.py --dump-tools tools.json` writes the tools.json
file the CLI --serve mode needs.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import needle  # noqa: E402
import tools  # noqa: E402


# The serve backend talks to 127.0.0.1. urllib honours *_proxy env vars, so
# without this the /reset and /complete calls can be sent to the egress
# proxy (which cannot route back into this machine) whenever no_proxy does
# not cover localhost -- e.g. in the sandbox where no_proxy must stay
# unset for Hugging Face downloads. Bypass proxies for loopback always.
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class CTypesBackend:
    name = "ctypes"

    def __init__(self, system_prompt):
        self._needle = needle.Needle(
            tools=list(tools.REGISTRY.values()), system=system_prompt
        )

    def complete(self, text):
        return self._needle.complete(text)

    def close(self):
        try:
            self._needle.close()
        except Exception:
            pass


class ServeBackend:
    name = "serve"

    def __init__(self, bin_path, port, system_prompt):
        self.bin = bin_path
        self.port = port
        self.proc = None
        if not self._port_open():
            self._spawn(system_prompt)
            self._wait_ready()

    def _port_open(self):
        # Readiness probe: a real HTTP POST /reset. A bare TCP connect can
        # confuse the tiny server (it resets the next request), so we speak
        # HTTP from the start. Any HTTP response code counts as "up".
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/reset",
                data=b"{}", method="POST",
                headers={"Content-Type": "application/json"},
            )
            with _LOCAL_OPENER.open(req, timeout=5):
                return True
        except urllib.error.HTTPError:
            return True  # server answered -> it is up
        except Exception:
            return False

    def _spawn(self, system_prompt):
        tmp = tempfile.mkdtemp(prefix="vq-needle-")
        tools_json = os.path.join(tmp, "tools.json")
        tools.dump_tools_json(tools_json)
        system_txt = os.path.join(tmp, "system.txt")
        with open(system_txt, "w", encoding="utf-8") as fh:
            fh.write(system_prompt)
        self.proc = subprocess.Popen(
            [self.bin, "--tools", tools_json, "--system", system_txt,
             "--serve", "--port", str(self.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _wait_ready(self, timeout=180):
        start = time.time()
        while time.time() - start < timeout:
            if self._port_open():
                return
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("needle serve process exited during startup")
            time.sleep(1.0)
        # don't leave an orphan behind
        try:
            self.proc.terminate()
        except Exception:
            pass
        raise RuntimeError(f"needle serve did not open port {self.port} in time")

    def complete(self, text):
        body = json.dumps({"input": text}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/complete",
            data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        last = None
        for attempt in range(2):  # one retry: first request after boot can reset
            try:
                with _LOCAL_OPENER.open(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last = exc
                time.sleep(2)
        raise RuntimeError(f"needle serve call failed: {last}")

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def find_needle_bin(explicit=""):
    if explicit and os.path.isfile(explicit):
        return explicit
    found = shutil.which("needle")
    if found:
        return found
    return ""


def build_backend(cfg):
    mode = (cfg["router"].get("backend") or "auto").lower()
    system_prompt = cfg["router"].get("system_prompt", "")
    port = int(cfg["router"].get("serve_port", 8080))

    def try_ctypes():
        return CTypesBackend(system_prompt)

    def try_serve():
        bin_path = find_needle_bin(cfg["router"].get("needle_bin", ""))
        if not bin_path:
            raise RuntimeError(
                "serve backend needs the `needle` binary (VQ config router.needle_bin "
                "or PATH). On Termux see phone-setup.md."
            )
        return ServeBackend(bin_path, port, system_prompt)

    if mode == "ctypes":
        return try_ctypes()
    if mode == "serve":
        return try_serve()
    # auto
    try:
        return try_ctypes()
    except Exception as exc:
        print(f"[router] ctypes backend failed ({exc}); trying serve backend")
        return try_serve()


class Router:
    """route(request) -> (tool_name | None, args dict, confidence float)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = build_backend(cfg)

    def route(self, request):
        try:
            resp = self.backend.complete(request)
        except Exception as exc:
            return None, {}, 0.0, f"router error: {exc}"
        calls = resp.get("function_calls") or []
        conf = resp.get("confidence")
        try:
            conf = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        if resp.get("type") != "call" or not calls:
            return None, {}, conf, "no tool call produced"
        call = calls[0]
        name = call.get("name")
        if name not in tools.REGISTRY:
            return None, {}, conf, f"unknown tool: {name}"
        args = call.get("arguments") or {}
        return name, args, conf, ""

    def close(self):
        self.backend.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dump-tools":
        path = tools.dump_tools_json(sys.argv[2])
        print("wrote", path)
    else:
        print("usage: python needle_router.py --dump-tools tools.json")
