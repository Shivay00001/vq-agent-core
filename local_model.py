"""Local-model slot: OpenAI-compatible chat client (Ollama/LM Studio/llama.cpp server).

Every call path degrades cleanly when no endpoint is reachable — the agent
then falls back to factual structured summaries instead of fake LLM prose.
"""
import json
import os
import urllib.parse
import urllib.request

BASE_URL = os.environ.get("VQ_LOCAL_MODEL_URL", "http://localhost:11434/v1").rstrip("/")
MODEL = os.environ.get("VQ_LOCAL_MODEL_NAME", "qwen2.5:1.5b")
TIMEOUT = int(os.environ.get("VQ_LOCAL_MODEL_TIMEOUT", "120"))

# The local-model endpoint is loopback or a LAN host -- it must never go
# through an egress proxy (same trap as needle_router's serve backend).
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _is_local_host(url):
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if host in ("localhost", "ip6-localhost", "::1"):
        return True
    if host.startswith("127."):
        return True
    if host.startswith(("10.", "192.168.")):
        return True
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
            if 16 <= second <= 31:
                return True
        except (ValueError, IndexError):
            pass
    return False


def _open(req, timeout):
    if _is_local_host(req.full_url):
        return _NO_PROXY_OPENER.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


class LocalModel:
    def __init__(self, base_url=None, model=None, timeout=None):
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.model = model or MODEL
        self.timeout = timeout or TIMEOUT

    def available(self):
        """Probe the endpoint. Short timeout; never raises."""
        try:
            req = urllib.request.Request(
                self.base_url + "/models", headers={"User-Agent": "VQAgentCore/1.0"}
            )
            with _open(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat(self, messages, max_tokens=512, temperature=0.3):
        """Chat completion. Raises RuntimeError (not fake text) on failure."""
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with _open(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"local model call failed: {exc}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"local model returned unexpected payload: {data!r}"[:300])
