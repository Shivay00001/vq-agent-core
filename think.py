"""THINK layer: the local LLM as the understanding / reasoning /
composition engine (text, audio, video, documents).

Two entry points used by the agent loop:
  understand(request, media_paths) -> {"understanding", "actionable_request",
                                       "plan_hint", "media", "via"}
  compose(request, steps)         -> final answer string

When no local model is reachable both degrade HONESTLY:
  understand() returns the raw request with via="skipped" (logged),
  compose() returns a factual structured summary -- never fake LLM prose.

Multimodal note: what the THINK layer can actually "see" depends on the
model behind the OpenAI-compatible endpoint. Text files are inlined
(head); binary media (audio/video/images/scanned docs) are described by
metadata only, unless the configured model is multimodal-capable
(e.g. Qwen2.5-Omni-class / Gemma-3n-class -- see phone-setup.md).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_model import LocalModel  # noqa: E402

_TEXT_EXTS = {".txt", ".md", ".json", ".csv", ".log", ".py", ".yaml", ".yml"}


def describe_media(paths):
    """Describe input files for the THINK prompt. Never fakes content."""
    described = []
    for path in paths or []:
        if not path or not os.path.isfile(path):
            described.append({"path": path, "error": "file not found"})
            continue
        size = os.path.getsize(path)
        ext = os.path.splitext(path)[1].lower()
        entry = {"path": path, "bytes": size, "ext": ext or "unknown"}
        if ext in _TEXT_EXTS and size < 100_000:
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    entry["head"] = fh.read(4000)
            except OSError as exc:
                entry["error"] = str(exc)
        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".gif",
                     ".mp3", ".wav", ".m4a", ".mp4", ".pdf"):
            entry["note"] = ("binary media: described by metadata only -- "
                             "needs a multimodal-capable local model to inspect content")
        described.append(entry)
    return described


def understand(request, media_paths=None, lm=None):
    """THINK step 1: understand the request (+ any media) and reason.

    Returns a dict; 'actionable_request' is what ACT routes on.
    """
    lm = lm or LocalModel()
    media = describe_media(media_paths)
    if not lm.available():
        return {
            "understanding": request,
            "actionable_request": request,
            "plan_hint": "",
            "media": media,
            "via": "skipped",
            "note": f"no local model at {lm.base_url}; routing raw request",
        }
    media_block = ""
    if media:
        media_block = "\nAttached files:\n" + json.dumps(media, ensure_ascii=False)[:3000]
    try:
        out = lm.chat(
            [
                {"role": "system",
                 "content": ("You are the thinking layer of an agentic system. "
                             "Understand the request, reason briefly, then output "
                             "ONLY JSON: {\"understanding\": \"...\", "
                             "\"actionable_request\": \"...\", \"plan_hint\": \"...\"}. "
                             "actionable_request = the request restated as one clear "
                             "imperative the tool router can act on.")},
                {"role": "user",
                 "content": f"Request: {request}{media_block}"},
            ],
            max_tokens=400,
            temperature=0.2,
        )
    except RuntimeError as exc:
        return {
            "understanding": request,
            "actionable_request": request,
            "plan_hint": "",
            "media": media,
            "via": "skipped",
            "note": f"local model failed ({exc}); routing raw request",
        }
    import re
    match = re.search(r"\{.*\}", out, re.S)
    if not match:
        return {
            "understanding": request,
            "actionable_request": request,
            "plan_hint": "",
            "media": media,
            "via": "skipped",
            "note": "local model returned no JSON; routing raw request",
        }
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        data = {}
    return {
        "understanding": data.get("understanding", request),
        "actionable_request": data.get("actionable_request", request),
        "plan_hint": data.get("plan_hint", ""),
        "media": media,
        "via": "local_model",
        "model": lm.model,
    }


def compose(request, steps, lm=None):
    """THINK step 2: compose the final answer.

    Local model when reachable; otherwise a factual structured summary.
    Returns (answer_text, via).
    """
    lm = lm or LocalModel()
    facts = json.dumps(steps, ensure_ascii=False, default=str)
    if lm.available():
        try:
            answer = lm.chat(
                [
                    {"role": "system",
                     "content": ("You are VQ Agent Core. Answer from the tool "
                                 "results below. Be factual and concise. Never "
                                 "invent data the tools did not return.")},
                    {"role": "user",
                     "content": f"Request: {request}\nTool results:\n{facts}"},
                ]
            )
            return answer, f"local_model:{lm.model}"
        except RuntimeError:
            pass
    lines = [f"Request: {request}", ""]
    for s in steps:
        lines.append(f"- {s}")
    lines += ["",
              "(Composed as a factual summary -- no local LLM was reachable. "
              "Point local_model.base_url at an on-device model or Ollama/LM Studio "
              "for natural-language answers. See phone-setup.md.)"]
    return "\n".join(lines), "summary_fallback"
