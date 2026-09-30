"""Jev (TypeSafe System One) decision brain.

Two auth modes, in order:
  1. Skill CLI  (this Hatch VM): ~/workspace/skills/typesafe/bin/jev.py
     reads JSON from stdin and authenticates via the connected
     custom.typesafe credential (surrogate). No raw keys involved.
  2. Direct API (phone / other machines): set JEV_API_KEY from your own
     TypeSafe dashboard (the key YOU created). POSTs to
     https://api.typesafe.ai/v1/systemone with a Bearer header.

If neither is available, JevUnavailable is raised and the agent degrades
gracefully (router-only, escalate-on-ambiguity). Never fake a decision.
"""
import json
import os
import subprocess
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_CLI = os.path.expanduser(
    os.environ.get("VQ_JEV_CLI", "~/workspace/skills/typesafe/bin/jev.py")
)
MODEL = os.environ.get("VQ_JEV_MODEL", "jev-latest")
TIMEOUT = int(os.environ.get("VQ_JEV_TIMEOUT", "120"))


class JevUnavailable(RuntimeError):
    pass


def _via_cli(payload):
    if not (os.path.isfile(JEV_CLI)):
        raise JevUnavailable(f"jev CLI not found at {JEV_CLI}")
    proc = subprocess.run(
        ["python3", JEV_CLI],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        timeout=TIMEOUT,
    )
    if proc.returncode != 0:
        raise JevUnavailable(
            "jev CLI failed: " + proc.stderr.decode("utf-8", "replace")[:300]
        )
    return json.loads(proc.stdout.decode("utf-8"))


def _via_direct(payload):
    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        raise JevUnavailable("JEV_API_KEY is not set")
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise JevUnavailable(f"Jev API call failed: {exc}")


def query(payload):
    """POST a System One request via CLI-first, direct-key fallback."""
    payload = dict(payload)
    payload.setdefault("model", MODEL)
    errors = []
    for fn in (_via_cli, _via_direct):
        try:
            return fn(payload)
        except JevUnavailable as exc:
            errors.append(str(exc))
    raise JevUnavailable("no Jev auth path available: " + " | ".join(errors))


def decide(state, questions):
    """Ask Jev one or more decision questions about a state.

    questions: {"id": {"type": "choice|score|noul", "instructions": ..., "criteria": {...}}}
    Returns the raw Jev answers dict.
    """
    return query({"state": state, "questions": questions})


def classify_reply(reply_text):
    """Classify a prospect reply. Returns (label, confidence, raw)."""
    ans = query(
        {
            "state": reply_text,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Classify the sales intent of this prospect reply.",
                    "criteria": {
                        "interested": "Positive interest, wants more info, no demo asked yet.",
                        "demo_requested": "Explicitly asks for a demo, call, or meeting.",
                        "not_interested": "Declines, says no, asks to stop contacting.",
                        "follow_up_later": "Defers to later, asks to reconnect another time.",
                    },
                }
            },
        }
    )
    node = (ans.get("answers") or {}).get("intent", {}) or {}
    label = node.get("choice") or node.get("value") or node.get("answer")
    return {
        "label": label,
        "confidence": node.get("confidence"),
        "distribution": node.get("distribution"),
        "raw": ans,
    }


def score_lead(lead_text):
    """Score a sales lead: temperature (hot/warm/cold) + fit 1-5."""
    ans = query(
        {
            "state": lead_text,
            "questions": {
                "temperature": {
                    "type": "choice",
                    "instructions": "Classify sales temperature of this lead.",
                    "criteria": {
                        "hot": "Clear buying intent, budget/timeline signals, asked for next step.",
                        "warm": "Some interest or fit, but no concrete buying signal yet.",
                        "cold": "No interest signal, poor fit, or purely informational.",
                    },
                },
                "fit": {
                    "type": "score",
                    "instructions": "Rate how well this lead fits an AI-automation seller (VisionQuantech).",
                    "criteria": [
                        "1 - Poor fit: services solve no visible problem here.",
                        "2 - Weak fit: marginal overlap with their needs.",
                        "3 - Moderate fit: plausible use case, nothing specific.",
                        "4 - Strong fit: clear use case in their enquiry/booking flow.",
                        "5 - Excellent fit: textbook case for AI automation services.",
                    ],
                },
            },
        }
    )
    answers = ans.get("answers") or {}
    temp = answers.get("temperature", {}) or {}
    fit = answers.get("fit", {}) or {}
    return {
        "temperature": temp.get("choice") or temp.get("value"),
        "temperature_confidence": temp.get("confidence"),
        "fit": fit.get("score") or fit.get("value"),
        "fit_confidence": fit.get("confidence"),
        "raw": ans,
    }


def should_escalate(request_text, context=""):
    """Ask Jev whether this request needs a human. Returns (probability, raw)."""
    state = request_text if not context else f"Request: {request_text}\nContext: {context}"
    ans = query(
        {
            "state": state,
            "questions": {
                "escalate": {
                    "type": "noul",
                    "instructions": (
                        "Should this request be escalated to a human instead of "
                        "being handled automatically? Answer yes for risky, "
                        "ambiguous, legal/financial, or reputation-sensitive requests."
                    ),
                },
                "action": {
                    "type": "choice",
                    "instructions": "Pick what the agent should do next.",
                    "criteria": {
                        "proceed": "Safe to execute the top-ranked tool.",
                        "clarify": "Ask the user a clarifying question first.",
                        "escalate": "Hand this to a human; do not act.",
                    },
                },
            },
        }
    )
    answers = ans.get("answers") or {}
    esc = answers.get("escalate", {}) or {}
    act = answers.get("action", {}) or {}
    return {
        "escalate_probability": esc.get("value", esc.get("probability")),
        "action": act.get("choice") or act.get("value"),
        "action_confidence": act.get("confidence"),
        "raw": ans,
    }
