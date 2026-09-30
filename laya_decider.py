"""Laya-backed local DECIDE layer -- a local Jev alternative.

Laya (Convai Innovations, Apache-2.0, https://github.com/NandhaKishorM/laya)
is a non-autoregressive System One decision model: 421M params (English
checkpoint, ModernBERT-large encoder + decision head), RLCD-trained so its
probabilities are calibrated. It answers typed choice / score / noul
questions about a state in ONE encoder forward pass -- the same interface
shape as TypeSafe Jev, with no API key and no network after the weights
are downloaded. Evaluated against poorjev and OpenJev; picked as the
`local` backend on merit (purpose-built decision model, calibrated,
real local inference verified on CPU).

Same interface as jev_brain -- decide(state, questions) with
choice / score / noul question types -- plus the helper wrappers
(classify_reply / score_lead / should_escalate) with identical questions
and return shapes, so callers can't tell which decider ran, except for
the extra "via": "laya" marker.

Question dicts pass straight through to laya: {"id": {"type": ...,
"instructions": ..., "criteria": ...}} -- no translation layer needed.

ACCURACY TRADEOFF (honest, from the published crossbench):
  - Jev wins outright on the multi-primitive set: accuracy 0.906, ECE 0.045.
  - Laya (open) on the same set: accuracy 0.775, ECE 0.215.
  - poorjev (open, NLI-based): accuracy 0.781, ECE 0.071.
  Laya is the strongest *purpose-built* local alternative evaluated here
  (poorjev and OpenJev were also evaluated; Laya won on interface fit and
  verified real inference). It is weaker than Jev but far more honest than
  asking a generative LLM to self-report confidence.
  Use Jev when judgment quality matters most; `decider: auto` = Jev
  first, Laya on failure.

RUNTIME NOTES:
  - Needs the `laya` pip package + torch (NOT Termux-friendly; on the
    phone `local` falls through to the prompt-based local LLM decider).
  - Weights: convaiinnovations/laya (~1.7 GB fp32) from Hugging Face,
    cached after first download. Set VQ_LAYA_MODEL to a local dir or
    another hub id (e.g. convaiinnovations/laya-multilingual).
  - CPU: ~200-460 ms per call per the authors' measurements.
  - Env: LAYA_DEVICE (default cpu), LAYA_THREADS (cap torch threads).

Nothing here ever fakes a decision: if the laya package or the weights
can't load, every entry point raises LayaDeciderUnavailable so the
caller (deciders.py) can fall through to the next local backend.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class LayaDeciderUnavailable(RuntimeError):
    pass


_agent = None
_agent_error = None


def _load_agent():
    global _agent, _agent_error
    if _agent is not None:
        return _agent
    if _agent_error is not None:
        raise LayaDeciderUnavailable(str(_agent_error))
    try:
        import laya  # noqa: E402
    except ImportError as exc:
        _agent_error = f"laya package not installed (pip install laya): {exc}"
        raise LayaDeciderUnavailable(str(_agent_error)) from exc
    model = os.environ.get("VQ_LAYA_MODEL", "convaiinnovations/laya")
    device = os.environ.get("LAYA_DEVICE", "cpu")
    try:
        _agent = laya.load(model, device=device)
    except Exception as exc:  # weights missing / corrupt / OOM
        _agent_error = f"could not load Laya model {model!r}: {exc}"
        raise LayaDeciderUnavailable(str(_agent_error)) from exc
    return _agent


def decide(state, questions):
    """Ask Laya one or more decision questions about a state.

    questions: {"id": {"type": "choice|score|noul", "instructions": ..., "criteria": {...}}}
    Returns the raw answers dict in Jev's shape: {"answers": {...}, "usage": {...}}.
    """
    agent = _load_agent()
    if not isinstance(state, str) or not state.strip():
        raise LayaDeciderUnavailable("state must be a non-empty string")
    if not questions:
        raise LayaDeciderUnavailable("decide() needs at least one question")
    try:
        return agent.predict(state, questions)
    except LayaDeciderUnavailable:
        raise
    except Exception as exc:
        raise LayaDeciderUnavailable(f"Laya predict failed: {exc}") from exc


def _answers(res):
    return (res.get("answers") or {}) if isinstance(res, dict) else {}


def _conf(node):
    """Laya reports two confidences: `confidence` (1 - normalized entropy)
    and `answer_confidence` (top option's probability). Jev's confidence IS
    the top option's probability, so prefer answer_confidence for
    Jev-shape compatibility; fall back to confidence."""
    node = node or {}
    for key in ("answer_confidence", "confidence"):
        val = node.get(key)
        if val is not None:
            return float(val)
    return None


def _choice_node(node):
    node = node or {}
    label = node.get("choice") or node.get("value") or node.get("answer")
    dist = node.get("probabilities") or node.get("distribution")
    return label, _conf(node), dist


def classify_reply(reply_text):
    """Classify a prospect reply. Returns (label, confidence, raw)."""
    res = decide(
        reply_text,
        {
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
    )
    label, conf, dist = _choice_node(_answers(res).get("intent"))
    if label is None:
        raise LayaDeciderUnavailable(f"Laya returned no choice label: {res!r}"[:400])
    return {"label": label, "confidence": conf, "distribution": dist, "raw": res, "via": "laya"}


def score_lead(lead_text):
    """Score a sales lead: temperature (hot/warm/cold) + fit 1-5."""
    res = decide(
        lead_text,
        {
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
    )
    answers = _answers(res)
    temp_label, temp_conf, _ = _choice_node(answers.get("temperature"))
    fit_node = answers.get("fit") or {}
    # Laya score is the zero-based expected rubric level; our fit scale is 1-5.
    fit_raw = fit_node.get("score")
    fit = (float(fit_raw) + 1.0) if fit_raw is not None else fit_node.get("value")
    if temp_label is None or fit is None:
        raise LayaDeciderUnavailable(f"Laya returned incomplete lead score: {res!r}"[:400])
    return {
        "temperature": temp_label,
        "temperature_confidence": temp_conf,
        "fit": round(float(fit), 2),
        "fit_confidence": _conf(fit_node),
        "raw": res,
        "via": "laya",
    }


def should_escalate(request_text, context=""):
    """Ask Laya whether this request needs a human. Returns (probability, raw)."""
    state = request_text if not context else f"Request: {request_text}\nContext: {context}"
    res = decide(
        state,
        {
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
    )
    answers = _answers(res)
    esc = answers.get("escalate") or {}
    prob = esc.get("noul")
    if prob is None:
        prob = esc.get("prob", esc.get("value"))
    act_label, act_conf, _ = _choice_node(answers.get("action"))
    if prob is None:
        raise LayaDeciderUnavailable(f"Laya returned no noul probability: {res!r}"[:400])
    prob = float(prob)
    return {
        "escalate_probability": prob,
        "action": act_label,
        "action_confidence": act_conf,
        "noul_confidence": max(prob, 1.0 - prob),
        "raw": res,
        "via": "laya",
    }
