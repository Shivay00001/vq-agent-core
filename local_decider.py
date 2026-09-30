"""Local DECIDE layer: the on-phone Jev alternative.

Same interface as jev_brain -- decide(state, questions) with
choice / score / noul question types -- but backed by the local LLM
(THINK-layer model) using strict JSON-only prompts.

Return shapes mirror jev_brain so callers can't tell which decider ran,
except for the extra "via": "local" marker.

ACCURACY TRADEOFF (honest):
  Jev is a purpose-built, calibrated decision model -- its confidence
  numbers mean something. A general local LLM asked to self-report
  confidence is NOT calibrated, and small phone models are worse at
  following the JSON-only instruction. Use the local decider when you
  need private / offline / free decisions; use Jev when judgment
  quality matters most. `decider: auto` = Jev first, local on failure.

Nothing here ever fakes a decision: unparseable or out-of-range model
output raises LocalDeciderUnavailable after one retry.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_model import LocalModel  # noqa: E402


class LocalDeciderUnavailable(RuntimeError):
    pass


SYSTEM = (
    "You are a decision function, not a chatbot. "
    "Reply with ONLY a JSON object. No prose, no markdown, no code fences."
)


def _extract_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise LocalDeciderUnavailable(f"model returned no JSON: {text!r}"[:300])
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LocalDeciderUnavailable(f"model returned invalid JSON: {exc}") from exc


def _clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        raise LocalDeciderUnavailable(f"confidence not a number: {value!r}")


def _ask_raw(lm, user_prompt):
    """One strict JSON-only call; returns the raw model text.

    Retries once when the output contains no valid JSON object.
    Raises LocalDeciderUnavailable when the model itself is unreachable.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    last_text, last_err = "", None
    for _ in range(2):
        try:
            text = lm.chat(messages, max_tokens=256, temperature=0.0)
        except RuntimeError as exc:
            raise LocalDeciderUnavailable(f"local model call failed: {exc}") from exc
        try:
            _extract_json(text)  # validate only
            return text
        except LocalDeciderUnavailable as exc:
            last_text, last_err = text, exc
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": "That was not valid JSON. Reply with ONLY the JSON object, nothing else.",
            })
    msg = (f"model would not produce valid JSON after retry: {last_err} "
           f"(last output: {last_text!r})")
    raise LocalDeciderUnavailable(msg[:400])


def parse_choice(text, criteria):
    """Pure function: validate a choice answer. (Unit-testable, no model.)"""
    data = _extract_json(text)
    option = data.get("option")
    if option not in criteria:
        raise LocalDeciderUnavailable(
            f"option {option!r} not in {sorted(criteria)}"
        )
    return {"option": option, "confidence": _clamp01(data.get("confidence"))}


def parse_score(text, low=1.0, high=5.0):
    """Pure function: validate a score answer. (Unit-testable, no model.)"""
    data = _extract_json(text)
    try:
        value = float(data.get("value"))
    except (TypeError, ValueError):
        raise LocalDeciderUnavailable(f"score value not a number: {data!r}")
    value = max(low, min(high, value))
    return {"value": value, "confidence": _clamp01(data.get("confidence"))}


def parse_noul(text):
    """Pure function: validate a yes/no probability answer. (Unit-testable.)"""
    data = _extract_json(text)
    try:
        prob = float(data.get("probability"))
    except (TypeError, ValueError):
        raise LocalDeciderUnavailable(f"probability not a number: {data!r}")
    return {"probability": _clamp01(prob)}


def _choice_prompt(state, instructions, criteria):
    bullets = "\n".join(f"- {key}: {desc}" for key, desc in criteria.items())
    return (
        f"Classify the following.\n\nState:\n{state}\n\n"
        f"Instructions: {instructions}\n\n"
        f"Options (reply with exactly one KEY):\n{bullets}\n\n"
        'Reply with exactly this JSON and nothing else:\n'
        '{"option": "<key>", "confidence": <0.0 to 1.0>}'
    )


def _score_prompt(state, instructions, rubric):
    bullets = "\n".join(f"- {line}" for line in rubric)
    return (
        f"Rate the following on the rubric.\n\nState:\n{state}\n\n"
        f"Instructions: {instructions}\n\nRubric:\n{bullets}\n\n"
        'Reply with exactly this JSON and nothing else:\n'
        '{"value": <number on the rubric scale>, "confidence": <0.0 to 1.0>}'
    )


def _noul_prompt(state, instructions):
    return (
        f"Answer yes/no as a probability.\n\nState:\n{state}\n\n"
        f"Instructions: {instructions}\n\n"
        'Reply with exactly this JSON and nothing else:\n'
        '{"probability": <0.0 = definitely no, 1.0 = definitely yes>}'
    )


def _require_model(lm):
    lm = lm or LocalModel()
    if not lm.available():
        raise LocalDeciderUnavailable(
            f"no local model reachable at {lm.base_url} "
            "(set local_model.base_url or run decider: jev)"
        )
    return lm


# --- Canonical question specs (same semantics as jev_brain) ---

REPLY_CRITERIA = {
    "interested": "Positive interest, wants more info, no demo asked yet.",
    "demo_requested": "Explicitly asks for a demo, call, or meeting.",
    "not_interested": "Declines, says no, asks to stop contacting.",
    "follow_up_later": "Defers to later, asks to reconnect another time.",
}

TEMPERATURE_CRITERIA = {
    "hot": "Clear buying intent, budget/timeline signals, asked for next step.",
    "warm": "Some interest or fit, but no concrete buying signal yet.",
    "cold": "No interest signal, poor fit, or purely informational.",
}

FIT_RUBRIC = [
    "1 - Poor fit: services solve no visible problem here.",
    "2 - Weak fit: marginal overlap with their needs.",
    "3 - Moderate fit: plausible use case, nothing specific.",
    "4 - Strong fit: clear use case in their enquiry/booking flow.",
    "5 - Excellent fit: textbook case for AI automation services.",
]


def decide(state, questions, lm=None):
    """decide(state, questions) -- same interface as jev_brain.decide.

    questions: {"id": {"type": "choice|score|noul", "instructions": str,
                       "criteria": {key: desc} | [rubric lines]}}
    Returns {"answers": {"id": {...}}, "via": "local"}.
    """
    lm = _require_model(lm)
    answers = {}
    for qid, spec in questions.items():
        qtype = spec.get("type")
        instructions = spec.get("instructions", "")
        criteria = spec.get("criteria", {})
        if qtype == "choice":
            raw = _ask_raw(lm, _choice_prompt(state, instructions, criteria))
            parsed = parse_choice(raw, criteria)
            answers[qid] = {"choice": parsed["option"],
                            "confidence": parsed["confidence"]}
        elif qtype == "score":
            raw = _ask_raw(lm, _score_prompt(state, instructions, criteria))
            parsed = parse_score(raw)
            answers[qid] = {"value": parsed["value"],
                            "confidence": parsed["confidence"]}
        elif qtype == "noul":
            raw = _ask_raw(lm, _noul_prompt(state, instructions))
            parsed = parse_noul(raw)
            answers[qid] = {"probability": parsed["probability"]}
        else:
            raise LocalDeciderUnavailable(f"unknown question type: {qtype!r}")
    return {"answers": answers, "via": "local"}


def classify_reply(reply_text, lm=None):
    ans = decide(
        reply_text,
        {"intent": {
            "type": "choice",
            "instructions": "Classify the sales intent of this prospect reply.",
            "criteria": REPLY_CRITERIA,
        }},
        lm=lm,
    )
    node = ans["answers"]["intent"]
    return {"label": node["choice"], "confidence": node["confidence"],
            "distribution": None, "via": "local"}


def score_lead(lead_text, lm=None):
    ans = decide(
        lead_text,
        {"temperature": {
             "type": "choice",
             "instructions": "Classify sales temperature of this lead.",
             "criteria": TEMPERATURE_CRITERIA},
         "fit": {
             "type": "score",
             "instructions": "Rate how well this lead fits an AI-automation seller (VisionQuantech).",
             "criteria": FIT_RUBRIC}},
        lm=lm,
    )
    a = ans["answers"]
    return {"temperature": a["temperature"]["choice"],
            "temperature_confidence": a["temperature"]["confidence"],
            "fit": a["fit"]["value"],
            "fit_confidence": a["fit"]["confidence"],
            "via": "local"}


def should_escalate(request_text, context="", lm=None):
    state = request_text if not context else f"Request: {request_text}\nContext: {context}"
    ans = decide(
        state,
        {"escalate": {
             "type": "noul",
             "instructions": ("Should this request be escalated to a human instead of "
                              "being handled automatically? Answer yes for risky, "
                              "ambiguous, legal/financial, or reputation-sensitive requests.")},
         "action": {
             "type": "choice",
             "instructions": "Pick what the agent should do next.",
             "criteria": {
                 "proceed": "Safe to execute the top-ranked tool.",
                 "clarify": "Ask the user a clarifying question first.",
                 "escalate": "Hand this to a human; do not act."}}},
        lm=lm,
    )
    a = ans["answers"]
    return {"escalate_probability": a["escalate"]["probability"],
            "action": a["action"]["choice"],
            "action_confidence": a["action"]["confidence"],
            "via": "local"}
