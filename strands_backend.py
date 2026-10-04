"""Strands-backed local DECIDE layer -- a local Jev alternative.

Strands Decider 2B (AWS Strands Labs, released 2026-10-01, Apache-2.0,
https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) is a
purpose-built System One decision model: a Qwen3.5-2B base (also Apache-2.0)
with the language-modeling head replaced by a ~1M-parameter pointer head
plus a LoRA adapter. It answers typed choice / score / noul questions about
a state in ONE forward pass -- no token-by-token decoding -- returning
calibrated probabilities. Designed for exactly this job: "should this tool
run?" / "which route fits this request?" as a cheap, reliable workflow step
around a more capable agent.

Same interface as jev_brain / laya_decider -- decide(state, questions) with
choice / score / noul question types -- plus the helper wrappers
(classify_reply / score_lead / should_escalate) with identical questions
and return shapes, so callers can't tell which decider ran, except for
the extra "via": "strands" marker.

Question dicts are translated to strands-decider's pydantic Question models;
answers are translated back to the Jev/Laya answer-node shapes:
  choice -> {"choice", "probabilities", "confidence"}
  noul   -> {"noul"}                                   (probability IS the uncertainty)
  score  -> {"score", "legend", "probabilities", "confidence"}
Our 1-5 fit rubric: strands score is the 0-based expected rubric level, so
fit = score + 1 (same convention as laya_decider).

RUNTIME NOTES:
  - Needs the `strands-decider` pip package + torch + weights (PC/VM, not Termux).
  - Weights live at ./models/strands-decider-2b (adapter repo; the Qwen3.5-2B
    base torso is pulled from the HF cache on first load). Override with
    VQ_STRANDS_MODEL. After download, decision time needs NO network.
  - Env: STRANDS_DEVICE (default cpu), STRANDS_THREADS (cap torch threads).
  - CPU: a few seconds per call on a 2-CPU box (GPU median ~115ms per the
    release notes). Batch questions about one state together -- encoding
    the state once and evaluating questions in parallel is nearly free.

Nothing here ever fakes a decision: if the package or the weights can't
load, every entry point raises StrandsDeciderUnavailable so the caller
(deciders.py) can fall through to the next local backend.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "strands-decider-2b"
)
DEFAULT_TORSO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3.5-2B-Base"
)
_TORSO_REPO_ID = "Qwen/Qwen3.5-2B-Base"


class StrandsDeciderUnavailable(RuntimeError):
    pass


def _ensure_torso_cache():
    """Make the Qwen3.5-2B torso resolvable offline via the HF hub cache.

    The checkpoint config names the torso by hub id
    ("Qwen/Qwen3.5-2B-Base"). When we have a full local copy at
    ./models/Qwen3.5-2B-Base, symlink it into the standard hub cache layout
    so transformers' from_pretrained() finds it with HF_HUB_OFFLINE=1 and
    never touches the network. Idempotent; a no-op when the local copy or
    a cached snapshot already exists.
    """
    torso_dir = os.environ.get("VQ_STRANDS_TORSO", DEFAULT_TORSO_DIR)
    if not os.path.isdir(torso_dir):
        return
    hf_home = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    repo_dir = os.path.join(
        hf_home, "hub", "models--" + _TORSO_REPO_ID.replace("/", "--")
    )
    snap_dir = os.path.join(repo_dir, "snapshots", "main")
    try:
        os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
        os.makedirs(os.path.join(repo_dir, "blobs"), exist_ok=True)
        # A previous partial hub download may have left refs/main pointing at
        # a commit hash with an incomplete snapshots/<hash>/ dir. Point the
        # "main" ref at our symlinked snapshot so the resolver uses it.
        if os.path.islink(snap_dir) and os.readlink(snap_dir) != torso_dir:
            os.unlink(snap_dir)
        if not os.path.lexists(snap_dir):
            os.symlink(torso_dir, snap_dir)
        with open(os.path.join(repo_dir, "refs", "main"), "w") as fh:
            fh.write("main")
    except OSError:
        pass  # cache setup is best-effort; load will raise clearly if offline


_engine = None
_engine_error = None


def _load_engine():
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error is not None:
        raise StrandsDeciderUnavailable(str(_engine_error))
    try:
        from strands_decider.infer import load_engine  # noqa: E402
        from strands_decider import infer as _sd_infer  # noqa: E402
    except ImportError as exc:
        _engine_error = (
            "strands-decider package not installed "
            "(use the project's .venv-strands): " + str(exc)
        )
        raise StrandsDeciderUnavailable(str(_engine_error)) from exc
    model = os.environ.get("VQ_STRANDS_MODEL", DEFAULT_MODEL_DIR)
    device = os.environ.get("STRANDS_DEVICE", "cpu")
    threads = os.environ.get("STRANDS_THREADS")
    if threads:
        try:
            import torch  # noqa: E402

            torch.set_num_threads(int(threads))
        except Exception:
            pass
    if device == "cpu" and os.environ.get("STRANDS_CPU_UPCAST") != "1":
        # The package upcasts the bf16 torso to fp32 on CPU for speed
        # (~7.6 GiB instead of ~3.8 GiB). This box can't afford that, and
        # bf16 runs natively here (avx512_bf16) with identical answers per
        # the package's own notes. Set STRANDS_CPU_UPCAST=1 on a big-RAM
        # machine to restore the faster default.
        _sd_infer.SystemOneEngine._upcast_torso_for_cpu = lambda self: None
    if not os.path.isdir(model):
        _engine_error = (
            f"Strands Decider weights not found at {model!r} "
            "(download the adapter repo + Qwen3.5-2B base; see README)"
        )
        raise StrandsDeciderUnavailable(str(_engine_error))
    _ensure_torso_cache()
    try:
        _engine = load_engine(model, device=device)
    except Exception as exc:  # weights missing / corrupt / OOM
        _engine_error = f"could not load Strands Decider model {model!r}: {exc}"
        raise StrandsDeciderUnavailable(str(_engine_error)) from exc
    return _engine


def _to_question(spec):
    """Plain question dict -> strands-decider pydantic Question."""
    from strands_decider.schema import (  # noqa: E402
        ChoiceQuestion,
        NoulQuestion,
        ScoreQuestion,
    )

    qtype = spec.get("type")
    instructions = spec.get("instructions", "")
    criteria = spec.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise StrandsDeciderUnavailable(
                f"choice needs a criteria dict with >= 2 options, got {criteria!r}"[:200]
            )
        return ChoiceQuestion(instructions=instructions, criteria=dict(criteria))
    if qtype == "score":
        if not isinstance(criteria, (list, tuple)) or not (2 <= len(criteria) <= 10):
            raise StrandsDeciderUnavailable(
                f"score needs 2-10 rubric lines, got {criteria!r}"[:200]
            )
        return ScoreQuestion(instructions=instructions, criteria=list(criteria))
    if qtype == "noul":
        return NoulQuestion(instructions=instructions)
    raise StrandsDeciderUnavailable(f"unknown question type: {qtype!r}")


def _to_node(answer):
    """strands-decider Answer -> Jev/Laya-shaped answer node."""
    kind = getattr(answer, "type", None)
    if kind == "choice":
        return {
            "choice": answer.choice,
            "probabilities": dict(answer.probabilities or {}),
            "confidence": float(answer.confidence),
        }
    if kind == "noul":
        return {"noul": float(answer.noul)}
    if kind == "score":
        return {
            "score": float(answer.score),
            "legend": {str(k): v for k, v in (answer.legend or {}).items()},
            "probabilities": {str(k): v for k, v in (answer.probabilities or {}).items()},
            "confidence": float(answer.confidence),
        }
    raise StrandsDeciderUnavailable(f"unexpected answer kind: {kind!r}")


def decide(state, questions):
    """decide(state, questions) -- same interface as jev_brain.decide.

    questions: {"id": {"type": "choice|score|noul", "instructions": str,
                       "criteria": {key: desc} | [rubric lines]}}
    Returns {"answers": {"id": {...}}, "via": "strands"}.
    """
    if not isinstance(state, str) or not state.strip():
        raise StrandsDeciderUnavailable("state must be a non-empty string")
    if not questions:
        raise StrandsDeciderUnavailable("decide() needs at least one question")
    engine = _load_engine()
    qmap = {qid: _to_question(spec) for qid, spec in questions.items()}
    try:
        res = engine.ask(state, qmap)
    except StrandsDeciderUnavailable:
        raise
    except Exception as exc:
        raise StrandsDeciderUnavailable(f"Strands Decider ask failed: {exc}") from exc
    return {"answers": {qid: _to_node(a) for qid, a in res.answers.items()},
            "via": "strands"}


def _answers(res):
    return (res.get("answers") or {}) if isinstance(res, dict) else {}


# --- Canonical question specs (same semantics as jev_brain / laya_decider) ---

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

ACTION_CRITERIA = {
    "proceed": "Safe to execute the top-ranked tool.",
    "clarify": "Ask the user a clarifying question first.",
    "escalate": "Hand this to a human; do not act.",
}


def classify_reply(reply_text):
    """Classify a prospect reply. Returns (label, confidence, distribution)."""
    res = decide(
        reply_text,
        {"intent": {
            "type": "choice",
            "instructions": "Classify the sales intent of this prospect reply.",
            "criteria": REPLY_CRITERIA,
        }},
    )
    node = _answers(res).get("intent") or {}
    label = node.get("choice")
    if label is None:
        raise StrandsDeciderUnavailable(f"Strands returned no choice label: {res!r}"[:400])
    return {"label": label, "confidence": float(node.get("confidence", 0.0)),
            "distribution": node.get("probabilities"), "via": "strands"}


def score_lead(lead_text):
    """Score a sales lead: temperature (hot/warm/cold) + fit 1-5."""
    res = decide(
        lead_text,
        {"temperature": {
             "type": "choice",
             "instructions": "Classify sales temperature of this lead.",
             "criteria": TEMPERATURE_CRITERIA},
         "fit": {
             "type": "score",
             "instructions": "Rate how well this lead fits an AI-automation seller (VisionQuantech).",
             "criteria": FIT_RUBRIC}},
    )
    answers = _answers(res)
    temp = answers.get("temperature") or {}
    fit_node = answers.get("fit") or {}
    # Strands score is the zero-based expected rubric level; our scale is 1-5.
    fit_raw = fit_node.get("score")
    fit = (float(fit_raw) + 1.0) if fit_raw is not None else None
    if temp.get("choice") is None or fit is None:
        raise StrandsDeciderUnavailable(f"Strands returned incomplete lead score: {res!r}"[:400])
    return {
        "temperature": temp["choice"],
        "temperature_confidence": float(temp.get("confidence", 0.0)),
        "fit": round(float(fit), 2),
        "fit_confidence": float(fit_node.get("confidence", 0.0)),
        "via": "strands",
    }


def should_escalate(request_text, context=""):
    """Decide proceed / clarify / escalate for a proposed agent action.

    Returns {"escalate_probability", "action", "action_confidence", "via"}.
    `action` is one of proceed | clarify | escalate -- the concrete
    proceed/escalate decision the DECIDE gate needs, with confidence.
    """
    state = request_text if not context else f"Request: {request_text}\nContext: {context}"
    res = decide(
        state,
        {"escalate": {
             "type": "noul",
             "instructions": ("Should this request be escalated to a human instead of "
                              "being handled automatically? Answer yes for risky, "
                              "ambiguous, legal/financial, or reputation-sensitive requests.")},
         "action": {
             "type": "choice",
             "instructions": "Pick what the agent should do next.",
             "criteria": ACTION_CRITERIA}},
    )
    answers = _answers(res)
    esc = answers.get("escalate") or {}
    act = answers.get("action") or {}
    prob = esc.get("noul")
    if prob is None:
        raise StrandsDeciderUnavailable(f"Strands returned no noul probability: {res!r}"[:400])
    prob = float(prob)
    if act.get("choice") is None:
        raise StrandsDeciderUnavailable(f"Strands returned no action choice: {res!r}"[:400])
    return {
        "escalate_probability": prob,
        "action": act["choice"],
        "action_confidence": float(act.get("confidence", 0.0)),
        "action_distribution": act.get("probabilities"),
        "noul_confidence": max(prob, 1.0 - prob),
        "via": "strands",
    }
