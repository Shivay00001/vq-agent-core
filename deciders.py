"""DECIDE-layer factory.

config `decider:` selects the judgment engine:
  jev   -> Jev cloud (TypeSafe System One). Sharpest, calibrated judgments.
  local -> Best available on-device decider, tried in order:
           1. Laya (laya_decider) -- local Jev alternative: purpose-built
              System One model, RLCD-calibrated, Apache-2.0.
              Needs the `laya` package + torch + weights (PC/VM, not Termux).
           2. Prompt-based local LLM (local_decider) -- last resort:
              private / offline / free; less sharp, uncalibrated confidence.
           If neither is available the error propagates and the agent
           escalates to a human (safe default).
  auto  -> Try Jev first; fall back to local when Jev is unreachable
           or fails. If both fail, the error propagates and the agent
           escalates to a human (safe default).

All three expose the same interface:
  decide(state, questions)  with choice / score / noul question types
  classify_reply(text)      -> {label, confidence, ...}
  score_lead(text)          -> {temperature, fit, ...}
  should_escalate(req, ctx) -> {escalate_probability, action, ...}

Tools read the VQ_DECIDER env var (set by agent.py from config);
default is "jev" so the tools also work standalone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jev_brain  # noqa: E402
import local_decider  # noqa: E402
import laya_decider  # noqa: E402


class JevDecider:
    name = "jev"

    def decide(self, state, questions):
        return jev_brain.decide(state, questions)

    def classify_reply(self, reply_text):
        return jev_brain.classify_reply(reply_text)

    def score_lead(self, lead_text):
        return jev_brain.score_lead(lead_text)

    def should_escalate(self, request_text, context=""):
        return jev_brain.should_escalate(request_text, context)


class LocalDecider:
    """`local`: Laya first, prompt-based local LLM as last resort."""

    name = "local"
    _backends = ("laya", "prompt")

    def __init__(self, backend=None):
        # backend: "laya" | "prompt" | None (None = try in order)
        if backend and backend not in self._backends:
            raise ValueError(f"unknown local backend: {backend!r}")
        self.backend = backend
        self.last_backend = None

    def _run(self, method, *args, **kwargs):
        order = [self.backend] if self.backend else list(self._backends)
        errors = []
        for name in order:
            mod = laya_decider if name == "laya" else local_decider
            try:
                res = getattr(mod, method)(*args, **kwargs)
            except (laya_decider.LayaDeciderUnavailable,
                    local_decider.LocalDeciderUnavailable) as exc:
                errors.append(f"{name}: {exc}")
                continue
            self.last_backend = name
            if isinstance(res, dict):
                res = dict(res)
                res.setdefault("local_backend", name)
            return res
        raise local_decider.LocalDeciderUnavailable(
            "no local decider backend available (" + "; ".join(errors) + ")"
        )

    def decide(self, state, questions):
        return self._run("decide", state, questions)

    def classify_reply(self, reply_text):
        return self._run("classify_reply", reply_text)

    def score_lead(self, lead_text):
        return self._run("score_lead", lead_text)

    def should_escalate(self, request_text, context=""):
        return self._run("should_escalate", request_text, context)


class AutoDecider:
    name = "auto"

    def __init__(self):
        self.jev = JevDecider()
        self.local = LocalDecider()
        self.last_used = None

    def _run(self, method, *args, **kwargs):
        try:
            res = getattr(self.jev, method)(*args, **kwargs)
            self.last_used = "jev"
            return res
        except jev_brain.JevUnavailable as jev_err:
            try:
                res = getattr(self.local, method)(*args, **kwargs)
            except local_decider.LocalDeciderUnavailable as local_err:
                raise RuntimeError(
                    "no decider available: "
                    f"Jev failed ({jev_err}); local decider failed ({local_err})"
                ) from local_err
            if isinstance(res, dict):
                res = dict(res)
                res["fallback"] = f"jev_unavailable: {jev_err}"
            self.last_used = "local"
            return res

    def decide(self, state, questions):
        return self._run("decide", state, questions)

    def classify_reply(self, reply_text):
        return self._run("classify_reply", reply_text)

    def score_lead(self, lead_text):
        return self._run("score_lead", lead_text)

    def should_escalate(self, request_text, context=""):
        return self._run("should_escalate", request_text, context)


def from_name(name):
    name = (name or "jev").lower()
    if name == "jev":
        return JevDecider()
    if name == "local":
        return LocalDecider()
    if name == "auto":
        return AutoDecider()
    raise ValueError(f"unknown decider: {name!r} (use jev | local | auto)")


def from_config(cfg):
    return from_name((cfg.get("decider") or "jev"))


def from_env():
    return from_name(os.environ.get("VQ_DECIDER", "jev"))
