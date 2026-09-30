"""DECIDE-layer tools: judgments via the configured decider.

The decider is chosen by the VQ_DECIDER env var (jev | local | auto),
set by agent.py from config.yaml. Default "jev" keeps the tools
usable standalone. In "auto" mode Jev is tried first and the local
LLM decider is the fallback; if neither is reachable the tool
returns {"error": "decider_unavailable"} -- never a faked judgment.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import needle  # noqa: E402

import deciders  # noqa: E402
import jev_brain  # noqa: E402
import local_decider  # noqa: E402


def _call(method, *args):
    try:
        res = getattr(deciders.from_env(), method)(*args)
    except jev_brain.JevUnavailable as exc:
        return {"error": "decider_unavailable", "detail": f"jev: {exc}"}
    except local_decider.LocalDeciderUnavailable as exc:
        return {"error": "decider_unavailable", "detail": f"local: {exc}"}
    except RuntimeError as exc:  # auto mode: both paths failed
        return {"error": "decider_unavailable", "detail": str(exc)}
    if isinstance(res, dict):
        res = dict(res)
        res.pop("raw", None)
    return res


@needle.tool
def score_lead(lead_text: str) -> str:
    """Score a sales lead: temperature hot/warm/cold + fit score 1-5.

    Use when the user asks to score, rate, qualify, or judge a lead/prospect.
    Judgment comes from the configured DECIDE layer (Jev cloud or local LLM).

    Args:
        lead_text: free-text description of the lead.
    """
    return json.dumps(_call("score_lead", lead_text), ensure_ascii=False)


@needle.tool
def classify_reply(reply_text: str) -> str:
    """Classify a prospect reply: interested / demo_requested /
    not_interested / follow_up_later.

    Judgment comes from the configured DECIDE layer (Jev cloud or local LLM).

    Args:
        reply_text: the prospect's reply text.
    """
    return json.dumps(_call("classify_reply", reply_text), ensure_ascii=False)


@needle.tool
def should_escalate(request_text: str, context: str = "") -> str:
    """Ask the DECIDE layer whether a request needs a human.

    Args:
        request_text: the user request to judge.
        context: optional extra context.
    """
    return json.dumps(_call("should_escalate", request_text, context),
                      ensure_ascii=False)
