"""Drafting tool: local model when reachable, else an honest template.

Jev is a DECISION model (choice/score/noul) — it cannot write prose.
So when no local model is available we return a clearly-labelled template
draft, never fake LLM output.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import needle

from local_model import LocalModel


def _template_draft(brief, kind):
    kind = (kind or "email").lower()
    if kind == "email":
        text = (
            f"Subject: Quick idea for you\n\n"
            f"Hi,\n\n"
            f"{brief.strip()}\n\n"
            f"Happy to share details or set up a quick call if useful.\n\n"
            f"Shivam | VisionQuantech"
        )
    elif kind == "whatsapp":
        text = f"Hi! {brief.strip()} — Shivam | VisionQuantech"
    else:
        text = f"# {kind.title()} draft\n\n{brief.strip()}\n\n— Shivam | VisionQuantech"
    return text


@needle.tool
def draft_text(brief: str, kind: str = "email") -> str:
    """Draft a short text (email / whatsapp / note) from a brief.

    Uses the local model when reachable; otherwise returns a labelled
    template draft (NOT presented as LLM prose).

    Args:
        brief: what the draft should say.
        kind: email | whatsapp | note.
    """
    lm = LocalModel()
    if lm.available():
        try:
            text = lm.chat(
                [
                    {
                        "role": "system",
                        "content": "You write short business outreach texts. Plain text, no hype.",
                    },
                    {
                        "role": "user",
                        "content": f"Write a {kind} from this brief:\n{brief}",
                    },
                ]
            )
        except RuntimeError as exc:
            return json.dumps({"error": f"local model failed: {exc}"})
        return json.dumps(
            {"source": "local_model", "model": lm.model, "text": text},
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "source": "template",
            "note": (
                "No local model reachable at "
                + lm.base_url
                + " — template draft below. This is NOT LLM-generated prose."
            ),
            "text": _template_draft(brief, kind),
        },
        ensure_ascii=False,
    )
