"""Local markdown notes. Works everywhere incl. Termux (plain files)."""
import datetime
import json
import os
import re

import needle

_NOTES_DIR = os.environ.get("VQ_NOTES_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes"
)


def _slug(title):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "note"
    return slug[:60]


@needle.tool
def save_note(title: str, text: str) -> str:
    """Save a short note to the local notes folder as a markdown file.

    Args:
        title: short title for the note.
        text: note body (markdown ok).
    """
    os.makedirs(_NOTES_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    fname = f"{stamp}-{_slug(title)}.md"
    path = os.path.join(_NOTES_DIR, fname)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n{text.strip()}\n")
    return json.dumps({"saved": path, "title": title})


@needle.tool
def list_notes() -> str:
    """List saved note files (newest first)."""
    if not os.path.isdir(_NOTES_DIR):
        return json.dumps({"notes": []})
    files = sorted(
        (f for f in os.listdir(_NOTES_DIR) if f.endswith(".md")), reverse=True
    )
    return json.dumps({"notes": files[:50], "dir": _NOTES_DIR})


@needle.tool
def read_notes(query: str = "") -> str:
    """Search saved notes for a keyword and return matches.

    Args:
        query: keyword to search for (empty = return the newest note).
    """
    if not os.path.isdir(_NOTES_DIR):
        return json.dumps({"matches": []})
    files = sorted(
        (f for f in os.listdir(_NOTES_DIR) if f.endswith(".md")), reverse=True
    )
    if not query:
        files = files[:1]
    matches = []
    for fname in files:
        path = os.path.join(_NOTES_DIR, fname)
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
        if not query or query.lower() in body.lower():
            matches.append({"file": fname, "excerpt": body[:600]})
        if len(matches) >= 5:
            break
    return json.dumps({"matches": matches}, ensure_ascii=False)
