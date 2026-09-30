"""Web tools: real HTTP calls, no API keys needed."""
import html
import json
import re
import urllib.parse
import urllib.request

import needle

_DDG = "https://api.duckduckgo.com/"


@needle.tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for a query. Returns instant-answer text plus related topics.

    Args:
        query: the search query.
        max_results: max related topics to include (1-10).
    """
    max_results = max(1, min(10, int(max_results)))
    params = urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    )
    req = urllib.request.Request(
        _DDG + "?" + params, headers={"User-Agent": "VQAgentCore/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # network/DNS failure -> structured error, not fake data
        return json.dumps({"error": f"web_search failed: {exc}"})

    out = {
        "query": query,
        "abstract": data.get("AbstractText") or "",
        "abstract_url": data.get("AbstractURL") or "",
        "related": [],
    }
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            out["related"].append(
                {"text": topic["Text"][:300], "url": topic.get("FirstURL", "")}
            )
        if len(out["related"]) >= max_results:
            break
    if not out["abstract"] and not out["related"]:
        out["note"] = "No instant answer returned for this query."
    return json.dumps(out, ensure_ascii=False)


_TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S)


@needle.tool
def fetch_url(url: str, max_chars: int = 5000) -> str:
    """Fetch a web page and return its visible text (HTML stripped).

    Args:
        url: http(s) URL to fetch.
        max_chars: max characters of text to return.
    """
    if not url.lower().startswith(("http://", "https://")):
        return json.dumps({"error": "fetch_url: only http(s) URLs are allowed"})
    host = urllib.parse.urlparse(url).hostname or ""
    # Basic SSRF guard: never fetch loopback / private targets.
    if host in ("localhost", "127.0.0.1", "::1") or host.startswith(
        ("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2",
         "172.30.", "172.31.")
    ):
        return json.dumps({"error": "fetch_url: private/local hosts are blocked"})
    req = urllib.request.Request(url, headers={"User-Agent": "VQAgentCore/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read(2_000_000).decode("utf-8", "replace")
    except Exception as exc:
        return json.dumps({"error": f"fetch_url failed: {exc}"})
    text = _TAG_RE.sub(" ", raw)
    text = re.sub(r"\s+", " ", html.unescape(text)).strip()
    return json.dumps(
        {"url": url, "chars": len(text), "text": text[: max(500, int(max_chars))]},
        ensure_ascii=False,
    )
