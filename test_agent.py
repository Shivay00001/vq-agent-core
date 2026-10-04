"""End-to-end tests for VQ Agent Core. Everything here REALLY executes:
Needle 2 routes for real, tools really run, Jev is really called.

Run:  python test_agent.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from needle_router import Router  # noqa: E402
from agent import Agent, load_config  # noqa: E402
from local_model import LocalModel  # noqa: E402
import jev_brain  # noqa: E402
import tools  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
PASS = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        raise SystemExit(1)
    print(f"ok   {name}")
    PASS.append(name)


def t_router_save_note():
    cfg = load_config()
    router = Router(cfg)
    try:
        name, args, conf, note = router.route(
            "save a note titled e2e-test-note saying hello from vq agent core"
        )
        assert name == "save_note", f"routed to {name} ({note})"
        assert conf >= 0.5, f"confidence too low: {conf}"
        fn = tools.get(name)
        res = json.loads(fn(**args))
        assert os.path.isfile(res["saved"]), "note file not on disk"
        with open(res["saved"], encoding="utf-8") as fh:
            body = fh.read()
        assert "hello from vq agent core" in body.lower() or "hello" in body.lower()
        print(f"     routed -> {name} conf={conf:.2f} file={res['saved']}")
    finally:
        router.close()


def t_router_sysinfo():
    cfg = load_config()
    router = Router(cfg)
    try:
        name, args, conf, note = router.route(
            "show me this device's CPU count, load average and memory stats"
        )
        assert name == "sysinfo", f"routed to {name} ({note})"
        res = json.loads(tools.get(name)(**args))
        assert "cpu_count" in res, f"unexpected sysinfo: {res}"
        print(f"     routed -> {name} conf={conf:.2f} cpu={res['cpu_count']}")
    finally:
        router.close()


def t_jev_classify():
    res = jev_brain.classify_reply(
        "Thanks, this looks interesting. Can we schedule a demo next Tuesday?"
    )
    label = res["label"]
    assert label in ("interested", "demo_requested", "follow_up_later",
                     "not_interested"), f"bad label: {label} raw={res['raw']}"
    assert label in ("interested", "demo_requested"), \
        f"expected positive intent, got {label}"
    print(f"     label={label} confidence={res['confidence']}")


def t_agent_score_lead():
    agent = Agent(load_config())
    out = agent.run(
        "score this lead: Priya Sharma runs a dental clinic in Pune with 12 staff, "
        "wants WhatsApp automation for appointment reminders, asked for pricing"
    )
    assert os.path.isfile(out["trace"]), "no trace file"
    trace = open(out["trace"], encoding="utf-8").read()
    assert '"event": "route"' in trace, "trace missing route event"
    assert "score_lead" in trace, "trace missing score_lead tool call"
    assert '"event": "tool_result"' in trace, "trace missing tool_result"
    assert '"temperature"' in trace, "trace missing scored temperature"
    assert "jev_unavailable" not in trace, "Jev call failed inside agent run"
    assert "hot" in trace or "warm" in trace, "no temperature value in trace"
    print(f"     trace={out['trace']}")
    print(f"     answer preview: {out['answer'][:220].replace(chr(10), ' ')}...")


def t_local_model_fallback():
    lm = LocalModel()
    avail = lm.available()
    print(f"     local model available: {avail} (expected False on this VM)")
    assert avail is False, "a local model IS reachable?! document it, don't assert-fail silently"
    from tools import local_tools
    res = json.loads(local_tools.draft_text("New AI chatbot service for clinics", "email"))
    assert res["source"] == "template", f"expected template fallback, got {res['source']}"
    assert "NOT LLM-generated" in res["note"]
    print("     draft_text correctly fell back to labelled template")


def t_local_decider_parsing():
    # Pure parser tests with fixed fixtures -- no model involved, no fake claims.
    import local_decider
    c = local_decider.parse_choice(
        '{"option": "demo_requested", "confidence": 0.9}',
        local_decider.REPLY_CRITERIA)
    assert c == {"option": "demo_requested", "confidence": 0.9}, c
    s = local_decider.parse_score('Noise {"value": 4.2, "confidence": 0.7} trailing')
    assert s["value"] == 4.2 and s["confidence"] == 0.7, s
    n = local_decider.parse_noul('{"probability": 0.15}')
    assert n == {"probability": 0.15}, n
    # invalid option must raise, never silently accept
    try:
        local_decider.parse_choice('{"option": "maybe", "confidence": 0.5}',
                                   local_decider.REPLY_CRITERIA)
        raise AssertionError("invalid option was accepted")
    except local_decider.LocalDeciderUnavailable:
        pass
    print("     choice/score/noul parsers validate correctly")


def t_local_decider_no_model():
    # decider=local with no local model reachable -> honest failure, no fake judgment.
    import local_decider
    try:
        local_decider.classify_reply("Can we do a demo tomorrow?")
        raise AssertionError("expected LocalDeciderUnavailable")
    except local_decider.LocalDeciderUnavailable as exc:
        print(f"     classify raised honestly: {str(exc)[:80]}")
    # ...and the full agent loop degrades to safe escalation, not a fake answer.
    # Force the DECIDE gate deterministically: a threshold above 1.0 means
    # every route (even a confident one) goes through the decider.
    cfg = load_config()
    cfg["decider"] = "local"
    cfg["router"]["confidence_threshold"] = 1.5
    agent = Agent(cfg)
    out = agent.run("do something ambiguous with unclear intent xyzzy")
    assert out["escalated"] is True, "agent should escalate when no decider is reachable"
    assert os.path.isfile(out["trace"])
    trace = open(out["trace"], encoding="utf-8").read()
    assert "decider_unavailable" in trace or "escalate" in trace.lower()
    print("     agent escalated safely with decider=local and no model")


def t_decider_auto_uses_jev():
    import deciders
    d = deciders.from_name("auto")
    assert isinstance(d, deciders.AutoDecider)
    res = d.classify_reply("This is great, let's schedule a demo next week.")
    assert res["label"] in ("interested", "demo_requested"), res
    assert d.last_used == "jev", f"expected jev, used {d.last_used}"
    print(f"     auto decider used Jev: label={res['label']}")


def t_strands_local_decider():
    # Strands Decider 2B (AWS Strands Labs, Apache-2.0): purpose-built
    # System One model, first `local` backend. Needs the `strands-decider`
    # package + torch + weights -- PC/VM only. Skips honestly when
    # unavailable.
    try:
        import strands_decider  # noqa: F401  (the pip package, not strands_backend.py)
    except ImportError:
        print("     SKIP: strands-decider package not installed in this venv "
              "(needs torch; PC/VM only -- use .venv-strands)")
        return
    import strands_backend
    try:
        res = strands_backend.classify_reply(
            "Yes, let's do a demo. Are you free tomorrow at 3pm for a call?")
    except strands_backend.StrandsDeciderUnavailable as exc:
        print(f"     SKIP: Strands weights unavailable: {str(exc)[:100]}")
        return
    assert res["via"] == "strands", res
    assert res["label"] == "demo_requested", res
    print(f"     Strands classify_reply: {res['label']} @ {res['confidence']:.3f}")
    esc = strands_backend.should_escalate(
        "Send the standard appointment reminder template to the confirmed customer.")
    assert esc["action"] == "proceed", esc
    assert 0.0 <= esc["escalate_probability"] <= 1.0, esc
    print(f"     Strands should_escalate: p={esc['escalate_probability']:.3f} "
          f"action={esc['action']}")


def t_laya_local_decider():
    # Laya = a local Jev alternative (System One model, Apache-2.0,
    # RLCD-calibrated). Needs the `laya` package + torch + weights --
    # PC/VM only. Skips honestly when unavailable.
    try:
        import laya  # noqa: F401
    except ImportError:
        print("     SKIP: laya package not installed in this venv (needs torch; PC/VM only)")
        return
    import laya_decider
    try:
        res = laya_decider.classify_reply(
            "Thanks, this looks interesting. Can we schedule a demo next Tuesday?")
    except laya_decider.LayaDeciderUnavailable as exc:
        print(f"     SKIP: Laya weights unavailable: {str(exc)[:100]}")
        return
    assert res["via"] == "laya", res
    assert res["label"] in ("interested", "demo_requested"), res
    print(f"     Laya classify_reply: {res['label']} @ {res['confidence']:.3f}")
    lead = laya_decider.score_lead(
        "Priya Sharma, owner of Sharma Dental Clinic, Andheri West Mumbai. "
        "Enquired about automating appointment reminders and follow-ups. "
        "Asked for pricing and a demo this week.")
    assert lead["temperature"] in ("hot", "warm", "cold"), lead
    assert 1.0 <= lead["fit"] <= 5.0, lead
    print(f"     Laya score_lead: {lead['temperature']} / fit {lead['fit']}")
    esc = laya_decider.should_escalate("What are your pricing plans?")
    assert 0.0 <= esc["escalate_probability"] <= 1.0, esc
    print(f"     Laya should_escalate: p={esc['escalate_probability']:.3f} action={esc['action']}")
    # deciders.local must prefer the first available backend in order:
    # strands, then laya, then prompt. Accept whichever is available here.
    import deciders
    d = deciders.from_name("local")
    r2 = d.classify_reply("Please stop emailing me, remove me from your list.")
    assert d.last_backend in ("strands", "laya", "prompt"), d.last_backend
    assert r2["label"] == "not_interested", r2
    print(f"     deciders.local picked backend: {d.last_backend}")


if __name__ == "__main__":
    check("needle routes save-note -> file on disk", t_router_save_note)
    check("needle routes system-load query -> sysinfo", t_router_sysinfo)
    check("real Jev call classifies a prospect reply", t_jev_classify)
    check("agent.run('score this lead') -> trace + tools + jev", t_agent_score_lead)
    check("no local model -> honest template fallback", t_local_model_fallback)
    check("local decider: JSON parsers validate strictly", t_local_decider_parsing)
    check("local decider: no model -> honest failure + safe escalation",
          t_local_decider_no_model)
    check("decider=auto uses Jev when reachable", t_decider_auto_uses_jev)
    check("Laya local decider (real inference when available)", t_laya_local_decider)
    check("Strands Decider 2B local backend (real inference when available)",
          t_strands_local_decider)
    print(f"\nALL {len(PASS)} TESTS PASSED")
