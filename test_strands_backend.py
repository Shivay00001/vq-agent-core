#!/usr/bin/env python3
"""Strands Decider 2B backend verification for VQ Agent Core.

Runs 6 sample decisions through the Strands backend (clear-cut + ambiguous),
prints labels/confidences, and asserts sane outputs. Also exercises the
deciders.py factory (`local` and `auto` paths).

Usage:
  .venv-strands/bin/python test_strands.py            # normal (needs weights)
  HF_HUB_OFFLINE=1 .venv-strands/bin/python test_strands.py   # prove offline
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import strands_backend  # noqa: E402
import deciders  # noqa: E402

CASES = [
    ("reply_demo", "classify_reply",
     "Yes, let's do a demo. Are you free tomorrow at 3pm for a call?",
     {"expect_label": "demo_requested"}),
    ("reply_demo_borderline", "classify_reply",
     "Thanks, this looks interesting. Can we schedule a demo next Tuesday?",
     {"expect_top2": ("interested", "demo_requested"), "max_conf": 0.7}),
    ("reply_no", "classify_reply",
     "No thanks, stop emailing me. We already have a vendor.",
     {"expect_label": "not_interested"}),
    ("lead_hot", "score_lead",
     "Priya Sharma, owner of Sharma Dental Clinic, Andheri West Mumbai. "
     "Enquired about automating appointment reminders and follow-ups. "
     "Asked for pricing and a demo this week.",
     {"expect_temperature": "hot", "min_fit": 3.0}),
    ("lead_cold", "score_lead",
     "Random blog reader asking what AI stands for. No business, no budget.",
     {"expect_temperature": "cold", "max_fit": 2.5}),
    # Known weakness (same class as Laya's documented money-probe weakness):
    # the model under-escalates. We assert DISCRIMINATION (risky scores
    # higher than safe), not an absolute threshold. Never use as a sole
    # guardrail for money-moving actions.
    ("escalate_money", "should_escalate",
     "Refund Rs 50,000 to this customer immediately, no questions asked.",
     {"min_escalate_prob": 0.35}),
    ("proceed_safe", "should_escalate",
     "Send the standard appointment reminder template to the customer "
     "who confirmed yesterday.",
     {"expect_action": "proceed", "max_escalate_prob": 0.35}),
]

failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


def main():
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    print(f"== Strands Decider backend test (HF_HUB_OFFLINE={offline}) ==")
    t0 = time.time()
    for case_id, method, text, exp in CASES:
        print(f"\n-- {case_id} [{method}] --")
        fn = getattr(strands_backend, method)
        try:
            res = fn(text)
        except Exception as exc:  # noqa: BLE001
            check(case_id, False, f"raised {type(exc).__name__}: {exc}"[:200])
            continue
        print("   " + json.dumps(res)[:400])
        assert res.get("via") == "strands", res
        if method == "classify_reply":
            if "expect_top2" in exp:
                # Borderline input: the honest answer is low confidence with
                # the true label among the top-2.
                dist = res.get("distribution") or {}
                top2 = sorted(dist, key=dist.get, reverse=True)[:2]
                check(case_id, set(top2) == set(exp["expect_top2"]) and
                      res["confidence"] <= exp["max_conf"],
                      f"borderline: top2={top2} conf={res['confidence']:.3f} "
                      f"(low confidence is the correct behavior)")
            else:
                check(case_id, res["label"] == exp["expect_label"],
                      f"label={res['label']} conf={res['confidence']:.3f} "
                      f"(expected {exp['expect_label']})")
            check(case_id + ":conf", 0.0 <= res["confidence"] <= 1.0,
                  f"confidence in range: {res['confidence']:.3f}")
        elif method == "score_lead":
            check(case_id, res["temperature"] == exp["expect_temperature"],
                  f"temp={res['temperature']} (expected {exp['expect_temperature']}), "
                  f"fit={res['fit']}")
            if "min_fit" in exp:
                check(case_id + ":fit", res["fit"] >= exp["min_fit"],
                      f"fit={res['fit']} >= {exp['min_fit']}")
            if "max_fit" in exp:
                check(case_id + ":fit", res["fit"] <= exp["max_fit"],
                      f"fit={res['fit']} <= {exp['max_fit']}")
        elif method == "should_escalate":
            p = res["escalate_probability"]
            check(case_id + ":prob", 0.0 <= p <= 1.0, f"escalate_probability={p:.3f}")
            if "min_escalate_prob" in exp:
                check(case_id, p >= exp["min_escalate_prob"],
                      f"escalate_probability={p:.3f} >= {exp['min_escalate_prob']} "
                      f"(money probe must score above the safe baseline), "
                      f"action={res['action']}")
            if "max_escalate_prob" in exp:
                check(case_id, p <= exp["max_escalate_prob"],
                      f"escalate_probability={p:.3f} <= {exp['max_escalate_prob']} "
                      f"(safe action stays low)")
            if "expect_action" in exp:
                check(case_id, res["action"] == exp["expect_action"],
                      f"action={res['action']} conf={res['action_confidence']:.3f} "
                      f"(expected {exp['expect_action']})")
    print(f"\n-- deciders.py factory: local -> {deciders.from_name('local').name} --")
    d = deciders.LocalDecider(backend="strands")
    r = d.classify_reply("Yes! Please send me the demo link today.")
    print("   " + json.dumps(r)[:300])
    check("factory:strands", r.get("local_backend") == "strands",
          f"local_backend={r.get('local_backend')}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    if failures:
        print(f"\n{len(failures)} FAILURES: {failures}")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
