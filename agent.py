"""VQ Agent Core -- the agentic loop.

Locked architecture (Shivay's 3 layers):

  input (text / audio / video / doc)
      |
      v
  THINK  (local LLM: understands + reasons; multimodal when the model is)
      |
      v
  ACT    (Needle 2: fast local grammar-constrained tool routing + execution)
      |
      v  (low confidence or ambiguity)
  DECIDE (Jev cloud and/or local LLM: approve / clarify / escalate,
          lead scoring, reply classification, plan choice)
      |
      v
  THINK  (local LLM composes the final response, or a factual summary
          fallback when no local model is reachable -- never fake prose)

Every step is appended as JSONL to runs/run-<timestamp>.jsonl.

Usage:
    python agent.py "score this lead: ..."
    python agent.py --config my.yaml "do X"
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402

import tools  # noqa: E402
import think  # noqa: E402
import deciders  # noqa: E402
from local_model import LocalModel  # noqa: E402
from needle_router import Router  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    path = path or os.path.join(BASE, "config.yaml")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Agent:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        # DECIDE layer selection; tools read the same via VQ_DECIDER.
        os.environ["VQ_DECIDER"] = str(self.cfg.get("decider", "jev"))
        # Laya model location for the local decider (env wins if already set).
        os.environ.setdefault("VQ_LAYA_MODEL", str(self.cfg.get("laya_model", "convaiinnovations/laya")))
        self.decider = deciders.from_config(self.cfg)
        self.threshold = float(self.cfg["router"].get("confidence_threshold", 0.7))
        self.router = Router(self.cfg)          # ACT layer (Needle 2)
        self.local_model = LocalModel(          # THINK layer (local LLM)
            base_url=self.cfg["local_model"].get("base_url"),
            model=self.cfg["local_model"].get("model"),
            timeout=int(self.cfg["local_model"].get("timeout", 120)),
        )
        runs_dir = self.cfg.get("runs_dir", "runs")
        if not os.path.isabs(runs_dir):
            runs_dir = os.path.join(BASE, runs_dir)
        os.makedirs(runs_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.trace_path = os.path.join(runs_dir, f"run-{stamp}.jsonl")
        self._trace_fh = open(self.trace_path, "a", encoding="utf-8")

    # ---------- tracing ----------

    def _log(self, event, **fields):
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
               "event": event}
        rec.update(fields)
        self._trace_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._trace_fh.flush()

    # ---------- ACT ----------

    def _execute(self, name, args):
        fn = tools.get(name)
        if fn is None:
            return {"error": f"unknown tool: {name}"}
        self._log("tool_start", tool=name, args=args)
        try:
            raw = fn(**args)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:
            return {"error": f"{name} raised: {exc}"}
        try:
            result = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            result = {"output": raw}
        self._log("tool_result", tool=name, result=result)
        return result

    # ---------- DECIDE ----------

    def _decide_gate(self, request, tool_name, args, confidence, reason):
        """Ask the DECIDE layer what to do about a weak/ambiguous route."""
        self._log("decide_gate", layer="DECIDE", decider=self.decider.name,
                  reason=reason, tool=tool_name, confidence=confidence)
        try:
            res = self.decider.should_escalate(
                request,
                context=f"Router picked tool '{tool_name}' with args {args} "
                        f"at confidence {confidence:.2f}. Reason for gate: {reason}.",
            )
        except Exception as exc:
            # No decider reachable at all -> safest is human review.
            self._log("decider_unavailable", detail=str(exc),
                      fallback="escalate (safe default)")
            return {"decision": "escalate",
                    "note": f"No decider reachable ({exc}); safe default: escalate."}
        if isinstance(res, dict):
            res = dict(res)
            res.pop("raw", None)
        self._log("decide_decision", result=res)
        return {"decision": res.get("action") or "escalate",
                "escalate_probability": res.get("escalate_probability"),
                "via": res.get("via", self.decider.name),
                "note": f"DECIDE layer ({res.get('via', self.decider.name)}) applied."}

    # ---------- full loop ----------

    def run(self, request, media_paths=None):
        self._log("request", text=request, media=media_paths,
                  act_backend=self.router.backend.name,
                  decider=self.decider.name)

        # ---- THINK: understand + reason ----
        thought = think.understand(request, media_paths, self.local_model)
        self._log("think_understand", layer="THINK", **{
            k: v for k, v in thought.items()})
        actionable = thought.get("actionable_request") or request
        steps = []

        # ---- ACT: Needle 2 routes, we execute ----
        tool_name, args, confidence, note = self.router.route(actionable)
        self._log("route", layer="ACT", tool=tool_name, args=args,
                  confidence=confidence, note=note)

        # ---- DECIDE: gate weak / ambiguous routes ----
        action = "proceed"
        gate_note = ""
        if tool_name is None or confidence < self.threshold:
            reason = note or f"confidence {confidence:.2f} < threshold {self.threshold}"
            gate = self._decide_gate(request, tool_name, args, confidence, reason)
            action = gate["decision"]
            gate_note = gate.get("note", "")
            if tool_name is None and action == "proceed":
                action = "escalate"  # nothing to execute; safest is human review
                gate_note += " (No tool was routed, so proceeding is impossible.)"

        if action == "escalate":
            answer = ("This needs a human review. " + gate_note
                      + (f" Router note: {note}." if note else ""))
            steps.append(f"Escalated to human. {gate_note}")
            self._log("escalated", answer=answer)
            return self._finish(request, steps, escalated=True)

        if action == "clarify":
            answer = ("I need a bit more detail before acting. " + gate_note
                      + (f" (Router picked '{tool_name}' at confidence "
                         f"{confidence:.2f}.)" if tool_name else ""))
            steps.append("Asked user for clarification.")
            self._log("clarify", answer=answer)
            return self._finish(request, steps, escalated=False)

        # ---- ACT: execute the approved tool ----
        result = self._execute(tool_name, args)
        if isinstance(result, dict) and result.get("error") == "decider_unavailable":
            steps.append(f"Tool '{tool_name}' could not reach any decider: "
                         f"{result.get('detail')}")
        elif isinstance(result, dict) and "error" in result:
            steps.append(f"Tool '{tool_name}' failed: {result['error']}")
        else:
            steps.append(
                f"Tool '{tool_name}' executed (confidence {confidence:.2f}): "
                f"{json.dumps(result, ensure_ascii=False, default=str)[:800]}")
        self._log("done")
        return self._finish(request, steps, escalated=False)

    def _finish(self, request, steps, escalated):
        # ---- THINK: compose the final response ----
        answer, via = think.compose(request, steps, self.local_model)
        self._log("think_compose", layer="THINK", via=via)
        self._close()
        return {"answer": answer, "escalated": escalated,
                "trace": self.trace_path}

    def _close(self):
        try:
            self._trace_fh.close()
        except Exception:
            pass
        self.router.close()


def main(argv):
    cfg_path = None
    if argv and argv[0] == "--config":
        cfg_path, argv = argv[1], argv[2:]
    if not argv:
        print("usage: python agent.py [--config cfg.yaml] \"<request>\"")
        return 2
    agent = Agent(load_config(cfg_path))
    out = agent.run(" ".join(argv))
    print(out["answer"])
    print(f"\n[trace: {out['trace']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
