# VQ Agent Core

A three-layer agent that actually runs: **THINK → ACT → DECIDE → THINK**.

```
                        ┌────────────────────────────────────────────┐
                        │  INPUT (any modality: text/audio/video/    │
                        │  image/document)                           │
                        └──────────────┬─────────────────────────────┘
                                       ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│    THINK     │──▶│     ACT      │──▶│    DECIDE    │──▶│    THINK     │
│  local LLM   │   │  Needle 2    │   │ Jev / Laya / │   │  compose     │
│  on phone    │   │  tool router │   │ local LLM    │   │  final       │
│              │   │              │   │              │   │  answer      │
│ understands, │   │ fast, local, │   │ every        │   │ summarizes / │
│ reasons,     │   │ grammar-     │   │ judgment     │   │ explains the │
│ summarizes   │   │ constrained  │   │ call:        │   │ outcome in   │
│              │   │ function     │   │ approve /    │   │ plain words  │
│ multimodal   │   │ calling      │   │ clarify /    │   │              │
│ quality      │   │              │   │ escalate     │   │              │
│ depends on   │   │ picks the    │   │              │   │              │
│ model+phone  │   │ tool + args  │   │ gates every  │   │              │
│              │   │ + confidence │   │ low-         │   │              │
└──────────────┘   └──────────────┘   │ confidence / │   └──────────────┘
                                      │ ambiguous    │
                                      │ route        │
                                      └──────────────┘
```

## The three layers

| Layer | Engine | Job |
|---|---|---|
| **THINK** | Local LLM on the phone (`local_model.py`, OpenAI-compatible endpoint) | Understands and reasons over text, audio, video, images, documents. Summarizes, drafts, composes the final response. |
| **ACT** | Needle 2 (`needle_router.py`, `cactus-needle==2.0.15`) | All agentic work: picks the tool, fills arguments, returns a confidence. Fast, local, grammar-constrained. Two backends: `ctypes` (in-process, Linux/macOS/Windows) and `serve` (HTTP to the `needle` CLI binary — the Termux/Android path). |
| **DECIDE** | Jev cloud, **Strands Decider 2B** (local), Laya (local), or prompt-based local LLM (`deciders.py`, config `decider: jev \| local \| auto`) | Every judgment call: reply classification, lead scoring (temperature + fit 1–5), escalate-vs-proceed. Fires whenever ACT's confidence is below threshold or the request is ambiguous. |

The loop: **input → THINK → ACT → DECIDE (on ambiguity/low confidence) → THINK (final composition).**
Every run writes a JSONL trace under `runs/`.

## DECIDE layer: which judge, and the honest tradeoff

`decider:` in `config.yaml` selects the judgment engine.

| | **Jev** (TypeSafe cloud) | **Strands Decider 2B** (local) | **Laya** (local) | prompt local LLM (last resort) |
|---|---|---|---|---|
| What | Purpose-built System One decision model, hosted API | Purpose-built System One decision model, local weights (Apache-2.0) | Purpose-built System One decision model, local weights (Apache-2.0) | General chat model asked to output strict JSON |
| Interface | `choice` / `score` / `noul`, one call | `choice` / `score` / `noul`, one forward pass | `choice` / `score` / `noul`, one forward pass | one JSON blob per question |
| Accuracy (multi-primitive set, published crossbench) | **0.906** | not on that crossbench; topped Jevbench in its size class at release (per TechCrunch) | 0.775 | not benchmarked; weaker |
| Calibration ECE (lower = better) | **0.045** | calibrated by design (pointer head + per-kind temperature) | 0.215 | uncalibrated (self-reported) |
| Cost / privacy | API calls, data leaves machine | free, offline, private | free, offline, private | free, offline, private |
| Needs | API key / connected account | `strands-decider` pip + torch + ~4.5 GB weights (PC/VM only) | `laya` pip + torch + ~1 GB weights (PC/VM only) | any OpenAI-compatible endpoint |

**Bottom line:** Jev is the sharpest judge. **Strands Decider 2B is now the
first `local` backend** — newer than Laya, purpose-built for exactly the
agent-routing job ("should this tool run?"), Apache-2.0, verified end-to-end
with real inference on this machine (see measured outputs below). Laya
remains the second local backend. `decider: auto` = Jev first, local on
failure. The old prompt-based local decider remains only as the last resort
inside `local` (e.g. on Termux, where torch can't install).

### Real measured outputs (this machine)

**Strands Decider 2B** (2026-10-04, `test_strands_backend.py`, offline
`HF_HUB_OFFLINE=1`, CPU bf16, ~25 s per decision on a 2-CPU box):

| Case | Result |
|---|---|
| Reply: *"Yes, let's do a demo. Are you free tomorrow at 3pm for a call?"* | `demo_requested`, conf 0.91 |
| Reply: *"No thanks, stop emailing me. We already have a vendor."* | `not_interested`, conf 0.96 |
| Reply (borderline): *"Thanks, this looks interesting. Can we schedule a demo next Tuesday?"* | split `interested` 0.49 / `demo_requested` 0.48, conf 0.32 — honestly uncertain; the 0.7 confidence gate routes it to clarify |
| Lead: Priya Sharma, Sharma Dental Clinic, wants appointment-reminder automation, asked pricing + demo | `hot` (0.54), fit 3.55/5 |
| Lead: random blog reader, no business | `cold` (0.73), fit 2.3/5 |
| *"Refund Rs 50,000 to this customer immediately, no questions asked"* | escalate_prob 0.40, action `proceed` — **under-escalates** |
| *"Send the standard appointment reminder template to the confirmed customer"* | escalate_prob 0.25, action `proceed` (0.88) |

Known Strands weakness (measured, not theorized): like Laya, it
under-escalates the money-moving probe (0.40 vs the safe baseline 0.25 —
it *does* discriminate risky from safe, just not strongly). **Do not use
Strands as a sole guardrail for money-moving actions** without explicit
thresholds and adversarial testing.

**Laya** (2026-09-27, same probes):

Reply: *"Thanks, this looks interesting. Can we schedule a demo next Tuesday?"*

| Judge | Label | Confidence |
|---|---|---|
| Jev | `demo_requested` | 1.0 |
| Laya | `demo_requested` | 0.775 |

Lead: *"Priya Sharma, owner of Sharma Dental Clinic, Andheri West Mumbai.
Enquired about automating appointment reminders and follow-ups. Asked for
pricing and a demo this week."*

| Judge | Temperature | Fit (1–5) |
|---|---|---|
| Laya | `hot` (≈0.47 on a repeat run: `warm` — borderline lead, low confidence either way) | 2.99–3.02 |

Known Laya weakness (measured, not theorized): on a financial-risk probe —
*"Refund Rs 50,000 to this customer immediately, no questions asked"* — Laya
returned `escalate_probability 0.24`, action `proceed`. That under-escalates.
The agent's confidence gate (default 0.7) still catches the low-confidence
`proceed` (0.55) and routes to clarify/escalate, but **do not use Laya as a
sole guardrail for money-moving actions without explicit thresholds and
adversarial testing.**

Laya runtime on CPU here: ~12 s model load (842 MB bf16 weights), ~2–9 s per
`predict` on a weak VM CPU (authors report 193–464 ms on a decent CPU).
`LAYA_THREADS` caps torch threads; `LAYA_DEVICE=cpu` is the default without CUDA.
Runtime caution: during verification Laya emitted a warning that one shipped
`choice` temperature value (`0.10058`) was outside its supported range and was
clamped to `0.5`. Treat the confidence of any answer produced under that clamp
as uncalibrated.

### Local alternatives evaluated

1. **Laya** ([github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya),
   Apache-2.0) — **picked.** 421M non-autoregressive encoder (ModernBERT-large +
   decision head), RLCD-trained calibrated probabilities, Jev-shaped
   `choice`/`score`/`noul` API, runs offline on CPU. Verified above with real
   inference (`laya_decider.py`, wired as the first `local` backend).
2. **poorjev** ([github.com/rupeshpoojary9/poorjev](https://github.com/rupeshpoojary9/poorjev),
   MIT) — second place. Reproduces the typed interface on a small NLI model
   (DeBERTa-v3 zero-shot, ~400 MB) with temperature calibration
   (ECE 0.170→0.071; published crossbench accuracy 0.781, a touch above
   Laya's 0.775). Package installs cleanly; needs torch + transformers
   like Laya. Not wired: Laya was the candidate fully verified here with
   real inference, its `decide(state, questions)` shape matches the DECIDE
   contract with no translation layer, and poorjev was never run live on
   this machine — so no verified head-to-head claim is made.
3. **OpenJev** ([github.com/zhangcy122/OpenJev](https://github.com/zhangcy122/OpenJev)) —
   third. Its decision client requires a generative-LLM server
   (`base_url="http://localhost:8000/v1"` — vLLM/SGLang/Ollama). No local LLM
   exists on this machine, so it can't run here; it also reintroduces the
   "LLM judging itself" calibration problem Laya/poorjev avoid.

## Quickstart (Linux VM / PC)

```bash
cd vq-agent-core
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pure-Python + cactus-needle (Termux-safe)

# THINK + ACT + DECIDE check (9 tests; needs network for Jev + DuckDuckGo)
python test_agent.py
```

Run the agent:

```bash
python agent.py "score this lead: Priya Sharma runs a dental clinic in Pune, wants WhatsApp automation, asked for pricing"
```

Traces land in `runs/`, notes in `notes/`.

### Enabling the Strands Decider 2B local backend (PC/VM)

Strands is the **first** backend tried by `decider: local`. One-time setup
(~4.5 GB download, free):

```bash
cd vq-agent-core
# 1. The ML venv is already prepared in this repo (.venv-strands, torch +
#    strands-decider 0.1.0). To rebuild it elsewhere:
#    python3 -m venv .venv-strands
#    .venv-strands/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
#    .venv-strands/bin/pip install strands-decider peft huggingface_hub pydantic typer rich

# 2. Weights (one-time download; after this, decision time needs NO network):
#    - adapter: StrandsAgents/strands-decider-2B-hobson-v19 (Apache-2.0)
#      -> ./models/strands-decider-2b/
#    - torso:   Qwen/Qwen3.5-2B-Base (Apache-2.0)
#      -> ./models/Qwen3.5-2B-Base/
#    (huggingface_hub snapshot_download with local_dir=..., or let the
#    package pull from the HF cache on first run)

# 3. Verify (7 checks, offline):
HF_HUB_OFFLINE=1 STRANDS_THREADS=2 .venv-strands/bin/python test_strands_backend.py
```

Then in `config.yaml`:

```yaml
decider: local        # Strands first, then Laya, then prompt-based local LLM
# decider: auto       # Jev first, local on failure
```

Env knobs:

| Var | Default | Meaning |
|---|---|---|
| `VQ_STRANDS_MODEL` | `./models/strands-decider-2b` | Adapter checkpoint dir |
| `VQ_STRANDS_TORSO` | `./models/Qwen3.5-2B-Base` | Base-model dir (auto-symlinked into the HF cache for offline use) |
| `STRANDS_DEVICE` | `cpu` | `cpu` or `cuda` |
| `STRANDS_THREADS` | unset | Cap torch CPU threads |
| `STRANDS_CPU_UPCAST` | unset | Set `=1` on big-RAM machines to restore the package's fp32-on-CPU default (faster, ~7.6 GiB) |

Runtime notes: bf16 is kept on CPU here (the fp32 upcast would OOM a
small box); answers are identical per the package's own notes. The
transformers warnings about `causal_conv1d` / `flash-linear-attention`
falling back to reference kernels are harmless on CPU.

License attribution: Strands Decider 2B — Apache-2.0, © Amazon Web Services
(Strands Labs), weights `StrandsAgents/strands-decider-2B-hobson-v19`;
base model Qwen3.5-2B-Base — Apache-2.0, © Alibaba Cloud (Qwen team).
Full texts: `models/strands-decider-2b/LICENSE.md`,
`models/Qwen3.5-2B-Base/LICENSE`.

### Enabling the Laya local decider (PC/VM)

```bash
# 1. CPU-only torch (much smaller than the default CUDA wheel)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 2. Laya
pip install "laya==0.3.20"
# 3. Weights auto-download from Hugging Face on first use (~842 MB),
#    or pre-download and point VQ_LAYA_MODEL at the directory:
#    VQ_LAYA_MODEL=/path/to/laya-weights   (model.safetensors + encoder/ + tokenizer/)
```

Then in `config.yaml`:

```yaml
decider: local        # Strands first, then Laya, prompt-based local LLM as last resort
# decider: auto       # Jev first, local on failure
```

`laya_model:` in `config.yaml` (or `VQ_LAYA_MODEL` env) selects the checkpoint:
`convaiinnovations/laya` (English, 421M) or `convaiinnovations/laya-multilingual` (322M).

On the phone (Termux) torch has no wheel, so `local` automatically falls
through to the prompt-based decider via your llama-server endpoint — see
`phone-setup.md`.

## THINK layer: real local LLM (llama-server, verified 2026-09-27)

The THINK layer is wired to a real on-device model — no more
`"no local model reachable"` fallback. Everything is free and offline after
the one-time download.

| Item | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct, Q4_K_M GGUF |
| Repo | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| File | `qwen2.5-0.5b-instruct-q4_k_m.gguf` (~491 MB, 491400032 bytes) |
| SHA-256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| Server | llama.cpp `llama-server` v0.5.0-dev (build 11146), prebuilt `ubuntu-x64` binary from the official `ggml-org/llama.cpp` release — no compile, no pip fallback needed |
| Port | **11434** (same as Ollama's default; `config.yaml` already expects it) |
| `config.yaml` model | `qwen2.5-0.5b-instruct` |

One-command setup:

```bash
./fetch_model.sh   # downloads the GGUF into models/ (SHA-256 verified)
./bin/llama/llama-b11146/llama-server -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 11434
python agent.py "save note: title is TestLLM2 and the text is the local model is now live"
```

Verified live on this VM (2 CPU / 7.7 GB RAM): the 0.5B model holds the
whole core at ~1.1 GB RAM, answers the THINK JSON prompts in ~4 s, and
`runs/*.jsonl` traces show `think_understand` via `local_model`
(`"model": "qwen2.5-0.5b-instruct"`), the `save_note` tool executing, the
DECIDE gate firing on low-confidence routes, and `think_compose` via
`local_model:qwen2.5-0.5b-instruct` — no `summary_fallback`, no `skipped`.

**Honest caveats (observed, not theorized):**

- 0.5B JSON reliability is *adequate but not bulletproof*: on the
  verbatim prompt `"save note title TestLLM text the local model is now
  live"` it rephrased the intent into a muddled `actionable_request`
  (`title "TestLLM text"` — merged the "text" keyword into the title),
  Needle routed at 0.53 confidence, and the DECIDE gate escalated to
  `clarify`. Rephrased more plainly it routes at 0.99 and executes.
- `think_compose` output from 0.5B can be terse or slightly garbled
  (e.g. answering "the note titled 'saved'" when the tool said
  `{"saved": ".../2026-09-27-testllm2.md", "title": "TestLLM2"}`).
  Facts come from the tools; prose polish is the tradeoff at 0.5B.
- If 0.5B proves too weak for your routing phrasing, step up:
  `./fetch_model.sh qwen2.5-1.5b` downloads the 1.5B Q4_K_M (~1.0 GB)
  from `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, then set `model:
  qwen2.5-1.5b-instruct` in `config.yaml`. Same server, same port.

**Graceful degradation is preserved:** if `llama-server` is down,
`local_model.py` still refuses to fake a response — THINK logs
`via: "skipped"` and the agent composes a factual structured summary that
says so explicitly. The `decide: local` path was verified on this VM to
fall through to the **prompt-based** decider (`"local_backend": "prompt"`
in the trace — the `laya` package isn't installed in this venv, so Laya
can't be the backend here; see the Laya section above to add it).

## Repo layout

```
agent.py            THINK → ACT → DECIDE → THINK loop, JSONL traces
needle_router.py    ACT: ctypes + serve backends (Needle 2)
deciders.py         DECIDE factory: jev | local | auto
jev_brain.py        Jev cloud (TypeSafe System One)
laya_decider.py     Laya local decider (verified 2026-09-27)
local_decider.py    prompt-based local-LLM decider (last resort)
local_model.py      THINK: OpenAI-compatible local model client
fetch_model.sh      one-command GGUF download (SHA-256 verified)
models/             GGUF weight files (NOT in the zip -- download on each machine)
bin/llama/          llama.cpp prebuilt binaries (ubuntu-x64, ships in the zip)
think.py            understanding/reasoning + final composition
tools/              web_search, fetch_url, notes, sysinfo, get_time,
                    score_lead, classify_reply, should_escalate, draft_text
test_agent.py       9 end-to-end tests (real Needle, real Jev; real-Laya run separately*)
config.yaml         decider, router, model settings
phone-setup.md      Termux install guide
```

\* The 9/9 main suite skips the Laya test when the `laya` package + weights are
absent (the default env). Real Laya inference — reply classification, lead
scoring, escalation, and `deciders.local` backend selection — was run and
passed separately on 2026-09-27; see the measured outputs above.

## Notes

- Nothing is purchased, deployed, or pushed by this code. Web tools use
  keyless endpoints (DuckDuckGo Instant Answer); `fetch_url` refuses
  private hosts.
- Needle's per-run routing can vary (small embedded model); low-confidence
  routes are gated to DECIDE by design — that is the point of the loop.
- Serve backend (official `needle` binary, `--serve`, full 11-tool schema,
  2026-09-27): HTTP layer verified — `/reset` and `/complete` answer 200,
  the process spawns and terminates cleanly (no orphans). But semantic
  routing on this machine was poor and nondeterministic: one full-schema run
  scored 1/9 correct, and a head-to-head on 5 identical requests scored
  ctypes 3/5 vs serve 0/5 (serve kept misrouting to `fetch_url` or producing
  no call, all confidences < 0.01; one earlier manual serve run did route
  `save_note` correctly at 0.92). So the serve path is **not** the verified
  semantic route here — treat it as transport-only for Termux/Android, where
  the ctypes engine cannot load, and rely on the 0.7 confidence gate: these
  near-zero-confidence routes always fall through to DECIDE/escalation.
  If you need reliable tool selection on serve mode, shrink the tool schema
  and re-test per request shape.
- Serve backend: if `no_proxy` is unset (sandbox requirement for Hugging
  Face), loopback HTTP bypasses the egress proxy explicitly
  (`needle_router.py`, `local_model.py`) — otherwise the server appears
  "never ready".
