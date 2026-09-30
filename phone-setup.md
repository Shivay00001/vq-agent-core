# Phone setup (Termux, Android)

VQ Agent Core runs on-device: Needle 2 as the ACT router, the local LLM for
THINK, and the DECIDE layer as Jev cloud or the on-phone local decider.
No paid services, no custom video backend — everything here is free.

> On the phone, `decider: local` uses the **prompt-based local LLM decider**
> (Laya needs torch, which has no Termux wheel — the code falls through to
> it automatically). `decider: auto` = Jev cloud first, on-phone decider on
> failure.

## 1. Termux base

```bash
pkg update -y && pkg upgrade -y
pkg install -y python git curl
```

## 2. Get the code

```bash
cd ~
unzip ~/vq-agent-core.zip -d vq-agent-core   # or: git clone <your repo>
cd vq-agent-core
pip install -r requirements.txt
```

`requirements.txt` is pure-Python + `cactus-needle` (Termux-safe). The Laya
lines are commented out — leave them commented on the phone.

## 3. ACT layer: Needle 2 in serve mode

The pip package's auto-downloaded engine is a glibc `.so` and cannot load on
Android's bionic libc, so on Termux we run the official Android CLI binary
as a server instead.

```bash
# download the Android binary (pick your arch; most phones: android-arm64)
mkdir -p ~/vq-agent-core/bin
curl -L https://huggingface.co/Cactus-Compute/needle2/resolve/main/android-arm64/needle \
  -o ~/vq-agent-core/bin/needle
chmod +x ~/vq-agent-core/bin/needle

# dump the tool schema the server needs
python needle_router.py --dump-tools tools.json

# start the server (keep running; use a second Termux session for the agent)
./bin/needle --tools tools.json --serve --port 8080
```

In `config.yaml`:

```yaml
router:
  backend: serve
  needle_bin: "bin/needle"   # path to the Android binary above
  serve_port: 8080
```

## 4. THINK layer: local model endpoint

**Same port everywhere: 11434, same model name everywhere:
`qwen2.5-0.5b-instruct`.** On the VM this is already verified live
(2026-09-27): llama-server + the GGUF below, `config.yaml` pointing at
`http://localhost:11434/v1`. Two free routes:

**A. On-phone (llama.cpp server, offline):**

```bash
# Termux: the llama-cpp package ships llama-server
pkg install -y llama-cpp

# one-command model download (same script as the VM; ~491 MB)
./fetch_model.sh

# same port as the VM config -- 11434
llama-server -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 11434
```

If `pkg install llama-cpp` is unavailable on your Termux, grab the
prebuilt `android-arm64` binary from a llama.cpp GitHub release the same
way `bin/needle` was fetched in step 3.

**B. PC on the same Wi-Fi (faster):** run `llama-server` / Ollama / LM Studio
on your computer, note its LAN IP (e.g. `192.168.1.5`), and use
`http://192.168.1.5:11434/v1` below.

In `config.yaml` (identical on phone and VM):

```yaml
local_model:
  base_url: "http://127.0.0.1:11434/v1"   # route B: http://192.168.1.5:11434/v1
  model: "qwen2.5-0.5b-instruct"           # must match what your server serves
```

The GGUF is the same everywhere: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`,
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (~491 MB) — `./fetch_model.sh` gets it
with a SHA-256 check. `models/` is excluded from the zip, so download it
on each machine.

Multimodal (text+audio+video+image+docs) depends on the model you load:
phone-realistic options are Gemma 3n-class or Qwen2.5-Omni-class via a
compatible on-device runner; a 4B multimodal model realistically wants a
device with **8 GB+ RAM**. On weaker phones use a small text model and the
summary fallback, a LAN-hosted endpoint, or Jev cloud.

## 5. DECIDE layer

```yaml
decider: auto    # Jev cloud first, on-phone local decider if Jev is down
# decider: local  # always on-phone (private/offline/free)
# decider: jev    # always Jev cloud (sharpest)
```

**Jev on the phone (honest version):** Jev is a hosted API. The `jev:` block
in `config.yaml` points at the Hatch-VM helper script; on your phone either
leave `cli: ""` and set the `JEV_API_KEY` env var (create the key in the
TypeSafe dashboard), or keep `decider: local`. There is no official
Android-native Jev runtime — network calls are the only route.

## 6. Run it

```bash
# Termux session 1: needle server (from step 3)
# Termux session 2:
cd ~/vq-agent-core
python agent.py "score this lead: Sharma Dental Clinic, Andheri West, wants appointment reminders, asked for pricing"
```

Traces: `runs/*.jsonl`. Notes: `notes/`.

## Troubleshooting

- `serve backend needs the needle binary` → `router.needle_bin` path is wrong
  or the binary isn't executable (`chmod +x`).
- `needle serve did not open port in time` → the binary may still be loading
  its model; wait and retry. Check it starts at all:
  `./bin/needle --tools tools.json --prompt "hi"`.
- `no local model reachable` → your llama-server URL/port is wrong, or the
  server isn't running. `local` decider and THINK both need it.
- Slow decisions on `local` → the prompt-based decider makes one LLM call
  per question; a 1.5B model on CPU takes seconds. Normal.
