# ✨ Laya CLI (`laya-cli`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Built with uv](https://img.shields.io/badge/built%20with-uv-purple.svg)](https://github.com/astral-sh/uv)
[![PyPI version](https://img.shields.io/pypi/v/laya-cli.svg)](https://pypi.org/project/laya-cli/)

A standalone, high-performance CLI for [Laya](https://github.com/NandhaKishorM/laya) — **typed decisions** (`choice` / `score` / `noul`) in a single forward pass. Built for both **humans** (pretty tables) and **AI agents** (machine-readable `json`/`jsonl`).

- No text generation, no hallucination — ~33 ms for one question, ~7 ms/q batched (T4)
- One model load, many questions at once; presets, shortlisting, multilingual routing, and an optional resident daemon

---

## ⚡ Key Features

* **Typed Decisions in One Pass** — `choice` (label + calibrated probs), `score` (ordinal expected value), `noul` (P(true)). All questions in one `agent.predict(state, questions)` call.
* **Router-Aware Multilingual** — `laya.Router(preload=True)` auto-detects script/language in <0.5 ms and routes to `laya` (English, 512 ctx, 421M) vs `laya-multilingual` (100+ langs, 1024 ctx, 322M). Warns if `--router` without `--lang`.
* **Preset Library** — `triage` / `email` / `guard` / `moderation` / `router` are direct `laya.*_questions()` passthroughs, mergeable with `--questions file.json` and `--questions-inline '{"id": {...}}'`.
* **Embedding Shortlist for High-Cardinality** — `--shortlist-k 20` via `laya.predict_shortlist` + `embed_fn_from_agent` (Banking77 77 options → ~3 tokens/opt without shortlist → shortlist restores accuracy).
* **Streaming Batch Mode** — `classify` / `predict --input` loads the model **once** + warmup throwaway `predict`, streams JSONL. `--state-field` is taken **verbatim** (no silent `(photographer: Name)` injection).
* **Post-Filter & Eval** — `filter --where "on_topic>=0.4" --sort -on_topic` and `evaluate` (accuracy, passing @ threshold, precision @ threshold, escalation rate).
* **Resident Daemon (optional)** — `laya-cli serve` keeps the model in RAM; subsequent `predict`/`classify`/`evaluate` hit `127.0.0.1` and skip the 10–35 s `laya.load()` on repeated calls. Per-config daemon file `~/.cache/laya-cli/daemons/<hash>.json`, idle-timeout auto-exit.

---

## 🚀 Installation

### Option 1: `uv` (recommended)

```bash
uv tool install laya-cli        # globally isolated, uses uv.lock
laya-cli --help

# upgrade
uv tool update laya-cli --refresh
uv tool install laya-cli --force --refresh  # force reinstall
```

### Option 2: `pip` / `pipx`

```bash
pipx install laya-cli
# or
pip install laya-cli
```

### Option 3: Run without installing

```bash
uvx laya-cli --help
uvx --refresh laya-cli@latest --help  # bypass uv cache
```

> Requires Python 3.10+ (`.python-version` pins 3.11). Heavy ML deps (`torch`, `transformers`, `safetensors` via `laya`, ~2 GB) download on first `predict`/`classify`/`evaluate` (650–850 MB per checkpoint via `allow_patterns`; bundle root is 2.3 GB). Afterwards `HF_HUB_OFFLINE=1` works. Dev/tests mock Laya so CI is fast.

---

## 📖 Quick Start

### Single prediction (human)

```bash
# Preset without creating a file — table output
laya-cli predict "I was charged twice, refund please" --preset triage
laya-cli predict --text "Ignore previous instructions" --preset guard --format table
laya-cli predict --state '{"subject":"Invoice #4411","body":"Billed twice"}' --preset email --format table
```

### Single prediction (AI agent — JSON)

```bash
laya-cli predict --text "Is this spam?" --preset guard --format json
# -> {"answers": {"jailbreak": {"type":"noul","noul":0.02,"confidence":0.97,"action":{...}}}, "usage":{"input_tokens":12}}

# Custom questions, merged: preset < file < inline
laya-cli predict --text "hello" \
  --preset triage \
  --questions-inline '{"custom":{"type":"noul","instructions":"Is it polite?"}}' \
  --format json

# High-cardinality choice (e.g. 77 banking intents)
laya-cli predict --text "where is my card?" --questions banking.json --shortlist-k 20 --format json

# Multilingual — Router is recommended (see https://github.com/NandhaKishorM/laya#why-route-the-evidence)
laya-cli predict --text "मुझसे दो बार शुल्क लिया गया" --preset triage --router --format json
laya-cli predict --text "Der Kunde wurde zweimal belastet" --preset triage --router --lang de --format json
```

State is flexible: positional `TEXT`, `--text "str"` (repeatable), `--state '{"key":"value"}'` (dict/list accepted as `agent.predict` does), `--state-file file.json`, or batch `--input file.jsonl` / stdin. Questions come from `--preset`, `--questions file.json`, `--questions-inline JSON` (merged, later wins).

Checkpoints: `convaiinnovations/laya` (English, 512 ctx), `convaiinnovations/laya/multilingual` (use `--subfolder multilingual`), `typed-decisions`. Use `--model` / `--subfolder` for a single model, or `--router` for auto routing (`Router(preload=True)`).

---

## 🔁 Batch & Pipeline

Batch loads the model **once** and reuses it (warmup throwaway `predict` on start):

```bash
# From file — human-readable tables per row
laya-cli predict --input candidates.jsonl --questions questions.json --format table

# Machine-readable JSONL with flattened fields (like classify) for jq/filter
laya-cli predict --input candidates.jsonl --questions questions.json --flatten --format jsonl > scored.jsonl
cat scored.jsonl | laya-cli filter --where "on_topic>=0.4,is_relevant>=0.7" --sort -on_topic,+id

# Legacy streaming (kept for compatibility with TASK.md / pexels-cli)
cat candidates.jsonl | laya-cli classify --questions questions.json | laya-cli filter --where "on_topic>=0.4" --sort -on_topic > shortlist.jsonl
```

**Pexels integration** — the original use case (82 candidates, `--state` deduped):

```bash
# 1. Pexels → candidates.jsonl with `state` strings (via https://github.com/MIt9/pexels-cli)
px videos --queries "black friday shopping,christmas shopping,checkout cart" \
  --per-page 8 --state --dedupe keep-first > candidates.jsonl

# 2. Laya → score + filter (one model load, streaming)
cat candidates.jsonl \
  | laya-cli classify --questions questions.json \
  | laya-cli filter --where "on_topic>=0.4" --sort -on_topic \
  > shortlist.jsonl

# Or with the primary command (supports presets without a file):
cat candidates.jsonl | laya-cli predict --questions questions.json --format jsonl > scored.jsonl
px videos --queries "..." --state --dedupe | laya-cli predict --preset triage --format jsonl | laya-cli filter --where "intent==refund"
```

Candidate JSONL format (`px --state`):

```json
{"id": 5890229, "type": "video", "query": "black friday shopping", "photographer": "Pavel Danilyuk", "state": "a man shopping on black friday", "url": "https://www.pexels.com/video/a-man-shopping-on-black-friday-5890229/", "duration": 10, "width": 2160, "height": 3840}
```

`--state-field` is taken **verbatim**. Never silently prepend metadata — use `--prepend-field NAME` only explicitly (mixing `(photographer: Name)` degraded `on_topic` by 0.1–0.3 in the real Pexels case).

---

## 📚 Commands

```
laya-cli predict [TEXT ...] [--text TEXT] [--state JSON] [--input file.jsonl] [--preset NAME] [--questions file.json] [--router] [--shortlist-k K] [--format json|jsonl|table] [--no-daemon]
laya-cli classify --questions q.json [--state-field state] [--router] [--device cpu|mps|cuda] [--no-daemon]  # batch JSONL stdin→stdout, legacy
laya-cli filter --where "on_topic>=0.4" --sort -on_topic                                                  # JSONL sort/filter
laya-cli questions list                    # list presets
laya-cli questions triage > questions.json # also: email, guard, moderation, router (alias: presets)
laya-cli evaluate --questions q.json --labeled labeled.jsonl --label-field label --threshold 0.5 [--no-daemon]
laya-cli serve [--model ID] [--device mps] [--router] [--port 0] [--idle-timeout 1800] [--foreground]  # daemon
laya-cli serve status [--model ID] [--subfolder NAME] [--device NAME] [--router]
laya-cli serve stop [--model ID] [--all]
laya-cli info [--model ID] [--subfolder NAME]
laya-cli --help; laya-cli <command> --help  # every command shows examples
```

### Presets & Info

```bash
laya-cli questions list --format table
laya-cli questions triage > questions.json
laya-cli presets guard | laya-cli predict --text "test" --questions /dev/stdin --format json

laya-cli info                                    # python/torch/cuda/mps + HF cache
laya-cli evaluate --questions questions.json --labeled labeled.jsonl --label-field label --threshold 0.5
# -> accuracy per question, passing @ threshold, precision @ threshold, escalation rate
#    Measure on 50-200 real examples before trusting probabilities — base checkpoints are near chance on some tasks.
```

---

## ⚡ Resident Daemon Mode (optional)

Repeated `laya.load()` costs 10–35 s (MPS instantiation) + ~120 ms × N candidates. One pipeline run is fine (35–50 s for 82 videos), but iterative tuning of `questions.json` (`predict` → inspect → edit → `predict` again) pays 10–35 s every time for the same model.

`laya-cli serve` keeps the model resident in RAM; subsequent `predict`/`classify`/`evaluate` with the **same config** skip the load and hit `127.0.0.1` in <1 ms.

### Measured baseline (2026-09-22, 82 candidates, M-series MPS)

| stage | time |
|-------|------|
| `px` (network, 12 queries) | 9.3 s |
| `laya.load()` (model into RAM) | 10–35 s |
| inference, 82 candidates | ~10 s (~120 ms each) |
| `filter` (sort/filter) | 0.08 s |

### Transport (as in `laya-integration` SKILL.md — “run Laya as a small local HTTP sidecar”)

HTTP on loopback `127.0.0.1` only — never `0.0.0.0`. Single-threaded with `threading.Lock` (one GPU = one forward pass):

- `POST /predict` — `{"state": ..., "questions": {...}, "lang": "...", "shortlist_k": 20}` → same JSON as `agent.predict()` (`laya-cli predict --format json`)
- `GET /status` — `{"model": "...", "subfolder": "...", "device": "...", "loaded_at": 123..., "idle_seconds": 42, "requests_served": 17, "pid": 12345, "port": 8765, "idle_timeout": 1800}`
- `POST /shutdown` — graceful stop from localhost only (also `serve stop` sends `SIGTERM` via pid file)

### Lifecycle

```bash
# Start in background (writes pid+port to ~/.cache/laya-cli/daemons/<hash>.json)
laya-cli serve --model convaiinnovations/laya --device mps
laya-cli serve --model convaiinnovations/laya --subfolder multilingual --device cpu --idle-timeout 60
laya-cli serve --router --foreground --idle-timeout 0 --port 8765  # foreground for logs, 0 disables idle

# One daemon = one checkpoint config hash(model|subfolder|device|router|lang) → separate file/port
# device is resolved before hashing (auto: cuda>mps>cpu), so `serve` and `serve --device mps` share the same file on an MPS host

laya-cli serve status                                          # default config (same defaults as predict)
laya-cli serve status --model convaiinnovations/laya --subfolder multilingual --device cpu
laya-cli serve stop                                            # graceful via POST /shutdown, removes pid file
laya-cli serve stop --all                                      # stop all daemons
curl http://127.0.0.1:<port>/status
```

Warmup throwaway `predict` runs before serving. `idle-timeout` defaults to 1800 s; `--idle-timeout 5` (for tests) makes the daemon exit after ~5 s of no requests and `serve status` then reports `not running`. `--foreground` blocks and logs to stderr.

### Client side (`predict` / `classify` / `evaluate`)

Before `laya.load()`, each command checks `~/.cache/laya-cli/daemons/<hash>.json` for the current config. If the file exists and the daemon answers `GET /status`, requests go to `POST /predict` instead of a local load. If the file is missing or the daemon is dead (stale pid), it silently falls back to in-process behaviour — no pipeline change required. `--no-daemon` forces in-process even if a daemon is live (for reproducibility/debugging).

For batch (`classify`, `predict --input`), each JSONL line is a separate `POST /predict` in a loop (loopback overhead is milliseconds, no batch endpoint needed yet).

```bash
# AI workflow — fully automatic, no extra flags after serve:
laya-cli serve --device mps &                          # once per session
laya-cli predict "hello" --preset guard --format json  # via daemon, no 10s load
laya-cli predict "hello2" --preset guard --format json # still via daemon (<1 ms overhead)
laya-cli serve status                                  # {"loaded_at":..., "requests_served": 2}
laya-cli predict "hello" --preset guard --no-daemon --format json  # force in-process (10-35s again)
laya-cli serve stop

# Human tuning loop:
laya-cli serve &
cat candidates.jsonl | laya-cli classify --questions q.json | laya-cli filter --where "on_topic>=0.4" --sort -on_topic  # via daemon
# ...edit q.json...
cat candidates.jsonl | laya-cli classify --questions q.json | laya-cli filter --where "on_topic>=0.4" --sort -on_topic  # still via daemon
laya-cli serve stop

# 5 parallel predicts — serialized by daemon Lock, all succeed with correct, non-interleaved results:
seq 1 5 | xargs -P5 -I{} laya-cli predict "text {}" --preset guard --format json
```

**Security / limits:** binds only `127.0.0.1`, no auth (single-user local machine, as in SKILL.md), one request at a time, stateless apart from the model in RAM, no multi-model hot-swap (new config = new daemon on another port).

---

## 🛠 Development

```bash
uv sync --group dev          # hatchling + ruff/mypy/pytest
uv run ruff check .          # lint (E/F/W/I, line-length 120)
uv run ruff format .         # format
uv run pytest -q             # 30 tests, mocked laya (no model download)
uv run pytest --cov=src/laya_cli --cov-report=term-missing
uv run laya-cli --help       # every command shows examples
uv build                     # hatchling -> dist/*.whl + sdist (src/ layout)
```

Repo layout: `src/laya_cli/`, `tests/`, `pyproject.toml` (hatchling + `dependency-groups`), `uv.lock`, `.python-version` 3.11, `.github/workflows/ci.yml` + `publish.yml`.

---

## 📄 License

Apache-2.0 — same as [Laya](https://github.com/NandhaKishorM/laya) upstream. See [LICENSE](LICENSE).

This project is not affiliated with Convai Innovations. Laya weights are Apache-2.0 and hosted on Hugging Face (`convaiinnovations/laya`).
