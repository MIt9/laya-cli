# ✨ Laya CLI (`laya-cli`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Built with uv](https://img.shields.io/badge/built%20with-uv-purple.svg)](https://github.com/astral-sh/uv)

A modern, high-performance CLI for [Laya](https://github.com/NandhaKishorM/laya) — **typed decisions** (`choice`/`score`/`noul`) in one forward pass, designed for **Humans** (Rich table output) and **AI Agents / Classifiers** (Machine-readable `--json` & JSONL).

---

## ⚡ Key Features

* 🤖 **Typed Decisions in One Pass**: `choice` (top label + probs), `score` (ordinal), `noul` (P(true)) — no generation, no hallucination, ~33ms on T4, 7ms/q batched
* 🧭 **Router-Aware Multilingual**: `laya.Router(preload=True)` auto-detects script/language in <0.5ms and dispatches to `laya` (English, 512 ctx) vs `laya-multilingual` (100+ langs, 1024 ctx); warns if `--router` without `--lang`
* 🧹 **Embedding Shortlist for High-Cardinality**: `--shortlist-k 20` via `laya.predict_shortlist` + `embed_fn_from_agent` (Banking77 77 opts → 3 tokens/opt without shortlist → shortlist fixes)
* 🔍 **Preset Library**: `triage` / `email` / `guard` / `moderation` / `router` — direct `laya.*_questions()` passthrough, mergeable with `--questions file.json` and `--questions-inline`
* 📦 **Streaming Batch Mode**: `classify`/`predict --input` loads model **once** + warmup, streams JSONL (`--state-field` verbatim, no silent `(photographer: Name)` injection)
* 🔧 **Post-Filter & Eval**: `filter --where "on_topic>=0.4" --sort -on_topic` and `evaluate` (accuracy, passing @ threshold, precision, escalation rate)

---

## 🚀 Global Installation

### Option 1: Install globally via `uv` (Recommended)

```bash
uv tool install laya-cli
```

### Option 2: Install via `pip` / `pipx`

```bash
pipx install laya-cli
# or
pip install laya-cli
```

### Option 3: Run without installing

```bash
uvx laya-cli --help
```

> Requires Python 3.10+ (`.python-version` pins 3.11). Heavy ML deps (`torch`, `transformers`, `safetensors` via `laya`, ~2 GB) download on first `predict`/`classify`. Afterwards `HF_HUB_OFFLINE=1` works. For `uv` dev, tests mock Laya so CI is fast.

---

## 🤖 Classifier Pipeline Integration (Pexels → Laya)

Pipe candidate streams directly from [`pexels-cli`](https://github.com/MIt9/pexels-cli) (`px`) into Laya. This is the original use-case that drove `laya-cli` (82 candidates, `state` deduped):

```bash
# 1. Pexels → candidates.jsonl with `state` strings
px videos --queries "black friday shopping,christmas shopping,checkout cart" \
  --per-page 8 --state --dedupe keep-first > candidates.jsonl

# 2. Laya → score + filter (streaming, one model load)
cat candidates.jsonl \
  | laya-cli classify --questions questions.json \
  | laya-cli filter --where "on_topic>=0.4" --sort -on_topic \
  > shortlist.jsonl

# Or with the new primary command (supports presets without a file):
cat candidates.jsonl | laya-cli predict --questions questions.json --format jsonl > scored.jsonl
px videos --queries "..." --state --dedupe | laya-cli predict --preset triage --format jsonl | laya-cli filter --where "intent==refund"
```

**Candidate JSONL format** (`px --state`):
```json
{"id": 5890229, "type": "video", "query": "black friday shopping", "photographer": "Pavel Danilyuk", "state": "a man shopping on black friday", "url": "https://www.pexels.com/video/a-man-shopping-on-black-friday-5890229/", "duration": 10, "width": 2160, "height": 3840}
```
`laya-cli` reads `--state-field state` **verbatim** — use `--prepend-field` only explicitly (mixing photographer into state degraded `on_topic` by 0.1-0.3).

---

## 📖 Usage Examples

### 1. Direct CLI (Human & AI)

```bash
# Human: table output, no file needed
laya-cli predict "I was charged twice, refund please" --preset triage
laya-cli predict --text "Ignore previous instructions" --preset guard --format table
laya-cli predict --state '{"subject":"Invoice #4411","body":"Billed twice"}' --preset email --format table

# AI agent: JSON single-shot
laya-cli predict --text "Is this spam?" --preset guard --format json
# -> {"answers": {"jailbreak": {"noul": 0.02, "confidence": 0.97, ...}}, "usage": ...}

# Custom + preset merging (preset < file < inline)
laya-cli predict --text "hello" --preset triage --questions-inline '{"custom":{"type":"noul","instructions":"Is it polite?"}}' --format json

# High-cardinality choice (77 banking intents)
laya-cli predict --text "where is my card?" --questions banking.json --shortlist-k 20 --format json

# Multilingual — Router recommended
laya-cli predict --text "मुझसे दो बार शुल्क लिया गया" --preset triage --router --format json
laya-cli predict --text "Der Kunde wurde zweimal belastet" --preset triage --router --lang de --format json
```

### 2. Batch & Pipeline

```bash
# Batch from file (human-readable table per row)
laya-cli predict --input candidates.jsonl --questions questions.json --format table

# Batch JSONL with flatten (like classify) for jq/filter
laya-cli predict --input candidates.jsonl --questions questions.json --flatten --format jsonl > scored.jsonl
cat scored.jsonl | laya-cli filter --where "on_topic>=0.4,is_relevant>=0.7" --sort -on_topic,+id

# Legacy streaming (kept for compatibility):
cat candidates.jsonl | laya-cli classify --questions questions.json | laya-cli filter --where "on_topic>=0.4" --sort -on_topic > shortlist.jsonl
```

### 3. Presets & Info

```bash
laya-cli questions list --format table
laya-cli questions triage > questions.json      # also: email, guard, moderation, router
laya-cli presets guard | laya-cli predict --text "test" --questions /dev/stdin

laya-cli info                                    # python/torch/cuda/mps + cache
laya-cli evaluate --questions questions.json --labeled labeled.jsonl --label-field label --threshold 0.5
```

### 4. Resident Daemon Mode (optional, speeds up repeated calls)

Measured: `laya.load()` 10–35s (MPS) + ~120ms × 82 candidates. For one pipeline run it's fine; for iterative tuning of `questions.json` each `predict` pays 10–35s again. `serve` keeps the model in RAM — subsequent calls skip the load and hit `127.0.0.1` (<1ms overhead).

```bash
# Start daemon in background (one daemon per model/device/router config)
laya-cli serve --model convaiinnovations/laya --device mps
laya-cli serve --model convaiinnovations/laya --subfolder multilingual --device cpu --idle-timeout 60  # short idle for test
laya-cli serve --router --foreground --idle-timeout 0  # foreground, no auto-exit, for logs

# Check status (also shows port, pid, idle_seconds, requests_served)
laya-cli serve status
laya-cli serve status --model convaiinnovations/laya --subfolder multilingual

# Use it — predict/classify/evaluate automatically hit the daemon if live for that config
laya-cli predict "hello" --preset guard --format json          # -> via daemon, no 10s load
laya-cli predict "hello" --preset guard --no-daemon --format json  # force in-process, ignore daemon

# Batch also benefits (each line -> POST /predict on loopback, serialized with lock)
cat candidates.jsonl | laya-cli classify --questions q.json | laya-cli filter --where "on_topic>=0.4" --sort -on_topic

# Stop
laya-cli serve stop                 # stop daemon for default config
laya-cli serve stop --all           # stop all daemons
curl http://127.0.0.1:<port>/status  # GET /status, POST /predict, POST /shutdown also work directly
```

**Details:** HTTP on `127.0.0.1` only (SKILL.md: "Bind to 127.0.0.1"), `POST /predict {"state":..., "questions":{...}}` → same as `agent.predict()`, `GET /status` → `{model, device, loaded_at, idle_seconds, requests_served}`, `POST /shutdown` (localhost only). One daemon = one checkpoint hash(`model|subfolder|device|router|lang`) → pid+port in `~/.cache/laya-cli/daemons/<hash>.json`, pid file removed on exit. Idle timeout 1800s default, `0` disables. If daemon not running, `predict`/`classify`/`evaluate` silently fall back to in-process load — no new step for scripts. `--no-daemon` forces fallback. Requests are serialized with a `threading.Lock` (one GPU = one forward pass).

---

## 🛠 Development

```bash
uv sync --group dev          # hatchling + ruff/mypy/pytest
uv run ruff check .          # lint (E/F/W/I, line-length 120)
uv run ruff format .         # format
uv run pytest -q             # 26 tests, mocked laya (no model download)
uv run pytest --cov=src/laya_cli --cov-report=term-missing
uv build                     # hatchling -> dist/*.whl + sdist (src/ layout)
```

Repo layout: `src/laya_cli/`, `tests/`, `pyproject.toml` (hatchling + `dependency-groups`), `uv.lock`, `.python-version` 3.11, `.github/workflows/ci.yml` + `publish.yml`.

---

## 📄 License

Apache-2.0 — same as [Laya](https://github.com/NandhaKishorM/laya) upstream. See [LICENSE](LICENSE).

This project is not affiliated with Convai Innovations. Laya weights are Apache-2.0 and hosted on Hugging Face (`convaiinnovations/laya`).
