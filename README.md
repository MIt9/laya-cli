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
* ⚡ **Resident Daemon (optional)**: `laya-cli serve` keeps model in RAM — `predict`/`classify`/`evaluate` auto-hit `127.0.0.1` and skip 10-35s `laya.load()` on repeated calls (hash → `~/.cache/laya-cli/daemons/<hash>.json`, idle-timeout 1800s)

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

**Виміряний факт (сесія 2026-09-22, 82 кандидати, MPS):**

| етап | час |
|------|-----|
| `px` (мережа, 12 запитів) | 9.3s |
| `laya.load()` (модель у пам'ять) | 10–35s |
| inference, 82 кандидати | ~10s (~120ms/шт) |
| `filter` | 0.08s |

Для одного пайплайну `px → classify → filter` 35–50s норм. Проблема — коли за сесію кілька разів викликаєш `predict`/`classify`/`evaluate` (підбір `questions.json`: прогнав → подивився → поправив → знову), кожен раз платиш 10–35s за ту саму модель. `laya-cli serve` тримає модель в RAM, наступні виклики летять на `127.0.0.1` (<1ms overhead).

#### Транспорт (як у `~/.claude/skills/laya-integration/SKILL.md` "Anything else... HTTP sidecar")

HTTP на loopback `127.0.0.1` (не `0.0.0.0`), один потік з `threading.Lock` (SKILL.md: "one GPU serves one forward pass at a time"):

- `POST /predict` → `{"state": ..., "questions": {...}, "lang": "...", "shortlist_k": 20}` → те саме що `agent.predict()` (той самий JSON що `laya-cli predict --format json`)
- `GET /status` → `{"model": "...", "subfolder": "...", "device": "...", "loaded_at": 123..., "idle_seconds": 42, "requests_served": 17, "pid": 12345, "port": 8765}`
- `POST /shutdown` → graceful stop (лише з localhost, також `serve stop` шле сигнал за pid-файлом)

#### Lifecycle

```bash
# Старт (без --foreground — форк у фон, пише pid+port в ~/.cache/laya-cli/daemons/<hash>.json)
laya-cli serve --model convaiinnovations/laya --device mps
laya-cli serve --model convaiinnovations/laya --subfolder multilingual --device cpu --idle-timeout 60
laya-cli serve --router --foreground --idle-timeout 0 --port 8765  # форграунд для логів, 0=disabled idle

# Один daemon = один конфіг (hash(model|subfolder|device|router|lang)), інший конфіг — окремий файл/порт
laya-cli serve status                                          # дефолтний конфіг (як у predict без прапорців)
laya-cli serve status --model convaiinnovations/laya --subfolder multilingual --device cpu
laya-cli serve stop                                            # graceful (POST /shutdown → pid файл видаляється)
laya-cli serve stop --all                                      # всі daemon-и
# Також: curl http://127.0.0.1:<port>/status
```

Перед прийомом запитів — прогрів throwaway `predict` (як у `TASK.md`). `idle-timeout 1800s` дефолт, `--idle-timeout 5` для тесту → daemon сам виходить через 5s без запитів і `serve status` каже `not running`.

#### Клієнт (`predict`/`classify`/`evaluate`)

Перед `laya.load()` перевіряє `~/.cache/laya-cli/daemons/<hash>.json` для поточного конфігу; якщо файл є і daemon відповідає на `/status` — шле туди, інакше мовчки падає назад на in-process (поведінка `v1` без змін). `--no-daemon` форсує in-process (для відтворюваності/дебагу). Для батчу (`classify`/`predict --input`) — кожен рядок окремий `POST /predict` в циклі (loopback мілісекунди, batch-ендпоінт не потрібен).

```bash
# AI workflow (повністю автоматичний):
laya-cli serve --device mps &                          # один раз на сесію
laya-cli predict "hello" --preset guard --format json  # -> via daemon, без 10s
laya-cli predict "hello2" --preset guard --format json # -> знову via daemon
laya-cli serve status                                  # {"loaded_at":..., "requests_served": 2}
laya-cli predict "hello" --preset guard --no-daemon    # форс in-process (знову 10s, для ізоляції)
laya-cli serve stop

# Human tuning loop (типовий):
laya-cli serve &                                       # фон
cat candidates.jsonl | laya-cli classify --questions q.json | laya-cli filter --where "on_topic>=0.4" --sort -on_topic  # via daemon
# ...поправив q.json...
cat candidates.jsonl | laya-cli classify --questions q.json | laya-cli filter ...  # знову via daemon, без перезавантаження
laya-cli serve stop

# 5 паралельних predict — не падають, результати не плутаються (серіалізація Lock)
seq 1 5 | xargs -P5 -I{} laya-cli predict "text {}" --preset guard --format json
```

**Безпека/межі:** лише `127.0.0.1`, без auth (однокористувацька машина, як у SKILL.md), один потік, stateless крім моделі.

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
