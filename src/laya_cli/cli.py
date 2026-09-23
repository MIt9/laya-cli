#!/usr/bin/env python3
"""laya-cli — ergonomic CLI for Laya (human + AI-agent friendly).

Primary command is `predict` for direct use:
  laya-cli predict "refund my order" --preset triage
  laya-cli predict --text "is this spam?" --preset guard --format json
  echo '{"state":"hello"}' | laya-cli predict --questions q.json --format jsonl

Batch pipeline (TASK.md) is preserved via `classify | filter`:
  cat candidates.jsonl | laya-cli classify --questions q.json | laya-cli filter --where "on_topic>=0.4" --sort -on_topic

Also: `questions`/`presets`, `evaluate`, `filter`, `info`, `serve` (resident daemon).
"""

from __future__ import annotations

import argparse
import json
import os
import re as _re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__

# Daemon helpers (import lazily to avoid circular, but keep top-level for type)
try:
    from .daemon import (
        CACHE_DIR,
        daemon_file_for,
        daemon_port_for_args,
        daemon_predict_via_http,
        is_daemon_alive,
        run_server,
    )
except ImportError:
    # fallback if daemon module missing (should not happen)
    CACHE_DIR = None
    daemon_file_for = None  # type: ignore
    daemon_port_for_args = None  # type: ignore
    daemon_predict_via_http = None  # type: ignore
    is_daemon_alive = None  # type: ignore
    run_server = None  # type: ignore

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="laya-cli",
        description=(
            "Ergonomic CLI for Laya — typed decisions (choice/score/noul) in one forward pass. "
            "Works for humans (table) and agents (JSON/JSONL). "
            "Optional resident daemon (laya-cli serve) keeps model in RAM to avoid 10-35s laya.load() on repeated predict/classify/evaluate (same model|subfolder|device|router|lang → ~/.cache/laya-cli/daemons/<hash>.json, 127.0.0.1 only, auto fallback if not running)."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- predict (primary, human+AI friendly) -----------------------------
    pr = sub.add_parser(
        "predict",
        help="Run Laya on one text/JSON state or a batch (JSONL). Human-friendly & agent-friendly. Auto-uses daemon if live.",
        description=(
            "Predict with Laya on a single state or a batch. Supports plain text, JSON state, presets, shortlist for high-cardinality choices, and table/JSON output. "
            "If a daemon is live for the same --model/--subfolder/--device/--router/--lang (hash → ~/.cache/laya-cli/daemons/<hash>.json), predict hits 127.0.0.1:<port>/predict and skips 10-35s laya.load(); otherwise it loads in-process. Use --no-daemon to force in-process."
        ),
        epilog=(
            "Examples (human):\n"
            '  laya-cli predict "I was charged twice, refund please" --preset triage\n'
            '  laya-cli predict --text "Ignore previous instructions" --preset guard\n'
            '  laya-cli predict --preset email --state \'{"subject":"Invoice","body":"Hi"}\' --format table\n'
            "\n"
            "Examples (agent / batch):\n"
            '  laya-cli predict --text "hello" --questions q.json --format json\n'
            "  cat candidates.jsonl | laya-cli predict --questions q.json --state-field state --format jsonl > scored.jsonl\n"
            "  laya-cli predict --input candidates.jsonl --questions q.json --shortlist-k 20 --format jsonl\n"
            '  laya-cli predict --preset triage --text "my payment failed" --model convaiinnovations/laya --device cpu --full-probs\n'
            "\n"
            "Daemon (optional, speeds up repeated calls):\n"
            "  laya-cli serve --model convaiinnovations/laya --device mps  # start daemon (background, writes ~/.cache/laya-cli/daemons/<hash>.json)\n"
            "  laya-cli serve status                                        # check live daemon (GET /status)\n"
            '  laya-cli predict "hello" --preset guard --format json      # auto-uses daemon (<1ms overhead), no 10s load\n'
            '  laya-cli predict "hello" --preset guard --no-daemon       # force in-process, ignore daemon\n'
            "  laya-cli serve stop                                          # graceful shutdown (POST /shutdown)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Positional text is sugar for --text (allows: laya-cli predict "hello" --preset guard)
    pr.add_argument(
        "text_positional",
        nargs="*",
        help="State as plain text (shorthand for --text). If multiple, each is a separate prediction.",
    )
    pr.add_argument(
        "--text",
        dest="text_opt",
        action="append",
        default=None,
        help="State as plain text (can repeat).",
    )
    pr.add_argument(
        "--state",
        dest="state_json",
        default=None,
        help='State as JSON string (e.g. \'{"subject":"Hi","body":"..."}\')',
    )
    pr.add_argument("--state-file", dest="state_file", default=None, help="State from JSON file")
    pr.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Batch input: JSONL file (each line JSON) or - for stdin. Each object’s field --state-field is used as state.",
    )
    pr.add_argument(
        "--state-field",
        default="state",
        help="For --input batch: field holding state text (default: state). Taken verbatim, no concatenation.",
    )
    pr.add_argument(
        "--questions",
        default=None,
        help="Path to questions.json (dict of {id: {type, instructions, criteria}})",
    )
    pr.add_argument(
        "--questions-inline",
        dest="questions_inline",
        default=None,
        help="Questions as JSON string (merges with --questions/--preset)",
    )
    pr.add_argument(
        "--preset",
        default=None,
        help="Built-in preset: triage | email | guard | moderation | router (can combine with --questions)",
    )
    pr.add_argument(
        "--model",
        default="convaiinnovations/laya",
        help="HF repo id (default: convaiinnovations/laya)",
    )
    pr.add_argument(
        "--subfolder",
        default=None,
        help="Subfolder checkpoint: multilingual / typed-decisions / '' (bundle root). Only that subfolder is downloaded.",
    )
    pr.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Device to force (default: auto: cuda > mps > cpu)",
    )
    pr.add_argument(
        "--router",
        action="store_true",
        help="Use laya.Router(preload=True) — recommended for multilingual (auto-detects script/language per request).",
    )
    pr.add_argument(
        "--lang",
        default=None,
        help="Force language for Router (e.g. en, fr, de). Without --lang Router only recognises {en,fr,de,es,pt,it,nl} among Latin scripts.",
    )
    pr.add_argument(
        "--shortlist-k",
        dest="shortlist_k",
        type=int,
        default=None,
        help="Enable shortlist for high-cardinality choice: keep top K labels via embeddings (uses agent encoder via embed_fn_from_agent).",
    )
    pr.add_argument(
        "--full-probs",
        action="store_true",
        help="Include full probability distributions in flattened output",
    )
    pr.add_argument(
        "--format",
        dest="out_format",
        choices=["json", "jsonl", "table"],
        default=None,
        help="Output format: json (single object), jsonl (one per line), table (human). Default: json for single, jsonl for batch.",
    )
    pr.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    pr.add_argument(
        "--flatten",
        action="store_true",
        help="For batch JSONL: flatten answers into top-level fields like classify does ({id}, {id}_p, {id}_confidence) instead of nested answers.",
    )
    pr.add_argument(
        "--prepend-field",
        default=None,
        help="For batch: prepend field to state text as '<value> <state>'. Disabled by default (mixing metadata degraded on_topic scores 0.1-0.3).",
    )
    pr.add_argument("--output", default=None, help="Output file (default: stdout)")
    pr.add_argument(
        "--no-daemon",
        action="store_true",
        help="Force in-process load, ignore resident daemon even if live (for reproducibility/debug).",
    )

    # ---- classify (legacy batch, kept for TASK.md compat) -----------------
    c = sub.add_parser(
        "classify",
        help="Batch classify JSONL from stdin via Laya (legacy, use predict for new code). Auto-uses daemon if live.",
        description=(
            "Classify JSONL from stdin via Laya; append flattened answers. Kept for pipeline compatibility (px | laya-cli classify). "
            "For new code prefer `predict`. If a daemon is live for the same config (hash → ~/.cache/laya-cli/daemons/<hash>.json), classify hits 127.0.0.1 and skips load; else in-process. Use --no-daemon to force."
        ),
        epilog=(
            "Examples:\n"
            "  cat candidates.jsonl | laya-cli classify --questions questions.json > scored.jsonl\n"
            '  px videos --queries "..." --state --dedupe keep-first \\\n'
            "    | laya-cli classify --questions questions.json \\\n"
            '    | laya-cli filter --where "on_topic>=0.4" --sort -on_topic > shortlist.jsonl\n'
            "  # with daemon (start once, then all classify hit daemon):\n"
            "  laya-cli serve --model convaiinnovations/laya & laya-cli classify --questions q.json < candidates.jsonl\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    c.add_argument("--questions", required=True, help="Path to questions.json")
    c.add_argument(
        "--state-field",
        default="state",
        help="Field name holding the state text (default: state). Taken verbatim.",
    )
    c.add_argument(
        "--model",
        default="convaiinnovations/laya",
        help="HF repo id (default: convaiinnovations/laya)",
    )
    c.add_argument(
        "--subfolder",
        default=None,
        help="Subfolder checkpoint: multilingual / typed-decisions / ''",
    )
    c.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"], help="Device to force")
    c.add_argument("--router", action="store_true", help="Use laya.Router(preload=True)")
    c.add_argument("--lang", default=None, help="Force language for Router")
    c.add_argument("--prepend-field", default=None, help="Optional field to prepend to state text")
    c.add_argument("--full-probs", action="store_true", help="Include full distribution ({id}_probs)")
    c.add_argument("--no-daemon", action="store_true", help="Force in-process load, ignore daemon")

    # ---- filter -----------------------------------------------------------
    f = sub.add_parser(
        "filter",
        help="Lightweight post-filter/sorter over JSONL.",
        description="Filter and sort JSONL from stdin. Replaces per-case build_shortlist*.py.",
        epilog=(
            "Examples:\n"
            '  cat scored.jsonl | laya-cli filter --where "on_topic>=0.4" --sort -on_topic\n'
            '  cat scored.jsonl | laya-cli filter --where "on_topic>=0.4,is_spam!=1" --sort -on_topic,+id\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    f.add_argument(
        "--where",
        default=None,
        help='Filter expression(s), comma = AND. Ops: == != = > < >= <=. Example: "on_topic>=0.4"',
    )
    f.add_argument(
        "--sort",
        default=None,
        help='Sort keys, comma-separated, "-" prefix = descending. Example: "-on_topic,+id"',
    )

    # ---- questions / presets ----------------------------------------------
    q = sub.add_parser(
        "questions",
        aliases=["presets"],
        help="Show or list built-in preset questions.json.",
        description="Presets are direct passthroughs of laya.*_questions() — triage, email, guard, moderation, router.",
        epilog=(
            "Examples:\n"
            "  laya-cli questions list\n"
            "  laya-cli questions triage > questions.json\n"
            '  laya-cli predict --preset triage --text "my payment failed"  # use without file\n'
            "  laya-cli questions guard | laya-cli predict --input /dev/stdin --state-field prompt  # pipe\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    q.add_argument(
        "preset",
        nargs="?",
        default=None,
        help="Preset name: triage | email | guard | moderation | router | list",
    )
    q.add_argument(
        "--format",
        dest="q_format",
        choices=["json", "table"],
        default="json",
        help="Output format (default: json)",
    )

    # ---- evaluate ---------------------------------------------------------
    e = sub.add_parser(
        "evaluate",
        help="Evaluate on a labelled JSONL set and print accuracy / threshold report. Auto-uses daemon if live.",
        description=(
            "Evaluate before shipping: run classify on a labelled set and report accuracy per question, pass rate at threshold, and precision @ threshold. "
            "If a daemon is live for the same --model/--device/--router, evaluate hits it and skips load; else in-process. Use --no-daemon to force."
        ),
        epilog=(
            "Examples:\n"
            "  laya-cli evaluate --questions questions.json --labeled labeled.jsonl --label-field label --threshold 0.5\n"
            "  laya-cli evaluate --questions q.json --labeled dev.jsonl --field on_topic --state-field state\n"
            "  laya-cli serve & laya-cli evaluate --questions q.json --labeled dev.jsonl  # via daemon\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    e.add_argument("--questions", required=True, help="Path to questions.json")
    e.add_argument(
        "--labeled",
        required=True,
        help="Path to labelled JSONL (each line: {state_field: text, label_field: true_label})",
    )
    e.add_argument("--state-field", default="state", help="Field holding the state text (default: state)")
    e.add_argument(
        "--label-field",
        default="label",
        help="Field holding the ground-truth label (default: label)",
    )
    e.add_argument("--field", default=None, help="Optional single question id to evaluate (default: all)")
    e.add_argument("--threshold", type=float, default=0.5, help="Threshold for pass/precision (default: 0.5)")
    e.add_argument("--model", default="convaiinnovations/laya", help="HF repo id")
    e.add_argument("--subfolder", default=None, help="Subfolder checkpoint")
    e.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"], help="Device to force")
    e.add_argument("--router", action="store_true", help="Use laya.Router(preload=True)")
    e.add_argument("--lang", default=None, help="Force language for Router")
    e.add_argument("--prepend-field", default=None, help="Optional field to prepend to state text")
    e.add_argument("--no-daemon", action="store_true", help="Force in-process load, ignore daemon")

    # ---- serve (resident daemon) ------------------------------------------
    s = sub.add_parser(
        "serve",
        help="Resident daemon that keeps model in memory (optional, speeds up repeated predict).",
        description=(
            "Start a resident daemon that holds the Laya model in RAM (10-35s laya.load() once). "
            "Subsequent `predict`/`classify`/`evaluate` with the SAME config (--model/--subfolder/--device/--router/--lang → hash → ~/.cache/laya-cli/daemons/<hash>.json) "
            "automatically hit 127.0.0.1:<port>/predict and skip the load (<1ms overhead). If no daemon, they fall back to in-process load — no pipeline change. "
            "Daemon binds only 127.0.0.1 (SKILL.md: Bind to 127.0.0.1), single-threaded lock (one GPU = one forward pass), idle-timeout auto-exit (default 1800s, 0=disabled), warmup throwaway predict on start."
        ),
        epilog=(
            "Workflow for AI/human (typical tuning loop):\n"
            "  laya-cli serve --model convaiinnovations/laya --device mps         # start once (background, writes ~/.cache/laya-cli/daemons/<hash>.json)\n"
            "  laya-cli serve status                                              # GET /status → {model,device,loaded_at,idle_seconds,requests_served,pid,port}\n"
            '  laya-cli predict "test" --preset triage --format json             # auto via daemon (no 10s load)\n'
            '  laya-cli predict "test2" --preset triage --format json            # still via daemon\n'
            '  laya-cli predict "test" --preset triage --no-daemon --format json # force in-process (ignore daemon)\n'
            "  cat candidates.jsonl | laya-cli classify --questions q.json        # batch: each line POST /predict (serialized)\n"
            "  laya-cli serve stop                                                # POST /shutdown or SIGTERM, removes pid file\n"
            "  laya-cli serve stop --all                                          # stop all configs\n"
            "\n"
            "Other examples:\n"
            "  laya-cli serve --foreground --idle-timeout 5 --port 0              # foreground for logs, 5s idle test\n"
            "  laya-cli serve --model convaiinnovations/laya --subfolder multilingual --device cpu\n"
            "  laya-cli serve status --model convaiinnovations/laya --subfolder multilingual  # check that config\n"
            '  curl http://127.0.0.1:<port>/status ; curl -X POST http://127.0.0.1:<port>/predict -d \'{"state":"hi","questions":{...}}\'\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # serve can be started with no subcommand, or with status/stop
    s.add_argument("--model", default="convaiinnovations/laya", help="HF repo id (default: convaiinnovations/laya)")
    s.add_argument("--subfolder", default=None, help="Subfolder: multilingual / typed-decisions / ''")
    s.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"], help="Device to force")
    s.add_argument("--router", action="store_true", help="Use laya.Router(preload=True)")
    s.add_argument("--lang", default=None, help="Force language for Router")
    s.add_argument("--port", type=int, default=0, help="Port to bind (0 = random free, default 0)")
    s.add_argument(
        "--idle-timeout",
        type=int,
        default=1800,
        help="Idle seconds before daemon auto-exits (0 = disabled, default 1800)",
    )
    s.add_argument("--foreground", action="store_true", help="Run in foreground, don't daemonize (for logging/debug)")
    # To support `serve status` / `serve stop` as subcommands, we add optional positional
    s.add_argument(
        "action",
        nargs="?",
        choices=["status", "stop"],
        help="Action: status (show) or stop (shutdown). No action = start daemon.",
    )
    s.add_argument(
        "--all", dest="all_daemons", action="store_true", help="For stop: stop all daemons regardless of config"
    )

    # ---- info -------------------------------------------------------------
    inf = sub.add_parser(
        "info",
        help="Show model, device and checkpoint info.",
        description="Show environment, available devices, and (if cached) checkpoint config.",
    )
    inf.add_argument("--model", default="convaiinnovations/laya", help="HF repo id to inspect")
    inf.add_argument("--subfolder", default=None, help="Subfolder to inspect")

    return p


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_questions(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        print(f"error: questions file {path!r} must be a non-empty dict", file=sys.stderr)
        sys.exit(2)
    for qid, qdef in data.items():
        if not isinstance(qdef, dict) or "type" not in qdef:
            print(f"error: question {qid!r} must have a 'type' field", file=sys.stderr)
            sys.exit(2)
        if qdef["type"] not in ("choice", "score", "noul"):
            print(f"error: question {qid!r} has unknown type {qdef['type']!r}", file=sys.stderr)
            sys.exit(2)
    return data


def _load_preset(name: str) -> dict[str, Any]:
    mapping = {
        "triage": "triage_questions",
        "email": "email_questions",
        "guard": "guard_questions",
        "moderation": "moderation_questions",
        "router": "router_questions",
    }
    key = name.strip().lower()
    if key not in mapping:
        print(
            f"error: unknown preset {name!r}; choose from {', '.join(sorted(mapping))}",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        import laya
    except ImportError as exc:
        print(f"error: laya not installed: {exc}", file=sys.stderr)
        sys.exit(2)
    fn = getattr(laya, mapping[key])
    return fn()


def _load_questions_merged(args) -> dict[str, Any]:
    """Merge --preset, --questions, --questions-inline (preset < file < inline)."""
    merged: dict[str, Any] = {}
    if getattr(args, "preset", None):
        merged.update(_load_preset(args.preset))
    if getattr(args, "questions", None):
        file_q = _load_questions(args.questions)
        merged.update(file_q)
    inline = getattr(args, "questions_inline", None)
    if inline:
        try:
            inline_q = json.loads(inline)
        except json.JSONDecodeError as exc:
            print(f"error: invalid --questions-inline JSON: {exc}", file=sys.stderr)
            sys.exit(2)
        if not isinstance(inline_q, dict):
            print("error: --questions-inline must be a JSON object (dict)", file=sys.stderr)
            sys.exit(2)
        merged.update(inline_q)
    if not merged:
        print(
            "error: no questions provided. Use --preset or --questions or --questions-inline",
            file=sys.stderr,
        )
        sys.exit(2)
    # validate merged
    for qid, qdef in merged.items():
        if not isinstance(qdef, dict) or "type" not in qdef:
            print(f"error: question {qid!r} must have a 'type' field", file=sys.stderr)
            sys.exit(2)
        if qdef["type"] not in ("choice", "score", "noul"):
            print(f"error: question {qid!r} has unknown type {qdef['type']!r}", file=sys.stderr)
            sys.exit(2)
    return merged


def _init_agent(args) -> Any:
    """Load Laya agent or Router exactly once; warm up; emit timing to stderr."""
    t0 = time.time()
    if getattr(args, "router", False):
        if not getattr(args, "lang", None):
            print(
                "[laya-cli] warning: --router is active without --lang. "
                "As of laya 0.3.4 the Router detector only recognises {en,fr,de,es,pt,it,nl} "
                "among Latin-script languages; Polish, Czech, Turkish, Swedish etc. are "
                "silently routed to the English checkpoint and may answer confidently and wrongly. "
                "Pass --lang when you know it.",
                file=sys.stderr,
            )
        try:
            from laya import Router
        except ImportError as exc:
            print(f"error: laya not installed: {exc}", file=sys.stderr)
            sys.exit(2)
        kwargs: dict[str, Any] = {"preload": True}
        if getattr(args, "device", None):
            kwargs["device"] = args.device
        agent = Router(**kwargs)
        try:
            dummy_q = {"_warmup": {"type": "noul", "instructions": "Is this warmup text relevant?"}}
            agent.predict("warmup text for kernel compilation", dummy_q, lang=getattr(args, "lang", None)) if getattr(
                args, "lang", None
            ) else agent.predict("warmup text for kernel compilation", dummy_q)
        except Exception:
            pass
        dt = time.time() - t0
        print(
            f"[laya-cli] Router loaded in {dt:.1f}s (device={getattr(agent, 'device', '?')})",
            file=sys.stderr,
        )
        return agent
    else:
        try:
            import laya
        except ImportError as exc:
            print(f"error: laya not installed: {exc}", file=sys.stderr)
            sys.exit(2)
        kwargs: dict[str, Any] = {}
        if getattr(args, "device", None):
            kwargs["device"] = args.device
        if getattr(args, "subfolder", None):
            kwargs["subfolder"] = args.subfolder
        agent = laya.load(getattr(args, "model", "convaiinnovations/laya"), **kwargs)
        try:
            dummy_q = getattr(args, "_warmup_questions", None)
            if dummy_q is None:
                dummy_q = {"_warmup": {"type": "noul", "instructions": "Is this warmup text relevant?"}}
            agent.predict("warmup text for kernel compilation", dummy_q)
        except Exception:
            pass
        dt = time.time() - t0
        print(
            f"[laya-cli] model {getattr(args, 'model', 'convaiinnovations/laya')!r} loaded in {dt:.1f}s (device={getattr(agent, 'device', '?')})",
            file=sys.stderr,
        )
        return agent


def _apply_answers(row: dict[str, Any], answers: dict[str, Any], questions: dict[str, Any], full_probs: bool) -> None:
    for qid, qdef in questions.items():
        ans = answers.get(qid)
        if ans is None:
            continue
        qtype = qdef.get("type")
        if qtype == "choice":
            row[qid] = ans.get("choice")
            probs = ans.get("probabilities", {})
            chosen = ans.get("choice")
            p = probs.get(chosen) if isinstance(probs, dict) else None
            if p is not None:
                row[f"{qid}_p"] = p
            if full_probs:
                row[f"{qid}_probs"] = probs
            if "confidence" in ans:
                row[f"{qid}_confidence"] = ans["confidence"]
        elif qtype == "score":
            row[f"{qid}_score"] = ans.get("score")
            if "confidence" in ans:
                row[f"{qid}_confidence"] = ans["confidence"]
            if full_probs and "probabilities" in ans:
                row[f"{qid}_probs"] = ans["probabilities"]
        elif qtype == "noul":
            row[qid] = ans.get("noul")
            if "confidence" in ans:
                row[f"{qid}_confidence"] = ans["confidence"]


def _collect_states_predict(args) -> list[tuple[Any, Any]]:
    """Return list of (original_row_or_none, state) for predict."""
    states: list[tuple[Any, Any]] = []

    # Explicit batch input
    if getattr(args, "input_path", None):
        path = args.input_path
        fh = sys.stdin if path == "-" else open(path, encoding="utf-8")
        try:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # treat as raw text line
                    row = {"state": line}
                if not isinstance(row, dict):
                    row = {"state": str(row)}
                # If requested state-field is missing, fall back to using the whole row as state (dict)
                # This matches `laya-cli predict --state '{"posting":"..."}'` which sends the dict.
                # Without this, batch with `{"posting":"..."}` and default --state-field state would send "" for every row.
                if args.state_field not in row:
                    # Only warn when using default field and row has other keys
                    if args.state_field == "state":
                        print(
                            f"[laya-cli] warning: field 'state' not in row {list(row.keys())}, using entire row as state. Use --state-field to specify.",
                            file=sys.stderr,
                        )
                    state: Any = row
                    states.append((row, state))
                    continue
                raw = row.get(args.state_field, "")
                if raw is None:
                    raw = ""
                if isinstance(raw, (dict, list)):
                    state_text = json.dumps(raw, ensure_ascii=False)
                    state: Any = raw  # keep dict/list for Laya (it serializes), but also keep string for backwards compat
                    # For Laya, passing the raw dict/list is better than json string; use raw
                    state = raw
                else:
                    state_text = str(raw)
                    state = state_text
                if getattr(args, "prepend_field", None):
                    extra = row.get(args.prepend_field)
                    if extra is not None and str(extra).strip() != "":
                        extra_s = (
                            json.dumps(extra, ensure_ascii=False) if isinstance(extra, (dict, list)) else str(extra)
                        )
                        # Need state_text for prepend, so ensure we have it
                        if not isinstance(raw, (dict, list)):
                            state_text = str(raw)
                        else:
                            state_text = json.dumps(raw, ensure_ascii=False)
                        state_text = f"{extra_s} {state_text}"
                        state = state_text
                # keep original row for output
                states.append((row, state))
        finally:
            if fh is not sys.stdin:
                fh.close()
        return states

    # Positional + --text
    texts: list[str] = []
    if getattr(args, "text_positional", None):
        texts.extend([t for t in args.text_positional if t is not None and str(t) != ""])
    if getattr(args, "text_opt", None):
        # text_opt is list of strings (action append)
        for t in args.text_opt:
            if t is not None:
                texts.append(t)
    if texts:
        for t in texts:
            states.append((None, t))
        return states

    # --state JSON string
    if getattr(args, "state_json", None):
        try:
            parsed = json.loads(args.state_json)
            states.append((None, parsed))
        except json.JSONDecodeError:
            states.append((None, args.state_json))
        return states

    if getattr(args, "state_file", None):
        with open(args.state_file, encoding="utf-8") as f:
            content = f.read().strip()
            try:
                parsed = json.loads(content)
                states.append((None, parsed))
            except json.JSONDecodeError:
                states.append((None, content))
        return states

    # Fallback: stdin as single state (human pipe) or batch if JSONL
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if not data.strip():
            return states
        # Try JSONL first: multiple lines JSON?
        lines = [line for line in data.splitlines() if line.strip() != ""]
        # If single line and it looks like JSON object, treat as state JSON
        if len(lines) == 1:
            try:
                parsed = json.loads(lines[0])
                if isinstance(parsed, dict) and args.state_field in parsed:
                    raw = parsed.get(args.state_field, "")
                    state_text = json.dumps(raw, ensure_ascii=False) if isinstance(raw, (dict, list)) else str(raw)
                    states.append((parsed, state_text))
                elif isinstance(parsed, dict):
                    # use whole object as state (Laya supports dict state)
                    states.append((None, parsed))
                else:
                    states.append((None, parsed))
                return states
            except json.JSONDecodeError:
                states.append((None, lines[0]))
                return states
        # Multiple lines: try batch JSONL
        has_json = False
        for line in lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    has_json = True
                    if args.state_field in row:
                        raw = row.get(args.state_field, "")
                        if isinstance(raw, (dict, list)):
                            st: Any = raw
                        else:
                            st = str(raw)
                            # handle prepend for stdin batch as well
                            if getattr(args, "prepend_field", None):
                                extra = row.get(args.prepend_field)
                                if extra is not None and str(extra).strip() != "":
                                    extra_s = json.dumps(extra, ensure_ascii=False) if isinstance(extra, (dict, list)) else str(extra)
                                    st = f"{extra_s} {st}"
                        states.append((row, st))
                    else:
                        # state field missing -> use whole row as state (dict), like --state '{"posting":...}'
                        if args.state_field == "state":
                            print(
                                f"[laya-cli] warning: field 'state' not in row {list(row.keys())}, using entire row as state",
                                file=sys.stderr,
                            )
                        states.append((row, row))
                else:
                    states.append((None, str(row)))
            except json.JSONDecodeError:
                states.append((None, line))
        if has_json:
            return states
        # Fallback: treat whole stdin as single text
        if not states:
            states.append((None, data.strip()))
        return states

    return states


def _write_output(data: Any, args, is_batch: bool = False) -> None:
    out_path = getattr(args, "output", None)
    fh = open(out_path, "w", encoding="utf-8") if out_path else sys.stdout
    try:
        fmt = getattr(args, "out_format", None)
        if fmt is None:
            fmt = "jsonl" if is_batch else "json"
        pretty = getattr(args, "pretty", False)
        if fmt == "table":
            # table is only for single currently; for batch we output per-row table sections
            if is_batch:
                for item in data:
                    _print_table_single(item, fh)
                    fh.write("\n")
            else:
                _print_table_single(data, fh)
        elif fmt == "jsonl":
            if is_batch:
                for item in data:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:  # json
            if is_batch:
                # when batch but format json -> output JSON array
                json.dump(data, fh, ensure_ascii=False, indent=2 if pretty else None)
                fh.write("\n")
            else:
                json.dump(data, fh, ensure_ascii=False, indent=2 if pretty else None)
                fh.write("\n")
    finally:
        if fh is not sys.stdout:
            fh.close()
        else:
            fh.flush()


def _print_table_single(result: dict[str, Any], fh) -> None:
    answers = result.get("answers", {})
    routing = result.get("routing")
    usage = result.get("usage", {})
    fh.write("Answers:\n")
    # header
    fh.write(f"  {'question':<18} {'type':<7} {'answer':<22} {'confidence':<10} {'details'}\n")
    fh.write("  " + "-" * 80 + "\n")
    for qid, ans in answers.items():
        qtype = ans.get("type", "")
        if qtype == "choice":
            detail = f"{ans.get('choice')}  p={ans.get('probabilities', {}).get(ans.get('choice'), '')}"
            answer = str(ans.get("choice"))
        elif qtype == "score":
            answer = str(ans.get("score"))
            detail = f"probs={ans.get('probabilities')}"
        elif qtype == "noul":
            answer = str(ans.get("noul"))
            detail = f"P(true)={ans.get('noul')}"
        else:
            answer = str(ans)
            detail = ""
        fh.write(f"  {qid:<18} {qtype:<7} {answer:<22} {ans.get('confidence', ''):<10} {detail}\n")
        # show full probs on next line if choice
        if qtype == "choice" and "probabilities" in ans:
            fh.write(f"    probs: {json.dumps(ans['probabilities'], ensure_ascii=False)}\n")
    if routing:
        fh.write(f"\nRouting: {routing.get('model')} — {routing.get('reason')}\n")
    if usage:
        fh.write(f"Usage: {usage}\n")


# ---------------------------------------------------------------------------
# Command impls
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> None:
    questions = _load_questions_merged(args)
    args._warmup_questions = questions

    # Try daemon first (unless --no-daemon)
    daemon_port = None
    agent = None
    if not getattr(args, "no_daemon", False) and daemon_port_for_args is not None:
        try:
            daemon_port = daemon_port_for_args(args)  # type: ignore
        except Exception:
            daemon_port = None
    if daemon_port:
        print(f"[laya-cli] using daemon 127.0.0.1:{daemon_port} (no 10-35s load)", file=sys.stderr)
    else:
        agent = _init_agent(args)

    states = _collect_states_predict(args)
    if not states:
        print(
            "error: no state provided. Use TEXT positional, --text, --state, --state-file or --input / stdin",
            file=sys.stderr,
        )
        sys.exit(2)

    # Decide output mode
    is_batch = len(states) > 1 or getattr(args, "input_path", None) is not None
    # If --input was used, we already know it's batch
    # For predict via stdin single JSONL line, we treat as single vs batch based on count

    results: list[dict[str, Any]] = []
    for orig, state in states:
        # state can be str/dict/list — pass as-is to Laya
        result: dict[str, Any] | None = None
        # Try daemon first if available
        if daemon_port is not None and daemon_predict_via_http is not None:  # type: ignore
            try:
                result = daemon_predict_via_http(  # type: ignore
                    daemon_port,
                    state,
                    questions,
                    lang=getattr(args, "lang", None),
                    shortlist_k=getattr(args, "shortlist_k", None),
                )
                if result is None:
                    raise RuntimeError("daemon returned no result")
            except Exception as exc:
                print(f"[laya-cli] daemon request failed ({exc}), falling back to in-process", file=sys.stderr)
                result = None
                # fallback: load agent if not yet loaded
                if agent is None:
                    agent = _init_agent(args)
                    daemon_port = None  # don't try daemon again for remaining states? keep trying? disable for rest
                # will fallback below
        if result is None:
            try:
                if agent is None:
                    agent = _init_agent(args)
                if getattr(args, "shortlist_k", None) is not None:
                    try:
                        from laya import embed_fn_from_agent, predict_shortlist
                    except ImportError as exc:
                        print(f"error: shortlist requires laya shortlist module: {exc}", file=sys.stderr)
                        sys.exit(2)
                    embed_fn = embed_fn_from_agent(agent)
                    k = int(args.shortlist_k)
                    if getattr(args, "router", False):
                        kwargs = {}
                        if getattr(args, "lang", None):
                            kwargs["lang"] = args.lang
                        result = predict_shortlist(agent, state, questions, embed_fn, k=k, **kwargs)
                    else:
                        result = predict_shortlist(agent, state, questions, embed_fn, k=k)
                else:
                    if getattr(args, "router", False):
                        kwargs = {}
                        if getattr(args, "lang", None):
                            kwargs["lang"] = args.lang
                        result = agent.predict(state, questions, **kwargs)
                    else:
                        result = agent.predict(state, questions)
            except Exception as exc:
                print(
                    f"[laya-cli] error: predict failed for state {str(state)[:80]!r}: {exc}",
                    file=sys.stderr,
                )
                result = {"error": str(exc), "state": state, "answers": {}}

        # For batch with --flatten, mimic classify output
        if is_batch and getattr(args, "flatten", False):
            # orig is original row dict (or None)
            row = dict(orig) if isinstance(orig, dict) else {"state": state} if orig is None else {"state": str(state)}
            # ensure state field preserved
            if isinstance(orig, dict) and args.state_field not in row and isinstance(state, str):
                row[args.state_field] = state
            answers = result.get("answers", {})
            _apply_answers(row, answers, questions, full_probs=getattr(args, "full_probs", False))
            # also keep routing/usage if Router
            if "routing" in result:
                row["_routing"] = result["routing"]
            results.append(row)
        elif is_batch:
            # Batch but not flattened: output per-row result with answers
            out_row = {}
            if isinstance(orig, dict):
                out_row = dict(orig)
            else:
                out_row = {"state": state}
            out_row["answers"] = result.get("answers", {})
            if "routing" in result:
                out_row["routing"] = result["routing"]
            if "usage" in result:
                out_row["usage"] = result["usage"]
            if "shortlist" in result:
                out_row["shortlist"] = result["shortlist"]
            if "error" in result:
                out_row["error"] = result["error"]
            results.append(out_row)
        else:
            # single: result is top-level
            if getattr(args, "flatten", False):
                # flatten single as well
                row = (
                    {"state": state}
                    if not isinstance(state, dict)
                    else dict(state)
                    if isinstance(state, dict)
                    else {"state": str(state)}
                )
                answers = result.get("answers", {})
                _apply_answers(row, answers, questions, full_probs=getattr(args, "full_probs", False))
                results = row  # type: ignore
            else:
                results.append(result)  # type: ignore

    if is_batch:
        _write_output(results, args, is_batch=True)
        print(f"[laya-cli] predict: {len(states)} states", file=sys.stderr)
    else:
        single = results[0] if isinstance(results, list) else results
        _write_output(single, args, is_batch=False)


def cmd_classify(args: argparse.Namespace) -> None:
    questions = _load_questions(args.questions)
    args._warmup_questions = questions
    # Try daemon
    daemon_port = None
    agent = None
    if not getattr(args, "no_daemon", False) and daemon_port_for_args is not None:
        try:
            daemon_port = daemon_port_for_args(args)  # type: ignore
        except Exception:
            daemon_port = None
    if daemon_port:
        print(f"[laya-cli] using daemon 127.0.0.1:{daemon_port}", file=sys.stderr)
    else:
        agent = _init_agent(args)
    n_in = n_out = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[laya-cli] warning: skipping invalid JSON line: {exc}", file=sys.stderr)
            continue
        if not isinstance(row, dict):
            print("[laya-cli] warning: skipping non-object JSON line", file=sys.stderr)
            continue
        raw_state = row.get(args.state_field, "")
        if raw_state is None:
            raw_state = ""
        if isinstance(raw_state, (dict, list)):
            state_text = json.dumps(raw_state, ensure_ascii=False)
        else:
            state_text = str(raw_state)
        if args.prepend_field:
            extra = row.get(args.prepend_field)
            if extra is not None and str(extra).strip() != "":
                extra_s = json.dumps(extra, ensure_ascii=False) if isinstance(extra, (dict, list)) else str(extra)
                state_text = f"{extra_s} {state_text}"
        n_in += 1
        result = None
        if daemon_port is not None and daemon_predict_via_http is not None:  # type: ignore
            try:
                result = daemon_predict_via_http(  # type: ignore
                    daemon_port, state_text, questions, lang=getattr(args, "lang", None)
                )
                if result is None:
                    raise RuntimeError("daemon no response")
            except Exception as exc:
                print(f"[laya-cli] daemon failed on line {n_in} ({exc}), falling back", file=sys.stderr)
                result = None
                if agent is None:
                    agent = _init_agent(args)
                daemon_port = None
        if result is None:
            try:
                if agent is None:
                    agent = _init_agent(args)
                if args.router:
                    kwargs = {}
                    if args.lang:
                        kwargs["lang"] = args.lang
                    result = agent.predict(state_text, questions, **kwargs)
                else:
                    result = agent.predict(state_text, questions)
            except Exception as exc:
                print(f"[laya-cli] error: predict failed on line {n_in}: {exc}", file=sys.stderr)
                row["_laya_error"] = str(exc)
                sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                continue
        answers = result.get("answers", result)
        if not isinstance(answers, dict):
            answers = {}
        _apply_answers(row, answers, questions, full_probs=args.full_probs)
        n_out += 1
        sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    print(f"[laya-cli] classify: {n_in} in, {n_out} out", file=sys.stderr)


# ---------------------------------------------------------------------------
# filter helpers
# ---------------------------------------------------------------------------

_WHERE_RE = _re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(>=|<=|==|!=|>|<|=)\s*(.+?)\s*$")


def _parse_where(expr: str):
    if not expr:
        return []
    parts = [p.strip() for p in expr.split(",") if p.strip() != ""]
    filters = []
    for p in parts:
        m = _WHERE_RE.match(p)
        if not m:
            print(
                f"error: invalid --where expression {p!r} (expected like 'field>=0.4')",
                file=sys.stderr,
            )
            sys.exit(2)
        field, op, raw_val = m.groups()
        if op == "=":
            op = "=="
        v_str = raw_val.strip()
        if len(v_str) >= 2 and v_str[0] == v_str[-1] and v_str[0] in ('"', "'"):
            value = v_str[1:-1]
        else:
            try:
                if "." in v_str or "e" in v_str.lower():
                    value = float(v_str)
                else:
                    value = int(v_str)
            except ValueError:
                value = v_str
        filters.append((field, op, value))
    return filters


def _parse_sort(expr: str | None):
    if not expr:
        return []
    parts = [p.strip() for p in expr.split(",") if p.strip() != ""]
    keys = []
    for p in parts:
        if p.startswith("-"):
            keys.append((p[1:], -1))
        elif p.startswith("+"):
            keys.append((p[1:], 1))
        else:
            keys.append((p, 1))
    return keys


def _row_get(row: dict[str, Any], field: str):
    return row.get(field)


def _passes_filters(row: dict[str, Any], filters) -> bool:
    for field, op, val in filters:
        cur = _row_get(row, field)
        if cur is None:
            return False
        if isinstance(val, (int, float)) and isinstance(cur, str):
            try:
                cur = float(cur) if "." in cur else int(cur)
            except ValueError:
                return False
        try:
            if op == "==":
                if cur != val:
                    return False
            elif op == "!=":
                if cur == val:
                    return False
            elif op == ">":
                if not (cur > val):
                    return False
            elif op == "<":
                if not (cur < val):
                    return False
            elif op == ">=":
                if not (cur >= val):
                    return False
            elif op == "<=":
                if not (cur <= val):
                    return False
        except TypeError:
            return False
    return True


def cmd_filter(args: argparse.Namespace) -> None:
    filters = _parse_where(args.where)
    sort_keys = _parse_sort(args.sort)
    rows: list[dict[str, Any]] = []
    need_sort = len(sort_keys) > 0
    if need_sort:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if filters and not _passes_filters(row, filters):
                continue
            rows.append(row)
        import functools

        def cmp(a, b):
            for field, direction in sort_keys:
                av = _row_get(a, field)
                bv = _row_get(b, field)
                if av is None and bv is None:
                    continue
                if av is None:
                    return 1
                if bv is None:
                    return -1
                try:
                    if av < bv:
                        return -direction
                    if av > bv:
                        return direction
                except TypeError:
                    sa, sb = str(av), str(bv)
                    if sa < sb:
                        return -direction
                    if sa > sb:
                        return direction
            return 0

        rows.sort(key=functools.cmp_to_key(cmp))
        for r in rows:
            sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        print(f"[laya-cli] filter: {len(rows)} rows (sorted by {args.sort})", file=sys.stderr)
    else:
        n = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if filters and not _passes_filters(row, filters):
                continue
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        sys.stdout.flush()
        print(f"[laya-cli] filter: {n} rows", file=sys.stderr)


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------


def cmd_questions(args: argparse.Namespace) -> None:
    preset = args.preset
    if preset is None or preset.strip().lower() in ("list", "ls", "--list"):
        # list presets
        presets = {
            "triage": "Support ticket triage (intent, urgency, frustration, churn)",
            "email": "Inbound email triage & threat filtering (category, spam, phishing)",
            "guard": "Real-time LLM input guardrails (jailbreak, injection, harm)",
            "moderation": "Content safety & moderation (toxic, harassment, threat, spam)",
            "router": "Intelligent model routing (difficulty, domain, tools, sensitivity)",
        }
        if getattr(args, "q_format", "json") == "table":
            print("Available presets:")
            for k, desc in presets.items():
                print(f"  {k:<12} {desc}")
        else:
            json.dump(presets, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        sys.stdout.flush()
        return
    key = preset.strip().lower()
    mapping = {
        "triage": "triage_questions",
        "email": "email_questions",
        "guard": "guard_questions",
        "moderation": "moderation_questions",
        "router": "router_questions",
    }
    if key not in mapping:
        print(
            f"error: unknown preset {preset!r}; choose from {', '.join(sorted(mapping))} or 'list'",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        import laya
    except ImportError as exc:
        print(f"error: laya not installed: {exc}", file=sys.stderr)
        sys.exit(2)
    fn = getattr(laya, mapping[key])
    qs = fn()
    if getattr(args, "q_format", "json") == "table":
        print(f"Preset: {key}")
        for qid, qdef in qs.items():
            crit = qdef.get("criteria")
            print(f"  {qid} ({qdef.get('type')}): {qdef.get('instructions')}")
            if isinstance(crit, dict):
                for k, v in crit.items():
                    print(f"    - {k}: {v}")
            elif isinstance(crit, list):
                for i, c in enumerate(crit):
                    print(f"    {i}: {c}")
    else:
        json.dump(qs, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> None:
    questions = _load_questions(args.questions)
    if args.field:
        if args.field not in questions:
            print(
                f"error: --field {args.field!r} not in questions.json (available: {', '.join(questions)})",
                file=sys.stderr,
            )
            sys.exit(2)
        questions = {args.field: questions[args.field]}
    labeled_path = args.labeled
    if not os.path.exists(labeled_path):
        print(f"error: labeled file not found: {labeled_path!r}", file=sys.stderr)
        sys.exit(2)

    class _FakeArgs:
        pass

    fake = _FakeArgs()
    fake.model = args.model
    fake.subfolder = args.subfolder
    fake.device = args.device
    fake.router = args.router
    fake.lang = args.lang
    fake.no_daemon = getattr(args, "no_daemon", False)
    fake._warmup_questions = questions
    daemon_port = None
    agent = None
    if not getattr(args, "no_daemon", False) and daemon_port_for_args is not None:
        try:
            daemon_port = daemon_port_for_args(fake)  # type: ignore
        except Exception:
            daemon_port = None
    if daemon_port:
        print(f"[laya-cli] using daemon 127.0.0.1:{daemon_port}", file=sys.stderr)
    else:
        agent = _init_agent(fake)
    rows: list[dict[str, Any]] = []
    with open(labeled_path, encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[laya-cli] warning: skipping invalid JSON line {idx}: {exc}", file=sys.stderr)
                continue
            rows.append(row)
    total = len(rows)
    if total == 0:
        print("No labeled examples found.", file=sys.stderr)
        sys.exit(2)
    stats: dict[str, dict[str, Any]] = {}
    for qid, qdef in questions.items():
        stats[qid] = {
            "type": qdef.get("type"),
            "correct": 0,
            "passing": 0,
            "correct_and_passing": 0,
            "n": 0,
        }
    threshold = args.threshold
    for row in rows:
        raw_state = row.get(args.state_field, "")
        if raw_state is None:
            raw_state = ""
        if isinstance(raw_state, (dict, list)):
            state_text = json.dumps(raw_state, ensure_ascii=False)
        else:
            state_text = str(raw_state)
        if args.prepend_field:
            extra = row.get(args.prepend_field)
            if extra is not None and str(extra).strip() != "":
                extra_s = json.dumps(extra, ensure_ascii=False) if isinstance(extra, (dict, list)) else str(extra)
                state_text = f"{extra_s} {state_text}"
        result = None
        if daemon_port is not None and daemon_predict_via_http is not None:  # type: ignore
            try:
                result = daemon_predict_via_http(daemon_port, state_text, questions, lang=getattr(args, "lang", None))  # type: ignore
                if result is None:
                    raise RuntimeError("daemon no response")
            except Exception as exc:
                print(f"[laya-cli] daemon failed ({exc}), falling back", file=sys.stderr)
                result = None
                if agent is None:
                    agent = _init_agent(fake)
                daemon_port = None
        if result is None:
            try:
                if agent is None:
                    agent = _init_agent(fake)
                if args.router:
                    kwargs = {}
                    if args.lang:
                        kwargs["lang"] = args.lang
                    result = agent.predict(state_text, questions, **kwargs)
                else:
                    result = agent.predict(state_text, questions)
            except Exception as exc:
                print(f"[laya-cli] warning: predict failed on row: {exc}", file=sys.stderr)
                continue
        answers = result.get("answers", result)
        true_val = row.get(args.label_field)
        for qid, qdef in questions.items():
            ans = answers.get(qid)
            if ans is None:
                continue
            qtype = qdef.get("type")
            if len(questions) > 1 and qid in row and qid != args.label_field:
                gt = row.get(qid)
            else:
                gt = true_val
            is_correct = False
            passing = False
            if qtype == "choice":
                pred = ans.get("choice")
                probs = ans.get("probabilities", {})
                p = probs.get(pred, 0) if isinstance(probs, dict) else 0
                is_correct = (str(pred) == str(gt)) if gt is not None else False
                passing = float(p) >= threshold
            elif qtype == "noul":
                p_true = float(ans.get("noul", 0))
                if isinstance(gt, bool):
                    gt_bool = gt
                elif isinstance(gt, (int, float)):
                    gt_bool = bool(gt)
                elif isinstance(gt, str):
                    gt_bool = gt.strip().lower() in ("true", "1", "yes", "y", "t")
                else:
                    gt_bool = bool(gt) if gt is not None else False
                pred_bool = p_true >= threshold
                is_correct = pred_bool == gt_bool
                passing = max(p_true, 1 - p_true) >= threshold
            elif qtype == "score":
                pred_score = float(ans.get("score", 0))
                try:
                    gt_score = float(gt) if gt is not None else None
                except (ValueError, TypeError):
                    gt_score = None
                if gt_score is not None:
                    is_correct = round(pred_score) == round(gt_score)
                passing = float(ans.get("confidence", 0)) >= threshold
            else:
                continue
            st = stats[qid]
            st["n"] += 1
            if is_correct:
                st["correct"] += 1
            if passing:
                st["passing"] += 1
                if is_correct:
                    st["correct_and_passing"] += 1
    out_lines = []
    out_lines.append("laya-cli evaluate report")
    out_lines.append(f"  questions: {', '.join(stats.keys())}")
    out_lines.append(f"  labeled file: {labeled_path}")
    out_lines.append(f"  state field: {args.state_field!r}, label field: {args.label_field!r}, threshold: {threshold}")
    out_lines.append(f"  examples: {total}")
    out_lines.append("")
    for qid, st in stats.items():
        n = st["n"]
        if n == 0:
            continue
        acc = st["correct"] / n if n else 0
        pass_rate = st["passing"] / n if n else 0
        prec = (st["correct_and_passing"] / st["passing"]) if st["passing"] else 0
        esc = 1 - pass_rate
        out_lines.append(f"  [{qid}] type={st['type']}")
        out_lines.append(f"    accuracy: {acc:.1%} ({st['correct']}/{n})")
        out_lines.append(f"    passing @ {threshold}: {pass_rate:.1%} ({st['passing']}/{n})")
        out_lines.append(
            f"    precision @ {threshold} (correct among passing): {prec:.1%} ({st['correct_and_passing']}/{st['passing']})"
            if st["passing"]
            else f"    precision @ {threshold}: n/a (no passing)"
        )
        out_lines.append(f"    escalation rate (1 - passing): {esc:.1%} — share that would go to LLM/human")
        out_lines.append("")
    report = "\n".join(out_lines)
    sys.stdout.write(report + "\n")
    sys.stdout.flush()


def cmd_info(args: argparse.Namespace) -> None:
    import platform

    print(f"laya-cli {__version__}")
    print(f"python: {platform.python_version()} ({sys.executable})")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  cuda devices: {torch.cuda.device_count()}")
        try:
            import torch.backends.mps as mps

            print(f"  mps available: {mps.is_available()}")
        except Exception:
            pass
    except ImportError:
        print("torch: not installed")
    try:
        import laya

        print(f"laya: {getattr(laya, '__version__', 'unknown')}")
        print(f"model: {args.model}" + (f" subfolder={args.subfolder}" if args.subfolder else ""))
        try:
            from huggingface_hub import scan_cache_dir

            cache = scan_cache_dir()
            repos = [r.repo_id for r in cache.repos]
            if args.model in repos:
                print("  cached: yes")
            else:
                print("  cached: not found (will download on first predict)")
        except Exception:
            pass
    except ImportError as e:
        print(f"laya: not installed ({e})")


# ---------------------------------------------------------------------------
# Serve (resident daemon)
# ---------------------------------------------------------------------------


def _serve_status_for_args(args) -> tuple[bool, dict | None, Path]:
    """Helper to get daemon status for given args config."""
    if daemon_file_for is None:
        return False, None, Path()
    daemon_file = daemon_file_for(
        getattr(args, "model", "convaiinnovations/laya"),
        getattr(args, "subfolder", None),
        getattr(args, "device", None),
        bool(getattr(args, "router", False)),
        getattr(args, "lang", None),
    )
    alive, status, _port = is_daemon_alive(daemon_file)  # type: ignore
    return alive, status, daemon_file


def cmd_serve(args: argparse.Namespace) -> None:
    action = getattr(args, "action", None)
    if action == "status":
        alive, status, daemon_file = _serve_status_for_args(args)
        if alive and status:
            print(f"Daemon running: {daemon_file}")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            # also list all daemons if requested?
        else:
            # check if any daemon files exist
            if daemon_file.exists():
                print(f"Daemon file {daemon_file} exists but not responding (stale, removed)")
                try:
                    daemon_file.unlink()
                except Exception:
                    pass
            print("No live daemon for this config (defaults: model=convaiinnovations/laya).")
            print("Run `laya-cli serve --help` to start one, or `laya-cli serve status --all` to list all.")
            # optionally list all daemons
            if CACHE_DIR and CACHE_DIR.exists():  # type: ignore
                files = list(CACHE_DIR.glob("*.json"))  # type: ignore
                if files:
                    print(f"\nOther daemon files ({len(files)}):")
                    for f in files:
                        alive2, st2, _ = is_daemon_alive(f)  # type: ignore
                        state = "alive" if alive2 else "stale"
                        try:
                            info = json.loads(f.read_text())
                            print(
                                f"  {f.name}: {state} pid={info.get('pid')} port={info.get('port')} model={info.get('model')}"
                            )
                        except Exception:
                            print(f"  {f.name}: {state}")
        return
    if action == "stop":
        if getattr(args, "all_daemons", False):
            if CACHE_DIR and CACHE_DIR.exists():  # type: ignore
                files = list(CACHE_DIR.glob("*.json"))  # type: ignore
                if not files:
                    print("No daemon files to stop.")
                    return
                for f in files:
                    alive, _status, port = is_daemon_alive(f)  # type: ignore
                    if alive and port:
                        from .daemon import _http_post_shutdown  # type: ignore

                        ok = _http_post_shutdown(port)  # type: ignore
                        print(f"Stopping {f.name} (port {port}) -> {'ok' if ok else 'failed, killing pid'}")
                        if not ok:
                            try:
                                info = json.loads(f.read_text())
                                os.kill(info.get("pid", 0), signal.SIGTERM)
                            except Exception as e:
                                print(f"  kill failed: {e}")
                        # wait a bit and clean
                        time.sleep(0.3)
                        if f.exists():
                            try:
                                f.unlink()
                            except Exception:
                                pass
                    else:
                        print(f"Removing stale {f.name}")
                        try:
                            f.unlink()
                        except Exception:
                            pass
                print("All daemons stopped.")
            else:
                print("No daemons running.")
            return
        else:
            alive, _status, daemon_file = _serve_status_for_args(args)
            # need port from file
            try:
                info = json.loads(daemon_file.read_text()) if daemon_file.exists() else {}
                port = info.get("port")
            except Exception:
                port = None
            if not alive:
                print(f"No live daemon for this config ({daemon_file}). Nothing to stop.")
                if daemon_file.exists():
                    try:
                        daemon_file.unlink()
                        print(f"Removed stale file {daemon_file}")
                    except Exception:
                        pass
                return
            from .daemon import _http_post_shutdown  # type: ignore

            ok = _http_post_shutdown(port)  # type: ignore
            if ok:
                print(f"Daemon on 127.0.0.1:{port} stopped via /shutdown")
            else:
                # fallback to SIGTERM
                try:
                    info = json.loads(daemon_file.read_text())
                    os.kill(info.get("pid", 0), signal.SIGTERM)
                    print(f"Daemon on 127.0.0.1:{port} killed via SIGTERM")
                except Exception as e:
                    print(f"Failed to stop daemon: {e}", file=sys.stderr)
                    sys.exit(1)
            # wait for file removal (daemon cleans itself)
            for _ in range(10):
                if not daemon_file.exists():
                    break
                time.sleep(0.2)
            if daemon_file.exists():
                try:
                    daemon_file.unlink()
                except Exception:
                    pass
            return

    # Start daemon (no action)
    # Check if already alive for this config
    alive, status, daemon_file = _serve_status_for_args(args)
    if alive and status:
        print(f"Daemon already running for this config: {daemon_file}")
        print(json.dumps(status, indent=2, ensure_ascii=False))
        print("Use `laya-cli serve stop` to stop it, or `laya-cli serve stop --all` for all.")
        return

    port = getattr(args, "port", 0) or 0
    idle_timeout = getattr(args, "idle_timeout", 1800)
    foreground = bool(getattr(args, "foreground", False))

    if foreground:
        # run in foreground (blocking)
        if run_server is None:
            print("error: daemon module not available", file=sys.stderr)
            sys.exit(2)
        daemon_file = daemon_file_for(  # type: ignore
            getattr(args, "model", "convaiinnovations/laya"),
            getattr(args, "subfolder", None),
            getattr(args, "device", None),
            bool(getattr(args, "router", False)),
            getattr(args, "lang", None),
        )
        run_server(  # type: ignore
            model=getattr(args, "model", "convaiinnovations/laya"),
            subfolder=getattr(args, "subfolder", None),
            device=getattr(args, "device", None),
            router=bool(getattr(args, "router", False)),
            lang=getattr(args, "lang", None),
            port=port,
            idle_timeout=idle_timeout,
            daemon_file=daemon_file,
        )
        return
    else:
        # background: spawn child with --foreground
        import subprocess

        daemon_file = daemon_file_for(  # type: ignore
            getattr(args, "model", "convaiinnovations/laya"),
            getattr(args, "subfolder", None),
            getattr(args, "device", None),
            bool(getattr(args, "router", False)),
            getattr(args, "lang", None),
        )
        # ensure cache dir exists
        daemon_file.parent.mkdir(parents=True, exist_ok=True)
        # Build child command
        cmd = [
            sys.executable,
            "-m",
            "laya_cli.cli",
            "serve",
            "--foreground",
            "--port",
            str(port),
            "--idle-timeout",
            str(idle_timeout),
        ]
        # forward config
        if getattr(args, "model", None) and getattr(args, "model") != "convaiinnovations/laya":
            cmd.extend(["--model", getattr(args, "model")])
        if getattr(args, "subfolder", None):
            cmd.extend(["--subfolder", getattr(args, "subfolder")])
        if getattr(args, "device", None):
            cmd.extend(["--device", getattr(args, "device")])
        if getattr(args, "router", False):
            cmd.append("--router")
        if getattr(args, "lang", None):
            cmd.extend(["--lang", getattr(args, "lang")])

        log_file = daemon_file.with_suffix(".log")
        print(f"[laya-cli] starting daemon in background (log: {log_file}) ...", file=sys.stderr)
        # detach: start new session, redirect output
        with open(log_file, "a") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
        # wait for daemon to be ready (poll status)
        for _ in range(30):
            time.sleep(0.5)
            alive, status, _ = _serve_status_for_args(args)
            if alive and status:
                print(
                    f"Daemon started on 127.0.0.1:{status.get('port')} pid={status.get('pid')} (daemon file: {daemon_file})",
                    file=sys.stderr,
                )
                print(json.dumps(status, indent=2, ensure_ascii=False))
                return
            # if child died quickly, check
            if proc.poll() is not None:
                print(f"Daemon process exited early with code {proc.returncode}. See log {log_file}", file=sys.stderr)
                try:
                    print(log_file.read_text()[-2000:], file=sys.stderr)
                except Exception:
                    pass
                sys.exit(1)
        print(f"Daemon did not become ready in 15s. Check log {log_file} and `laya-cli serve status`", file=sys.stderr)
        sys.exit(1)


def _preprocess_argv(argv: list[str]) -> list[str]:
    known_flags = {
        "--questions",
        "--state-field",
        "--model",
        "--subfolder",
        "--device",
        "--router",
        "--lang",
        "--prepend-field",
        "--full-probs",
        "--where",
        "--sort",
        "--labeled",
        "--label-field",
        "--field",
        "--threshold",
        "--help",
        "-h",
        "--version",
        "--text",
        "--state",
        "--state-file",
        "--input",
        "--questions-inline",
        "--preset",
        "--shortlist-k",
        "--format",
        "--pretty",
        "--flatten",
        "--output",
        "--q_format",
        "--port",
        "--idle-timeout",
        "--foreground",
        "--all",
        "--no-daemon",
    }
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        out.append(tok)
        if tok == "--sort" and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt not in known_flags and (nxt.startswith("-") or nxt.startswith("+")):
                out[-1] = f"--sort={nxt}"
                i += 1
        i += 1
    return out


def main() -> None:
    sys.argv = _preprocess_argv(sys.argv)
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "classify":
        cmd_classify(args)
    elif args.cmd == "filter":
        cmd_filter(args)
    elif args.cmd in ("questions", "presets"):
        cmd_questions(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "serve":
        cmd_serve(args)
    elif args.cmd == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
