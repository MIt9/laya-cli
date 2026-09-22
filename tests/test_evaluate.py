import io
import json
import sys
import tempfile

import laya_cli.cli as cli


def _run(args):
    old_argv, old_stdout, old_stderr = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["laya-cli"] + args
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
    finally:
        sys.argv, sys.stdout, sys.stderr = old_argv, old_stdout, old_stderr
    return out, err


def test_evaluate_reports_numbers(fake_laya):
    q = {
        "on_topic": {
            "type": "choice",
            "instructions": "Is good?",
            "criteria": {"yes": "good", "no": "bad"},
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        json.dump(q, qf)
        qpath = qf.name
    labeled = [
        {"state": "this is good", "label": "yes"},
        {"state": "this is good and nice", "label": "yes"},
        {"state": "bad stuff", "label": "no"},
        {"state": "good bad mix", "label": "no"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as lf:
        for r in labeled:
            lf.write(json.dumps(r) + "\n")
        lpath = lf.name
    out, err = _run(
        [
            "evaluate",
            "--questions",
            qpath,
            "--labeled",
            lpath,
            "--label-field",
            "label",
            "--threshold",
            "0.5",
        ]
    )
    assert "accuracy:" in out
    assert "passing @" in out
    assert "precision @" in out
    assert "escalation rate" in out
    assert "examples: 4" in out


def test_evaluate_single_field(fake_laya):
    # multi-question file but evaluate only one field via --field
    q = {
        "a": {"type": "choice", "instructions": "Is?", "criteria": {"yes": "y", "no": "n"}},
        "b": {"type": "noul", "instructions": "Is?"},
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        json.dump(q, qf)
        qpath = qf.name
    labeled = [{"state": "good", "label": "yes"}, {"state": "bad", "label": "no"}]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as lf:
        for r in labeled:
            lf.write(json.dumps(r) + "\n")
        lpath = lf.name
    out, err = _run(["evaluate", "--questions", qpath, "--labeled", lpath, "--field", "a"])
    assert "questions: a" in out or "[a]" in out
