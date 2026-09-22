import io
import json
import sys
import tempfile

import laya_cli.cli as cli


def _run_classify(args, stdin_data):
    old_argv, old_stdin, old_stdout, old_stderr = sys.argv, sys.stdin, sys.stdout, sys.stderr
    sys.argv = ["laya-cli"] + args
    sys.stdin = io.StringIO(stdin_data)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        out, err = sys.stdout.getvalue(), sys.stderr.getvalue()
    finally:
        sys.argv, sys.stdin, sys.stdout, sys.stderr = old_argv, old_stdin, old_stdout, old_stderr
    return out, err


def test_classify_state_verbatim_and_single_load(fake_laya):
    mod, calls = fake_laya
    q = {
        "on_topic": {
            "type": "choice",
            "instructions": "Is about nature?",
            "criteria": {"yes": "nature", "no": "city"},
        },
        "is_relevant": {"type": "noul", "instructions": "Is it relevant?"},
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        json.dump(q, qf)
        qpath = qf.name
    # include extra field that should NOT be mixed into state
    lines = "\n".join(
        [
            json.dumps({"id": 1, "state": "forest at dawn", "photographer": "Alice"}),
            json.dumps({"id": 2, "state": "city street"}),
        ]
    )
    calls.clear()
    out, err = _run_classify(["classify", "--questions", qpath], lines)
    rows = [json.loads(line) for line in out.strip().split("\n") if line.strip()]
    assert len(rows) == 2
    # calls[0] is warmup, calls[1]/[2] are actual states
    assert len(calls) == 3
    assert calls[1][0] == "forest at dawn"
    assert calls[2][0] == "city street"
    # flattened fields present, probs not unless --full-probs
    assert "on_topic" in rows[0] and "on_topic_p" in rows[0] and "on_topic_confidence" in rows[0]
    assert "on_topic_probs" not in rows[0]
    assert "is_relevant" in rows[0]


def test_classify_prepend_field(fake_laya):
    mod, calls = fake_laya
    q = {"on_topic": {"type": "choice", "instructions": "Is?", "criteria": {"yes": "y", "no": "n"}}}
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        json.dump(q, qf)
        qpath = qf.name
    calls.clear()
    out, err = _run_classify(
        ["classify", "--questions", qpath, "--prepend-field", "photographer"],
        json.dumps({"state": "forest", "photographer": "Alice"}),
    )
    assert calls[1][0] == "Alice forest"


def test_classify_full_probs(fake_laya):
    q = {"on_topic": {"type": "choice", "instructions": "Is?", "criteria": {"yes": "y", "no": "n"}}}
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        json.dump(q, qf)
        qpath = qf.name
    out, err = _run_classify(["classify", "--questions", qpath, "--full-probs"], json.dumps({"state": "forest"}))
    row = json.loads(out.strip())
    assert "on_topic_probs" in row
