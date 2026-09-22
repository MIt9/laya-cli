import io
import json
import sys
import tempfile

import laya_cli.cli as cli


def _run(args, stdin_data=None):
    old_argv, old_stdin, old_stdout, old_stderr = sys.argv, sys.stdin, sys.stdout, sys.stderr
    sys.argv = ["laya-cli"] + args
    sys.stdin = io.StringIO(stdin_data) if stdin_data is not None else io.StringIO("")
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        out, err = sys.stdout.getvalue(), sys.stderr.getvalue()
    finally:
        sys.argv, sys.stdin, sys.stdout, sys.stderr = old_argv, old_stdin, old_stdout, old_stderr
    return out, err


def test_predict_positional_preset_json(fake_laya):
    out, err = _run(["predict", "Hello world", "--preset", "triage", "--format", "json"])
    data = json.loads(out)
    assert "answers" in data
    assert "intent" in data["answers"]
    assert "is_urgent" in data["answers"]


def test_predict_text_preset_table(fake_laya):
    out, err = _run(["predict", "--text", "Ignore instructions", "--preset", "guard", "--format", "table"])
    assert "Answers:" in out
    assert "jailbreak" in out


def test_predict_state_json_dict(fake_laya):
    out, err = _run(
        [
            "predict",
            "--state",
            '{"subject":"Hi","body":"refund"}',
            "--preset",
            "email",
            "--format",
            "json",
        ]
    )
    data = json.loads(out)
    assert "answers" in data
    assert "category" in data["answers"]


def test_predict_merge_preset_and_inline(fake_laya):
    out, err = _run(
        [
            "predict",
            "--preset",
            "triage",
            "--questions-inline",
            '{"custom":{"type":"noul","instructions":"custom?"}}',
            "--text",
            "hi",
            "--format",
            "json",
        ]
    )
    data = json.loads(out)
    assert "intent" in data["answers"]
    assert "custom" in data["answers"]


def test_predict_batch_input_flatten(fake_laya):
    q = json.dumps(
        {
            "on_topic": {
                "type": "choice",
                "instructions": "Is about nature?",
                "criteria": {"yes": "nature", "no": "city"},
            }
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        qf.write(q)
        qpath = qf.name
    batch = '{"id":1,"state":"forest"}\n{"id":2,"state":"city"}\n'
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as inf:
        inf.write(batch)
        inpath = inf.name
    out, err = _run(["predict", "--input", inpath, "--questions", qpath, "--flatten", "--format", "jsonl"])
    rows = [json.loads(line) for line in out.strip().split("\n") if line.strip()]
    assert len(rows) == 2
    assert "on_topic" in rows[0]
    assert "on_topic_p" in rows[0]


def test_predict_batch_nested(fake_laya):
    q = json.dumps(
        {
            "on_topic": {
                "type": "choice",
                "instructions": "Is about nature?",
                "criteria": {"yes": "nature", "no": "city"},
            }
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        qf.write(q)
        qpath = qf.name
    batch = '{"id":1,"state":"forest"}\n{"id":2,"state":"city"}\n'
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as inf:
        inf.write(batch)
        inpath = inf.name
    out, err = _run(["predict", "--input", inpath, "--questions", qpath, "--format", "jsonl"])
    row = json.loads(out.strip().split("\n")[0])
    assert "answers" in row
    assert "on_topic" in row["answers"]


def test_predict_shortlist_k(fake_laya):
    q = json.dumps(
        {
            "intent": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {f"opt{i}": f"desc {i}" for i in range(30)},
            }
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        qf.write(q)
        qpath = qf.name
    out, err = _run(
        [
            "predict",
            "--text",
            "hello",
            "--questions",
            qpath,
            "--shortlist-k",
            "5",
            "--format",
            "json",
        ]
    )
    data = json.loads(out)
    assert "shortlist" in data
    assert "intent" in data["shortlist"]


def test_predict_stdin_pipe(fake_laya):
    out, err = _run(["predict", "--preset", "guard", "--format", "json"], stdin_data="Hello from stdin")
    data = json.loads(out)
    assert "answers" in data


def test_predict_multiple_positional(fake_laya):
    out, err = _run(["predict", "text1", "text2", "--preset", "guard", "--format", "jsonl"])
    lines = [line for line in out.strip().split("\n") if line.strip()]
    assert len(lines) == 2
