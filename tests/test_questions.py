import io
import json
import sys

import laya_cli.cli as cli


def _run(args):
    old_argv, old_stdout, old_stderr = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["laya-cli"] + args
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        out = sys.stdout.getvalue()
    finally:
        sys.argv, sys.stdout, sys.stderr = old_argv, old_stdout, old_stderr
    return out


def test_questions_list(fake_laya):
    out = _run(["questions", "list"])
    data = json.loads(out)
    assert "triage" in data
    assert "guard" in data


def test_questions_list_table(fake_laya):
    out = _run(["questions", "list", "--format", "table"])
    assert "triage" in out and "guard" in out


def test_questions_preset_valid_json_and_classify_compatible(fake_laya):
    for preset in ["triage", "email", "guard", "moderation", "router"]:
        out = _run(["questions", preset])
        data = json.loads(out)
        assert isinstance(data, dict) and len(data) > 0
        # every question has type
        for qid, qdef in data.items():
            assert qdef["type"] in ("choice", "score", "noul")


def test_questions_unknown_exits(fake_laya):
    import pytest

    old_argv = sys.argv
    sys.argv = ["laya-cli", "questions", "unknown"]
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    sys.argv = old_argv
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__


def test_presets_alias(fake_laya):
    out = _run(["presets", "triage"])
    data = json.loads(out)
    assert "intent" in data
