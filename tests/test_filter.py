import io
import json
import sys

import laya_cli.cli as cli


def _run_filter(args, stdin_data):
    old_argv, old_stdin, old_stdout, old_stderr = sys.argv, sys.stdin, sys.stdout, sys.stderr
    sys.argv = ["laya-cli"] + args
    sys.stdin = io.StringIO(stdin_data)
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        out = sys.stdout.getvalue()
    finally:
        sys.argv, sys.stdin, sys.stdout, sys.stderr = old_argv, old_stdin, old_stdout, old_stderr
    return out


def test_filter_where_and_sort():
    data = '{"id":1,"on_topic":0.9,"category":"a"}\n{"id":2,"on_topic":0.3,"category":"b"}\n{"id":3,"on_topic":0.5,"category":"a"}\n{"id":4,"on_topic":0.7,"category":"b"}\n'
    out = _run_filter(["filter", "--where", "on_topic>=0.5", "--sort", "-on_topic"], data)
    rows = [json.loads(line) for line in out.strip().split("\n") if line.strip()]
    assert [r["id"] for r in rows] == [1, 4, 3]

    out2 = _run_filter(["filter", "--where", "on_topic>=0.4,category==a"], data)
    rows2 = [json.loads(line) for line in out2.strip().split("\n") if line.strip()]
    assert {r["id"] for r in rows2} == {1, 3}

    # sort asc via +
    out3 = _run_filter(["filter", "--where", "on_topic>0.4", "--sort", "+on_topic"], data)
    rows3 = [json.loads(line) for line in out3.strip().split("\n") if line.strip()]
    assert rows3[0]["on_topic"] == 0.5


def test_filter_sort_dash_without_equals():
    # --sort -on_topic (dash) requires argv preprocessing
    data = '{"v":0.1}\n{"v":0.9}\n{"v":0.5}\n'
    out = _run_filter(["filter", "--where", "v>=0.4", "--sort", "-v"], data)
    rows = [json.loads(line) for line in out.strip().split("\n") if line.strip()]
    assert rows[0]["v"] == 0.9


def test_filter_invalid_where_exits():
    import pytest

    old_argv = sys.argv
    sys.argv = ["laya-cli", "filter", "--where", "invalid expr"]
    import laya_cli.cli as cli

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    sys.argv = old_argv
