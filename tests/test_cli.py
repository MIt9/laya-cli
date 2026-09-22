import io
import sys

import laya_cli.cli as cli


def _help_text(args):
    old_argv, old_stdout, old_stderr = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["laya-cli"] + args
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        try:
            cli.main()
        except SystemExit as e:
            # argparse --help exits 0
            out = sys.stdout.getvalue() + sys.stderr.getvalue()
            return out, e.code
        out = sys.stdout.getvalue() + sys.stderr.getvalue()
        return out, 0
    finally:
        sys.argv, sys.stdout, sys.stderr = old_argv, old_stdout, old_stderr


def test_version_flag():
    out, code = _help_text(["--version"])
    assert code == 0


def test_help_has_examples():
    for cmd in [
        ["predict", "--help"],
        ["classify", "--help"],
        ["filter", "--help"],
        ["questions", "--help"],
        ["evaluate", "--help"],
    ]:
        out, code = _help_text(cmd)
        assert "Examples" in out or "examples" in out.lower()
        assert code == 0


def test_router_warning_without_lang(fake_laya, capsys=None):
    old_argv, old_stdin, old_stdout, old_stderr = sys.argv, sys.stdin, sys.stdout, sys.stderr
    import io

    sys.argv = [
        "laya-cli",
        "predict",
        "--preset",
        "guard",
        "--router",
        "--text",
        "hello",
        "--format",
        "json",
    ]
    sys.stdin = io.StringIO("")
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        err = sys.stderr.getvalue()
    finally:
        sys.argv, sys.stdin, sys.stdout, sys.stderr = old_argv, old_stdin, old_stdout, old_stderr
    assert "warning" in err.lower() and "router" in err.lower()

    # with --lang should not warn about missing lang
    sys.argv = [
        "laya-cli",
        "predict",
        "--preset",
        "guard",
        "--router",
        "--lang",
        "en",
        "--text",
        "hello",
        "--format",
        "json",
    ]
    sys.stdin = io.StringIO("")
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        err2 = sys.stderr.getvalue()
    finally:
        sys.argv, sys.stdin, sys.stdout, sys.stderr = old_argv, old_stdin, old_stdout, old_stderr
    assert "without --lang" not in err2


def test_info_command(fake_laya):
    out, code = _help_text(["info", "--help"])
    assert code == 0
    # actual run
    old_argv, old_stdout, old_stderr = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["laya-cli", "info"]
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        out = sys.stdout.getvalue()
    finally:
        sys.argv, sys.stdout, sys.stderr = old_argv, old_stdout, old_stderr
    assert "laya-cli" in out
    assert "python:" in out
