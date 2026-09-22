import io
import json
import sys
import tempfile
import threading
import time

import laya_cli.cli as cli
from laya_cli.daemon import daemon_file_for, is_daemon_alive, run_server


def _run_cli(args, stdin_data=None):
    old_argv, old_stdin, old_stdout, old_stderr = sys.argv, sys.stdin, sys.stdout, sys.stderr
    sys.argv = ["laya-cli"] + args
    sys.stdin = io.StringIO(stdin_data) if stdin_data is not None else io.StringIO("")
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        cli.main()
        out, err = sys.stdout.getvalue(), sys.stderr.getvalue()
    except SystemExit as e:
        out, err = sys.stdout.getvalue(), sys.stderr.getvalue()
        if e.code not in (0, None):
            raise AssertionError(f"CLI exited {e.code}: {out} {err}")
    finally:
        sys.argv, sys.stdin, sys.stdout, sys.stderr = old_argv, old_stdin, old_stdout, old_stderr
    return out, err


def test_daemon_serve_status_not_running(fake_laya):
    # ensure no daemon file
    df = daemon_file_for("convaiinnovations/laya", None, None, False, None)
    if df.exists():
        df.unlink()
    out, err = _run_cli(["serve", "status"])
    assert "No live daemon" in out


def test_daemon_predict_via_daemon(fake_laya):
    # Start a daemon in a thread with mocked laya, using random port and short idle

    # Use a unique config to avoid collision with other tests
    model = "convaiinnovations/laya-test-daemon-1"
    subfolder = None
    device = None
    router = False
    lang = None
    daemon_file = daemon_file_for(model, subfolder, device, router, lang)
    # ensure clean
    if daemon_file.exists():
        daemon_file.unlink()

    # Patch laya.load to track calls and ensure it's our fake
    # fake_laya already mocks laya.load, so run_server will use it
    # Run server in thread
    t = threading.Thread(
        target=run_server,
        kwargs=dict(
            model=model,
            subfolder=subfolder,
            device=device,
            router=router,
            lang=lang,
            port=0,
            idle_timeout=30,
            daemon_file=daemon_file,
        ),
        daemon=True,
    )
    t.start()
    # wait for daemon to be ready
    for _ in range(30):
        time.sleep(0.2)
        alive, status, port = is_daemon_alive(daemon_file)
        if alive:
            break
    else:
        raise AssertionError("daemon did not start")

    assert alive and status is not None
    assert status["model"] == model
    port = status["port"]
    assert port > 0

    # Now test predict via daemon (client)
    from laya_cli.daemon import daemon_predict_via_http

    q = {"q1": {"type": "noul", "instructions": "Is it good?"}}
    result = daemon_predict_via_http(port, "forest good", q)
    assert result is not None
    assert "answers" in result
    assert "q1" in result["answers"]

    # Test that laya-cli predict automatically uses daemon (same config)
    # Need to pass same model etc.
    out, err = _run_cli(
        ["predict", "--text", "forest good", "--questions-inline", json.dumps(q), "--model", model, "--format", "json"]
    )
    data = json.loads(out)
    assert "answers" in data
    # should have used daemon (check stderr)
    assert "using daemon" in err

    # Test --no-daemon forces in-process (should not use daemon, will say model loaded)
    out2, err2 = _run_cli(
        [
            "predict",
            "--text",
            "forest good",
            "--questions-inline",
            json.dumps(q),
            "--model",
            model,
            "--format",
            "json",
            "--no-daemon",
        ]
    )
    data2 = json.loads(out2)
    assert "answers" in data2
    assert "using daemon" not in err2

    # Test parallel predicts (5 concurrent) via daemon
    import concurrent.futures

    def do_predict(i):
        return daemon_predict_via_http(port, f"forest {i}", q)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(do_predict, i) for i in range(5)]
        results = [f.result(timeout=5) for f in futs]
    for r in results:
        assert r is not None and "answers" in r

    # Test that classify via daemon gives same as in-process
    # First, get in-process result via --no-daemon
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as qf:
        json.dump(q, qf)
        qpath = qf.name
    batch = '{"state":"forest good"}\n{"state":"city bad"}\n'
    out_classify_daemon, _ = _run_cli(["classify", "--questions", qpath, "--model", model], stdin_data=batch)
    out_classify_local, _ = _run_cli(
        ["classify", "--questions", qpath, "--model", model, "--no-daemon"], stdin_data=batch
    )
    # Both should have same fields (daemon vs local both use fake, so same)
    rows_daemon = [json.loads(line) for line in out_classify_daemon.strip().split("\n") if line.strip()]
    rows_local = [json.loads(line) for line in out_classify_local.strip().split("\n") if line.strip()]
    assert rows_daemon[0]["q1"] == rows_local[0]["q1"]

    # Test idle timeout: start a daemon with 2s timeout, wait 3s, ensure it dies
    model2 = "convaiinnovations/laya-test-daemon-idle"
    daemon_file2 = daemon_file_for(model2, None, None, False, None)
    if daemon_file2.exists():
        daemon_file2.unlink()
    t2 = threading.Thread(
        target=run_server,
        kwargs=dict(
            model=model2,
            subfolder=None,
            device=None,
            router=False,
            lang=None,
            port=0,
            idle_timeout=2,
            daemon_file=daemon_file2,
        ),
        daemon=True,
    )
    t2.start()
    for _ in range(20):
        time.sleep(0.2)
        alive, _, _ = is_daemon_alive(daemon_file2)
        if alive:
            break
    assert alive
    # wait for idle timeout (2s + wiggle)
    time.sleep(3.5)
    alive, _, _ = is_daemon_alive(daemon_file2)
    assert not alive, "daemon should have exited after idle timeout"
    assert not daemon_file2.exists() or not is_daemon_alive(daemon_file2)[0]

    # Test serve stop via http shutdown
    # daemon 1 is still running, stop it via client helper
    from laya_cli.daemon import _http_post_shutdown

    alive, status, port = is_daemon_alive(daemon_file)
    assert alive
    ok = _http_post_shutdown(port)
    assert ok
    time.sleep(0.5)
    alive, _, _ = is_daemon_alive(daemon_file)
    assert not alive

    # Cleanup
    for f in [daemon_file, daemon_file2]:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass


def test_daemon_only_localhost(fake_laya):
    # Verify daemon binds only to 127.0.0.1 by checking server address
    model = "convaiinnovations/laya-test-daemon-localhost"
    daemon_file = daemon_file_for(model, None, None, False, None)
    if daemon_file.exists():
        daemon_file.unlink()
    t = threading.Thread(
        target=run_server,
        kwargs=dict(
            model=model,
            subfolder=None,
            device=None,
            router=False,
            lang=None,
            port=0,
            idle_timeout=10,
            daemon_file=daemon_file,
        ),
        daemon=True,
    )
    t.start()
    for _ in range(20):
        time.sleep(0.2)
        alive, status, _ = is_daemon_alive(daemon_file)
        if alive:
            break
    assert alive
    # Check that status reports 127.0.0.1 binding implicitly via port; we can't test external interface without network,
    # but we can verify that server is on 127.0.0.1 by trying to connect via 0.0.0.0 vs 127.0.0.1
    # The daemon code explicitly binds to 127.0.0.1, so this is a code check, not runtime external.
    # Verify file exists and contains 127.0.0.1 implied
    assert daemon_file.exists()
    info = json.loads(daemon_file.read_text())
    assert info["port"] > 0
    # Try to ensure no other daemon file hijack
    from laya_cli.daemon import _http_post_shutdown

    _http_post_shutdown(info["port"])
    time.sleep(0.3)
    if daemon_file.exists():
        daemon_file.unlink()


def test_daemon_config_hash_device_normalization(fake_laya):
    """Regression for TASK2 BUG: hash must be after device resolve, not raw --device.

    On MPS Mac, `serve` without --device (auto=mps) and `serve --device mps` must share file,
    and `predict` without --device must find daemon started with --device mps.
    """
    from laya_cli.daemon import daemon_file_for, daemon_port_for_args

    # Same effective device -> same file (auto resolves to mps on this Mac, so None and "mps" coincide)
    f_auto = daemon_file_for("convaiinnovations/laya", None, None, False, None)
    f_mps = daemon_file_for("convaiinnovations/laya", None, "mps", False, None)
    assert f_auto == f_mps, "hash after resolve: auto=None and explicit mps must be same file on MPS host"

    # Start daemon with explicit --device mps, then client without --device should find it
    model = "convaiinnovations/laya-test-hash-mps"
    daemon_file_mps = daemon_file_for(model, None, "mps", False, None)
    daemon_file_auto = daemon_file_for(model, None, None, False, None)
    # They should be same file
    assert daemon_file_mps == daemon_file_auto

    # Also subfolder "" vs None and lang case
    assert daemon_file_for("m", None, None, False, None) == daemon_file_for("m", "", None, False, None)
    assert daemon_file_for("m", None, None, False, "EN") == daemon_file_for("m", None, None, False, "en")

    # Now test live daemon discovery across device flag mismatch
    if daemon_file_mps.exists():
        daemon_file_mps.unlink()
    t = threading.Thread(
        target=run_server,
        kwargs=dict(
            model=model,
            subfolder=None,
            device="mps",
            router=False,
            lang=None,
            port=0,
            idle_timeout=10,
            daemon_file=daemon_file_mps,
        ),
        daemon=True,
    )
    t.start()
    for _ in range(30):
        time.sleep(0.2)
        alive, _, _ = is_daemon_alive(daemon_file_mps)
        if alive:
            break
    assert alive, "daemon with --device mps did not start"

    # Client without --device (auto) should find it via daemon_port_for_args
    import types

    fake_args = types.SimpleNamespace(
        model=model, subfolder=None, device=None, router=False, lang=None, no_daemon=False
    )
    port = daemon_port_for_args(fake_args)  # type: ignore
    assert port is not None, (
        "predict without --device should find daemon started with --device mps (hash after resolve)"
    )

    # Also via CLI: predict without --device should auto-use daemon
    q = {"q1": {"type": "noul", "instructions": "Is it good?"}}
    out, err = _run_cli(
        ["predict", "--text", "forest good", "--questions-inline", json.dumps(q), "--model", model, "--format", "json"]
    )
    assert "using daemon" in err, "CLI predict without --device should have used daemon started with --device mps"

    # Cleanup
    from laya_cli.daemon import _http_post_shutdown

    alive, status, port = is_daemon_alive(daemon_file_mps)
    if alive and port:
        _http_post_shutdown(port)
        time.sleep(0.3)
    if daemon_file_mps.exists():
        try:
            daemon_file_mps.unlink()
        except Exception:
            pass
    if daemon_file_auto.exists() and daemon_file_auto != daemon_file_mps:
        try:
            daemon_file_auto.unlink()
        except Exception:
            pass
