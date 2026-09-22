"""Resident daemon for laya-cli — HTTP sidecar on 127.0.0.1.

Implements the SKILL.md pattern: one GPU = one forward pass at a time,
so requests are serialized with a threading.Lock. Daemon binds only to
127.0.0.1 and stores pid+port in ~/.cache/laya-cli/daemons/<hash>.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config hashing & file paths
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "laya-cli" / "daemons"


def config_hash(model: str, subfolder: str | None, device: str | None, router: bool, lang: str | None) -> str:
    raw = "|".join([model or "", subfolder or "", device or "", "1" if router else "0", lang or ""])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def daemon_file_for(model: str, subfolder: str | None, device: str | None, router: bool, lang: str | None) -> Path:
    h = config_hash(model, subfolder, device, router, lang)
    return CACHE_DIR / f"{h}.json"


def daemon_file_default() -> Path:
    # default config = same as predict defaults
    return daemon_file_for("convaiinnovations/laya", None, None, False, None)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class DaemonHandler(BaseHTTPRequestHandler):
    # class-level shared state, set by run_server
    agent: Any = None
    lock: threading.Lock | None = None
    loaded_at: float = 0
    requests_served: int = 0
    last_request: float = 0
    idle_timeout: int = 1800
    config: dict[str, Any] = {}
    shutdown_flag: threading.Event | None = None

    def log_message(self, format, *args):
        # suppress default logging, use our own
        sys.stderr.write(f"[daemon] {self.client_address[0]} {format % args}\n")

    def _json_response(self, obj: Any, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/status", "/status/"):
            idle = time.time() - self.__class__.last_request if self.__class__.last_request else 0
            # need to touch last_request? No, /status shouldn't count as request for idle?
            # But we count it as activity to keep daemon alive if someone polls status
            # For idle, only /predict counts. So don't update last_request here.
            info = {
                "model": self.__class__.config.get("model"),
                "subfolder": self.__class__.config.get("subfolder"),
                "device": str(getattr(self.__class__.agent, "device", self.__class__.config.get("device") or "auto")),
                "router": self.__class__.config.get("router", False),
                "lang": self.__class__.config.get("lang"),
                "loaded_at": self.__class__.loaded_at,
                "idle_seconds": idle,
                "requests_served": self.__class__.requests_served,
                "pid": os.getpid(),
                "port": self.server.server_address[1],
                "idle_timeout": self.__class__.idle_timeout,
            }
            self._json_response(info)
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(body_bytes.decode() or "{}")
        except json.JSONDecodeError as e:
            self._json_response({"error": f"invalid json: {e}"}, 400)
            return

        if self.path in ("/predict", "/predict/"):
            state = body.get("state")
            questions = body.get("questions")
            lang = body.get("lang")
            shortlist_k = body.get("shortlist_k")
            if questions is None:
                self._json_response({"error": "missing questions"}, 400)
                return
            # serialize requests (one GPU)
            lock = self.__class__.lock
            agent = self.__class__.agent
            if lock is not None:
                with lock:
                    result = self._run_predict(agent, state, questions, lang, shortlist_k)
            else:
                result = self._run_predict(agent, state, questions, lang, shortlist_k)
            self.__class__.requests_served += 1
            self.__class__.last_request = time.time()
            # include routing if router
            self._json_response(result)
        elif self.path in ("/shutdown", "/shutdown/"):
            # only from localhost, already bound to 127.0.0.1
            self._json_response({"ok": True})

            # trigger shutdown in background thread to allow response to flush
            def _shutdown():
                time.sleep(0.1)
                if self.__class__.shutdown_flag is not None:
                    self.__class__.shutdown_flag.set()
                # shutdown server
                try:
                    self.server.shutdown()
                except Exception:
                    pass

            threading.Thread(target=_shutdown, daemon=True).start()
        else:
            self._json_response({"error": "not found"}, 404)

    @staticmethod
    def _run_predict(agent: Any, state: Any, questions: dict[str, Any], lang: str | None, shortlist_k: Any):
        # handle shortlist if requested
        if shortlist_k is not None:
            try:
                from laya import embed_fn_from_agent, predict_shortlist

                embed_fn = embed_fn_from_agent(agent)
                # Router vs Agent: Router's predict_shortlist signature forwards kwargs
                if hasattr(agent, "route"):  # Router
                    return (
                        predict_shortlist(agent, state, questions, embed_fn, k=int(shortlist_k), lang=lang)
                        if lang
                        else predict_shortlist(agent, state, questions, embed_fn, k=int(shortlist_k))
                    )
                else:
                    return predict_shortlist(agent, state, questions, embed_fn, k=int(shortlist_k))
            except Exception as e:
                return {"error": f"shortlist failed: {e}"}
        else:
            if hasattr(agent, "route"):  # Router
                if lang:
                    return agent.predict(state, questions, lang=lang)
                else:
                    return agent.predict(state, questions)
            else:
                return agent.predict(state, questions)


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------


def run_server(
    model: str,
    subfolder: str | None,
    device: str | None,
    router: bool,
    lang: str | None,
    port: int,
    idle_timeout: int,
    daemon_file: Path,
) -> None:
    """Load model, warm up, then serve. Blocks until shutdown."""
    # Load agent (same logic as cli._init_agent but simplified for daemon)
    print(
        f"[daemon] loading model {model!r} subfolder={subfolder!r} device={device!r} router={router} lang={lang!r} ...",
        file=sys.stderr,
    )
    t0 = time.time()
    agent = None
    if router:
        from laya import Router

        agent = Router(preload=True, device=device)
    else:
        import laya

        agent = laya.load(model, device=device, subfolder=subfolder) if subfolder else laya.load(model, device=device)

    # Warmup
    try:
        dummy_q = {"_warmup": {"type": "noul", "instructions": "Is this warmup text relevant?"}}
        if router and lang:
            agent.predict("warmup text for kernel compilation", dummy_q, lang=lang)
        elif router:
            agent.predict("warmup text for kernel compilation", dummy_q)
        else:
            agent.predict("warmup text for kernel compilation", dummy_q)
    except Exception as e:
        print(f"[daemon] warmup failed (ignored): {e}", file=sys.stderr)

    dt = time.time() - t0
    print(f"[daemon] model ready in {dt:.1f}s (device={getattr(agent, 'device', '?')})", file=sys.stderr)

    # Setup handler shared state
    DaemonHandler.agent = agent
    DaemonHandler.lock = threading.Lock()
    DaemonHandler.loaded_at = time.time()
    DaemonHandler.requests_served = 0
    DaemonHandler.last_request = time.time()
    DaemonHandler.idle_timeout = idle_timeout
    DaemonHandler.config = {
        "model": model,
        "subfolder": subfolder,
        "device": device,
        "router": router,
        "lang": lang,
    }
    DaemonHandler.shutdown_flag = threading.Event()

    # Bind to 127.0.0.1 only
    server = ThreadingHTTPServer(("127.0.0.1", port), DaemonHandler)
    # allow reuse
    server.daemon_threads = True
    actual_port = server.server_address[1]

    # Write daemon file
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "pid": os.getpid(),
        "port": actual_port,
        "model": model,
        "subfolder": subfolder,
        "device": device,
        "router": router,
        "lang": lang,
        "started_at": DaemonHandler.loaded_at,
        "config_hash": config_hash(model, subfolder, device, router, lang),
    }
    # atomic write
    tmp = daemon_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=2))
    tmp.rename(daemon_file)
    print(
        f"[daemon] listening on 127.0.0.1:{actual_port} pid={os.getpid()} file={daemon_file} idle_timeout={idle_timeout}s",
        file=sys.stderr,
    )

    # Idle watcher thread
    def _idle_watcher():
        while not DaemonHandler.shutdown_flag.is_set():
            time.sleep(1)
            if idle_timeout and idle_timeout > 0:
                idle = time.time() - DaemonHandler.last_request
                if idle >= idle_timeout:
                    print(f"[daemon] idle timeout {idle:.0f}s >= {idle_timeout}s, shutting down", file=sys.stderr)
                    DaemonHandler.shutdown_flag.set()
                    try:
                        server.shutdown()
                    except Exception:
                        pass
                    break

    watcher = threading.Thread(target=_idle_watcher, daemon=True)
    watcher.start()

    # Signal handling
    def _handle_sig(signum, frame):
        print(f"[daemon] signal {signum} received, shutting down", file=sys.stderr)
        DaemonHandler.shutdown_flag.set()
        try:
            server.shutdown()
        except Exception:
            pass

    try:
        signal.signal(signal.SIGTERM, _handle_sig)
        signal.signal(signal.SIGINT, _handle_sig)
    except Exception:
        pass  # not main thread?

    try:
        server.serve_forever()
    finally:
        # cleanup daemon file if we own it
        try:
            if daemon_file.exists():
                # ensure pid matches
                data = json.loads(daemon_file.read_text())
                if data.get("pid") == os.getpid():
                    daemon_file.unlink()
                    print(f"[daemon] removed daemon file {daemon_file}", file=sys.stderr)
        except Exception:
            pass
        print("[daemon] stopped", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers for client + management
# ---------------------------------------------------------------------------


def _http_get_status(port: int, timeout: float = 1.0) -> dict | None:
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", "/status")
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = resp.read()
        return json.loads(data.decode())
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _http_post_predict(port: int, payload: dict, timeout: float = 30.0) -> dict | None:
    import http.client

    body = json.dumps(payload, ensure_ascii=False).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/predict",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            return None
        return json.loads(data.decode())
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _http_post_shutdown(port: int, timeout: float = 2.0) -> bool:
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", "/shutdown", body=b"{}", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        return resp.status == 200
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def is_daemon_alive(daemon_file: Path) -> tuple[bool, dict | None, int | None]:
    """Check if daemon file points to live daemon. Returns (alive, status_json, port)."""
    if not daemon_file.exists():
        return False, None, None
    try:
        info = json.loads(daemon_file.read_text())
    except Exception:
        return False, None, None
    port = info.get("port")
    pid = info.get("pid")
    if not port or not pid:
        return False, None, None
    # check pid alive (optional, but status check is authoritative)
    try:
        os.kill(pid, 0)
    except OSError:
        # stale pid, but still try http (maybe pid reused)
        pass
    status = _http_get_status(port, timeout=1.0)
    if status is None:
        return False, None, port
    return True, status, port


# ---------------------------------------------------------------------------
# Client helpers (used by predict/classify/evaluate)
# ---------------------------------------------------------------------------


def daemon_port_for_args(args) -> int | None:
    """Return live daemon port for given CLI args config, or None."""
    if getattr(args, "no_daemon", False):
        return None
    model = getattr(args, "model", "convaiinnovations/laya")
    subfolder = getattr(args, "subfolder", None)
    device = getattr(args, "device", None)
    router = bool(getattr(args, "router", False))
    lang = getattr(args, "lang", None)
    daemon_file = daemon_file_for(model, subfolder, device, router, lang)
    alive, status, port = is_daemon_alive(daemon_file)
    if alive and port:
        return port
    # stale file? clean up silently for next fallback
    if daemon_file.exists() and not alive:
        try:
            # only remove if status failed (stale)
            daemon_file.unlink()
        except Exception:
            pass
    return None


def daemon_predict_via_http(
    port: int,
    state: Any,
    questions: dict[str, Any],
    lang: str | None = None,
    shortlist_k: int | None = None,
    timeout: float = 30.0,
) -> dict | None:
    payload: dict[str, Any] = {"state": state, "questions": questions}
    if lang is not None:
        payload["lang"] = lang
    if shortlist_k is not None:
        payload["shortlist_k"] = shortlist_k
    return _http_post_predict(port, payload, timeout=timeout)
