#!/usr/bin/env python3
"""Convenience launcher for Hilton's AI Crypto Trading — double-click or run to start."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

PID_FILE = Path(__file__).resolve().parent / "data" / "aicryptotrading.pid"


def _kill_existing() -> None:
    if not PID_FILE.exists():
        return
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except OSError:
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    except (ValueError, OSError):
        pass
    PID_FILE.unlink(missing_ok=True)


def _write_pid(pid: int) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Hilton's AI Crypto Trading.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind (default 8081, distinct from AITrading's 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    return parser.parse_args()


def wait_and_open_browser(port: int, timeout: int = 30, use_https: bool = False):
    scheme = "https" if use_https else "http"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                webbrowser.open(f"{scheme}://localhost:{port}")
                return
        except OSError:
            time.sleep(0.5)


def main() -> int:
    root = Path(__file__).resolve().parent
    args = parse_args()

    from dotenv import load_dotenv
    load_dotenv(root / ".env")
    port = int(os.getenv("PORT", args.port))

    _kill_existing()

    cmd = [sys.executable, "-m", "uvicorn", "web.app:app", "--host", args.host, "--port", str(port)]
    ssl_certfile = os.getenv("SSL_CERTFILE")
    ssl_keyfile = os.getenv("SSL_KEYFILE")
    if ssl_certfile and ssl_keyfile:
        cmd += ["--ssl-certfile", ssl_certfile, "--ssl-keyfile", ssl_keyfile]
    proc = subprocess.Popen(cmd)
    _write_pid(proc.pid)
    if not args.no_browser:
        threading.Thread(
            target=wait_and_open_browser,
            args=(port,), kwargs={"use_https": bool(ssl_certfile and ssl_keyfile)},
            daemon=True,
        ).start()

    def handle_signal(sig: int, frame: object) -> None:
        del sig, frame
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        PID_FILE.unlink(missing_ok=True)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        return proc.wait()
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)
        return 0
    finally:
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
