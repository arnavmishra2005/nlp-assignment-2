"""Subprocess streaming with a quiet-period heartbeat and checked exit code."""
import queue
import subprocess
import sys
import threading
import time


def run_streaming(command, log_path=None, cwd=None, heartbeat_seconds=45):
    process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=cwd)
    messages = queue.Queue()
    def reader():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    output = open(log_path, "a", encoding="utf-8", buffering=1) if log_path else None
    last_line = time.monotonic()
    try:
        while True:
            try:
                line = messages.get(timeout=1)
            except queue.Empty:
                if time.monotonic() - last_line >= heartbeat_seconds:
                    line = f"[heartbeat] subprocess PID={process.pid} is alive={process.poll() is None}; no new log output. This does not prove useful progress.\n"
                    print(line, end="", flush=True)
                    if output:
                        output.write(line)
                    last_line = time.monotonic()
                continue
            if line is None:
                break
            print(line, end="", flush=True)
            if output:
                output.write(line)
            last_line = time.monotonic()
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        return code
    finally:
        if output:
            output.close()
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
