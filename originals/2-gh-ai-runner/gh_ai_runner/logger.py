import time
from datetime import datetime


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _elapsed(since):
    return f"{time.time() - since:.1f}s"


def _log(msg, since=None, verbose=True):
    if not verbose:
        return
    suffix = f"  (+{_elapsed(since)})" if since else ""
    print(f"[{_ts()}] {msg}{suffix}", flush=True)