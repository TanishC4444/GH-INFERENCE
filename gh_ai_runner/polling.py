import time

import requests

from .logger import _log
from .repo import API, _headers


def _find_run(token, username, repo_name, correlation_id, verbose, branch="main", created_at=None, timeout=180, http_timeout=30):
    """Find the exact job, including runs beyond the first API page."""
    deadline = time.monotonic() + (timeout if timeout > 0 else http_timeout)
    while True:
        page = 1
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            response = requests.get(
                f"{API}/repos/{username}/{repo_name}/actions/workflows/inference.yml/runs",
                headers=_headers(token), timeout=min(http_timeout, remaining),
                params={"event": "workflow_dispatch", "branch": branch, "per_page": 100, "page": page},
            )
            response.raise_for_status()
            runs = response.json().get("workflow_runs", [])
            for run in runs:
                if run.get("display_title") == f"AI inference {correlation_id}":
                    _log(f"Runner picked up job (run #{run['id']})", verbose=verbose)
                    return run
            if len(runs) < 100:
                break
            page += 1
        if timeout <= 0:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for workflow run {correlation_id} to appear")
        time.sleep(min(2, remaining))


def _wait_for_completion(token, username, repo_name, run_id, verbose, timeout=900, poll=8, http_timeout=30):
    deadline = time.monotonic() + timeout
    last_status = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for workflow to complete; the remote job may still be running")
        response = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs/{run_id}",
            headers=_headers(token), timeout=min(http_timeout, remaining),
        )
        response.raise_for_status()
        data = response.json()
        status = data["status"]
        if status != last_status:
            _log(f"Runner: {status} ({data.get('conclusion') or 'pending'})", verbose=verbose)
            last_status = status
        if status == "completed":
            return data
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
