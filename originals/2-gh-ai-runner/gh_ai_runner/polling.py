import time

import requests

from .logger import _log
from .repo import API, _headers


def _snapshot_run_ids(token, username, repo_name):
    r = requests.get(
        f"{API}/repos/{username}/{repo_name}/actions/runs",
        headers=_headers(token),
        params={"per_page": 20},
    )
    return {run["id"] for run in r.json().get("workflow_runs", [])}


def _wait_for_run(token, username, repo_name, seen_ids, verbose, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)  # reduced from 4s — GitHub usually registers within 3s
        r = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs",
            headers=_headers(token),
            params={"per_page": 10},
        )
        for run in r.json().get("workflow_runs", []):
            if run["id"] not in seen_ids:
                _log(f"Runner picked up job (run #{run['id']})", verbose=verbose)
                return run["id"]
    raise TimeoutError("Timed out waiting for workflow run to appear.")


def _wait_for_completion(token, username, repo_name, run_id, verbose, timeout=900, poll=8):
    # poll reduced from 12s to 8s — shaves ~30s off a typical cached run
    start       = time.time()
    last_status = None
    while time.time() - start < timeout:
        r          = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs/{run_id}",
            headers=_headers(token),
        )
        data       = r.json()
        status     = data["status"]
        conclusion = data.get("conclusion")

        if status != last_status:
            _log(f"Runner: {status}{' -> ' + conclusion if conclusion else ''}",
                 verbose=verbose)
            last_status = status

        if status == "completed":
            if conclusion != "success":
                raise RuntimeError(
                    f"Workflow failed ({conclusion}) — "
                    f"https://github.com/{username}/{repo_name}/actions/runs/{run_id}"
                )
            return

        time.sleep(poll)
    raise TimeoutError("Timed out waiting for workflow to complete.")