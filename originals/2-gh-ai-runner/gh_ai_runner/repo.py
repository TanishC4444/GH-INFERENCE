import base64
import hashlib
import time

import requests

from .logger import _log
from .runner import INFERENCE_SCRIPT, WORKFLOW_YAML

API = "https://api.github.com"


def _headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_username(token):
    r = requests.get(f"{API}/user", headers=_headers(token))
    r.raise_for_status()
    return r.json()["login"]


def _repo_exists(token, username, repo_name):
    return requests.get(
        f"{API}/repos/{username}/{repo_name}", headers=_headers(token)
    ).status_code == 200


def _get_remote_sha256(token, username, repo_name, path):
    """Return SHA-256 of a file's content on GitHub, or None if it doesn't exist."""
    url = f"{API}/repos/{username}/{repo_name}/contents/{path}"
    r   = requests.get(url, headers=_headers(token))
    if r.status_code != 200:
        return None
    raw = base64.b64decode(r.json()["content"].replace("\n", ""))
    return hashlib.sha256(raw).hexdigest()


def _commit_file(token, username, repo_name, path, content, message):
    url      = f"{API}/repos/{username}/{repo_name}/contents/{path}"
    existing = requests.get(url, headers=_headers(token))
    sha      = existing.json().get("sha") if existing.status_code == 200 else None
    body     = {"message": message, "content": base64.b64encode(content.encode()).decode()}
    if sha:
        body["sha"] = sha
    requests.put(url, headers=_headers(token), json=body).raise_for_status()


def _wait_for_workflow(token, username, repo_name, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        r         = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/workflows",
            headers=_headers(token),
        )
        workflows = r.json().get("workflows", [])
        if any(w["path"] == ".github/workflows/inference.yml" for w in workflows):
            return
        time.sleep(3)
    raise TimeoutError("Workflow never registered.")


def _ensure_repo(token, username, repo_name, verbose):
    if _repo_exists(token, username, repo_name):
        _log("Repo ready", verbose=verbose)
        return

    t = time.time()
    _log("Creating repo...", verbose=verbose)
    r = requests.post(f"{API}/user/repos", headers=_headers(token), json={
        "name": repo_name, "private": False, "auto_init": True,
        "description": "gh-ai-runner inference repo",
    })
    r.raise_for_status()
    time.sleep(2)

    _commit_file(token, username, repo_name,
                 "run_inference.py", INFERENCE_SCRIPT, "Add inference script")
    _commit_file(token, username, repo_name,
                 ".github/workflows/inference.yml", WORKFLOW_YAML, "Add inference workflow")

    _wait_for_workflow(token, username, repo_name)
    _log("Repo created and ready", since=t, verbose=verbose)


def _sync_files(token, username, repo_name, verbose):
    """Only commit files that have actually changed — skips both if already up to date."""
    local_script_hash   = hashlib.sha256(INFERENCE_SCRIPT.encode()).hexdigest()
    local_workflow_hash = hashlib.sha256(WORKFLOW_YAML.encode()).hexdigest()

    remote_script_hash   = _get_remote_sha256(token, username, repo_name, "run_inference.py")
    remote_workflow_hash = _get_remote_sha256(token, username, repo_name, ".github/workflows/inference.yml")

    script_changed   = remote_script_hash   != local_script_hash
    workflow_changed = remote_workflow_hash != local_workflow_hash

    if not script_changed and not workflow_changed:
        _log("Runner files up to date, skipping sync", verbose=verbose)
        return

    if script_changed:
        _log("Syncing inference script...", verbose=verbose)
        _commit_file(token, username, repo_name,
                     "run_inference.py", INFERENCE_SCRIPT, "Sync inference script")

    if workflow_changed:
        _log("Syncing workflow...", verbose=verbose)
        _commit_file(token, username, repo_name,
                     ".github/workflows/inference.yml", WORKFLOW_YAML, "Sync workflow")

    _log("Runner files synced", verbose=verbose)