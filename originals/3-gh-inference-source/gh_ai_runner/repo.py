import base64
import hashlib
import time

import requests
from nacl import encoding, public

from .logger import _log
from .runner import INFERENCE_SCRIPT, WORKFLOW_YAML

API = "https://api.github.com"


def _headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _set_actions_secret(token, username, repo_name, secret_name, value, *, http_timeout=30):
    key_response = requests.get(
        f"{API}/repos/{username}/{repo_name}/actions/secrets/public-key",
        headers=_headers(token), timeout=http_timeout,
    )
    key_response.raise_for_status()
    key_data = key_response.json()
    public_key = public.PublicKey(key_data["key"].encode(), encoding.Base64Encoder())
    encrypted = public.SealedBox(public_key).encrypt(value.encode())
    response = requests.put(
        f"{API}/repos/{username}/{repo_name}/actions/secrets/{secret_name}",
        headers=_headers(token), timeout=http_timeout,
        json={
            "encrypted_value": encoding.Base64Encoder.encode(encrypted).decode(),
            "key_id": key_data["key_id"],
        },
    )
    response.raise_for_status()


def _get_username(token, *, http_timeout=30):
    r = requests.get(f"{API}/user", headers=_headers(token), timeout=http_timeout)
    r.raise_for_status()
    return r.json()["login"]


def _get_default_branch(token, username, repo_name, *, http_timeout=30):
    r = requests.get(f"{API}/repos/{username}/{repo_name}", headers=_headers(token), timeout=http_timeout)
    r.raise_for_status()
    return r.json().get("default_branch", "main")


def _repo_exists(token, username, repo_name, *, http_timeout=30):
    response = requests.get(
        f"{API}/repos/{username}/{repo_name}", headers=_headers(token), timeout=http_timeout
    )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def _get_remote_sha256(token, username, repo_name, path, *, http_timeout=30):
    """Return SHA-256 of a file's content on GitHub, or None if it doesn't exist."""
    url = f"{API}/repos/{username}/{repo_name}/contents/{path}"
    r   = requests.get(url, headers=_headers(token), timeout=http_timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    raw = base64.b64decode(r.json()["content"].replace("\n", ""))
    return hashlib.sha256(raw).hexdigest()


def _commit_file(token, username, repo_name, path, content, message, *, http_timeout=30):
    url      = f"{API}/repos/{username}/{repo_name}/contents/{path}"
    existing = requests.get(url, headers=_headers(token), timeout=http_timeout)
    if existing.status_code != 404:
        existing.raise_for_status()
    sha      = existing.json().get("sha") if existing.status_code == 200 else None
    body     = {"message": message, "content": base64.b64encode(content.encode()).decode()}
    if sha:
        body["sha"] = sha
    requests.put(url, headers=_headers(token), timeout=http_timeout, json=body).raise_for_status()


def _wait_for_workflow(token, username, repo_name, timeout=60, *, http_timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r         = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/workflows/inference.yml",
            headers=_headers(token), timeout=min(http_timeout, max(0.001, deadline - time.monotonic())),
        )
        if r.status_code != 404:
            r.raise_for_status()
            return
        time.sleep(min(3, max(0, deadline - time.monotonic())))
    raise TimeoutError("Workflow never registered.")


def _ensure_repo(token, username, repo_name, verbose, private=True, *, http_timeout=30):
    if _repo_exists(token, username, repo_name, http_timeout=http_timeout):
        _log("Repo ready", verbose=verbose)
        return

    t = time.time()
    _log("Creating repo...", verbose=verbose)
    r = requests.post(f"{API}/user/repos", headers=_headers(token), timeout=http_timeout, json={
        "name": repo_name, "private": private, "auto_init": True,
        "description": "gh-ai-runner inference repo",
    })
    r.raise_for_status()
    time.sleep(2)

    _log("Repo created", since=t, verbose=verbose)


def _sync_files(token, username, repo_name, verbose, *, http_timeout=30, registration_timeout=60):
    """Only commit files that have actually changed — skips both if already up to date."""
    local_script_hash   = hashlib.sha256(INFERENCE_SCRIPT.encode()).hexdigest()
    local_workflow_hash = hashlib.sha256(WORKFLOW_YAML.encode()).hexdigest()

    remote_script_hash   = _get_remote_sha256(token, username, repo_name, "run_inference.py", http_timeout=http_timeout)
    remote_workflow_hash = _get_remote_sha256(token, username, repo_name, ".github/workflows/inference.yml", http_timeout=http_timeout)

    script_changed   = remote_script_hash   != local_script_hash
    workflow_changed = remote_workflow_hash != local_workflow_hash

    if not script_changed and not workflow_changed:
        _log("Runner files up to date, skipping sync", verbose=verbose)
        return

    if script_changed:
        _log("Syncing inference script...", verbose=verbose)
        _commit_file(token, username, repo_name,
                     "run_inference.py", INFERENCE_SCRIPT, "Sync inference script", http_timeout=http_timeout)

    if workflow_changed:
        _log("Syncing workflow...", verbose=verbose)
        _commit_file(token, username, repo_name,
                     ".github/workflows/inference.yml", WORKFLOW_YAML, "Sync workflow", http_timeout=http_timeout)

    if workflow_changed:
        _wait_for_workflow(token, username, repo_name, timeout=registration_timeout, http_timeout=http_timeout)
    _log("Runner files synced", verbose=verbose)
