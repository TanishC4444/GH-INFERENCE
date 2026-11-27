import time

import requests

from .artifact import _download_output
from .logger import _elapsed, _log
from .models import MODELS, RAM_PER_1K_CTX_GB, RUNNER_RAM_GB
from .polling import _snapshot_run_ids, _wait_for_completion, _wait_for_run
from .repo import API, _ensure_repo, _get_username, _headers, _sync_files
from .validation import _validate


def ai_call(
    github_token: str,
    prompt:       str,
    model:        str   = "tinyllama",
    system:       str   = "You are a helpful assistant.",
    max_tokens:   int   = 512,
    temperature:  float = 0.7,
    cache:        bool  = True,
    n_ctx:        int   = None,
    repo_name:    str   = "ai-inference-runner",
    verbose:      bool  = True,
) -> str:
    """
    Run AI inference on GitHub Actions and return the model output.

    Required:
        github_token: Your GitHub personal access token (needs repo + workflow scopes).
        prompt:       The message / question to send to the model.

    Optional:
        model:        "tinyllama" (default) or "llama".
        system:       System prompt. Default: "You are a helpful assistant."
        max_tokens:   Max tokens to generate. Default: 512. Hard limit: 4096.
        temperature:  0.0 = deterministic, 2.0 = very creative. Default: 0.7.
        cache:        Cache model weights between runs. Default: True.
        n_ctx:        Context window size. Default: 2048 (tinyllama) / 4096 (llama). Max: 8192.
        repo_name:    GitHub repo to create/reuse. Default: "ai-inference-runner".
        verbose:      Print detailed logs. Default: True. Set False for silent mode.

    Returns:
        Model response as a string.

    Raises:
        ValueError: If any parameter is out of safe range.
    """
    system = system + " Be concise and direct. Avoid unnecessary filler."

    if model not in MODELS:
        raise ValueError(f"model must be one of: {list(MODELS.keys())}")

    _validate(model, max_tokens, temperature, n_ctx)

    call_start      = time.time()
    cfg             = MODELS[model]
    username        = _get_username(github_token)
    effective_n_ctx = n_ctx or cfg["n_ctx"]
    extra_ctx       = max(0, effective_n_ctx - cfg["n_ctx"])
    est_ram         = cfg["size_gb"] + (extra_ctx / 1000) * RAM_PER_1K_CTX_GB + 1.0

    _log("=" * 50,                                                                 verbose=verbose)
    _log("ai_call started",                                                        verbose=verbose)
    _log(f"Model      : {cfg['name']} ({cfg['size_gb']} GB GGUF)",                verbose=verbose)
    _log(f"n_ctx      : {effective_n_ctx}  |  Est. RAM: {est_ram:.1f} GB",        verbose=verbose)
    _log(f"Max tokens : {max_tokens}  |  Temp: {temperature}  |  Cache: {cache}", verbose=verbose)
    _log(f"Prompt     : {prompt[:80]}{'...' if len(prompt) > 80 else ''}",        verbose=verbose)
    _log("=" * 50,                                                                 verbose=verbose)

    _ensure_repo(github_token, username, repo_name, verbose)
    _sync_files(github_token, username, repo_name, verbose)

    seen_ids = _snapshot_run_ids(github_token, username, repo_name)

    t = time.time()
    _log("Dispatching workflow...", verbose=verbose)
    r = requests.post(
        f"{API}/repos/{username}/{repo_name}/actions/workflows/inference.yml/dispatches",
        headers=_headers(github_token),
        json={
            "ref": "main",
            "inputs": {
                "prompt":      prompt,
                "system":      system,
                "model":       model,
                "cache":       "true" if cache else "false",
                "max_tokens":  str(max_tokens),
                "temperature": str(temperature),
                "n_ctx":       str(n_ctx) if n_ctx else "",
            },
        },
    )
    r.raise_for_status()
    _log("Workflow dispatched", since=t, verbose=verbose)

    t = time.time()
    _log("Waiting for runner...", verbose=verbose)
    run_id = _wait_for_run(github_token, username, repo_name, seen_ids, verbose)

    t = time.time()
    _log("Runner working (first run ~5 min, cached ~1-2 min)...", verbose=verbose)
    _wait_for_completion(github_token, username, repo_name, run_id, verbose)
    _log("Runner finished", since=t, verbose=verbose)

    t = time.time()
    _log("Downloading output...", verbose=verbose)
    output = _download_output(github_token, username, repo_name, run_id, verbose)
    _log("Output downloaded", since=t, verbose=verbose)

    _log("=" * 50,                               verbose=verbose)
    _log(f"Total time : {_elapsed(call_start)}", verbose=verbose)
    _log("=" * 50,                               verbose=verbose)

    return output