from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

from .client import GitHubAIRunner
from .models import RUNNER_DISK_GB, RUNNER_RAM_GB
from .types import AgentTimings, InferenceResult, RetryPolicy


def ai_call(
    github_token: str,
    prompt:       str,
    model:        str   = "tinyllama",
    system:       str   = "You are a helpful assistant.",
    max_tokens:   int   = 512,
    temperature:  float = 0.7,
    cache:        bool  = True,
    n_ctx:        Optional[int] = None,
    repo_name:    str   = "ai-inference-runner",
    verbose:      bool  = True,
    *,
    private:      bool  = True,
    runner_label: str = "ubuntu-latest",
    runner_ram_gb: float = RUNNER_RAM_GB,
    runner_disk_gb: float = RUNNER_DISK_GB,
    track_time: bool = False,
    limits=None,
    model_options=None,
    generation_options=None,
) -> str | InferenceResult:
    """
    Run AI inference on GitHub Actions and return the model output.

    Required:
        github_token: Your GitHub personal access token (needs repo + workflow scopes).
        prompt:       The message / question to send to the model.

    Optional:
        model:        A key in gh_ai_runner.MODELS (default: "tinyllama").
        system:       System prompt. Default: "You are a helpful assistant."
        max_tokens:   Max tokens to generate. Default: 512; bounded by model/context.
        temperature:  Sampling temperature. Default: 0.7; bounds are editable with Limits.
        cache:        Cache model weights between runs. Default: True.
        n_ctx:        Total context tokens. Default: 8192 (tinyllama: 2048).
                      Advertised context/RAM policy can be overridden with Limits.
        repo_name:    GitHub repo to create/reuse. Default: "ai-inference-runner".
        verbose:      Print detailed logs. Default: True. Set False for silent mode.

    Returns:
        Model response as a string, or an InferenceResult with timings when
        track_time=True. Remote timing includes setup, waiting, and retrieval.

    Raises:
        ValueError: If any parameter is out of safe range.
    """
    if type(track_time) is not bool:
        raise TypeError("track_time must be a bool")
    started = time.monotonic()
    runner = GitHubAIRunner(
        github_token, repo_name=repo_name, verbose=verbose, private=private,
        runner_label=runner_label, runner_ram_gb=runner_ram_gb, runner_disk_gb=runner_disk_gb,
        limits=limits,
    )
    job = runner.submit(
        prompt,
        model=model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=cache,
        n_ctx=n_ctx,
        retry_policy=RetryPolicy(),
        model_options=model_options,
        generation_options=generation_options,
    )
    result = job.result()
    if track_time:
        elapsed = time.monotonic() - started
        return replace(result, timings=AgentTimings(total_seconds=elapsed, remote_seconds=elapsed))
    return result.output
