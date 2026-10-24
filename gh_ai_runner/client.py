from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from .artifact import _download_result
from .batch import BatchJob, ComparisonJob
from .job import AIJob
from .logger import _log
from .limits import Limits, validate_options
from .metadata import DEFAULT_METADATA_PATH, MetadataStore, now_iso, result_fields
from .models import MODELS, RUNNER_DISK_GB, RUNNER_RAM_GB, model_resources
from .polling import _find_run, _wait_for_completion
from .repo import (
    API,
    _ensure_repo,
    _get_default_branch,
    _get_username,
    _headers,
    _set_actions_secret,
    _sync_files,
)
from .types import (
    InferenceRequest,
    InferenceResult,
    JobRecord,
    JobStatus,
    ProviderConfig,
    RetryPolicy,
)
from .validation import _validate, validate_text
from .hub import resolve_model


class GitHubAIRunner:
    """Client for dispatching and managing ephemeral GitHub Actions AI jobs."""

    def __init__(
        self,
        github_token: Optional[str] = None,
        *,
        repo_name: str = "ai-inference-runner",
        verbose: bool = True,
        private: bool = True,
        runner_label: str = "ubuntu-latest",
        runner_ram_gb: float = RUNNER_RAM_GB,
        runner_disk_gb: float = RUNNER_DISK_GB,
        run_discovery_timeout: float = 180,
        completion_timeout: float = 1800,
        poll_interval: float = 8,
        http_timeout: float = 30,
        artifact_timeout: float = 60,
        workflow_registration_timeout: float = 60,
        limits: Optional[Limits] = None,
        metadata_path: Optional[str | Path] = DEFAULT_METADATA_PATH,
    ) -> None:
        github_token = github_token or os.environ.get("GH_AI_TOKEN")
        if not github_token:
            raise ValueError("Set GH_AI_TOKEN or pass github_token")
        for name, value in (("run_discovery_timeout", run_discovery_timeout),
                            ("completion_timeout", completion_timeout), ("poll_interval", poll_interval),
                            ("runner_ram_gb", runner_ram_gb), ("runner_disk_gb", runner_disk_gb),
                            ("http_timeout", http_timeout), ("artifact_timeout", artifact_timeout),
                            ("workflow_registration_timeout", workflow_registration_timeout)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(runner_label, str) or not runner_label.strip():
            raise ValueError("runner_label must be a nonempty Actions runner label")
        if type(private) is not bool:
            raise ValueError("private must be a bool")
        self.private = private
        if limits is not None and not isinstance(limits, Limits):
            raise TypeError("limits must be a Limits instance")
        self.limits = limits if limits is not None else Limits()
        self.runner_label = runner_label
        self.runner_ram_gb = runner_ram_gb
        self.runner_disk_gb = runner_disk_gb
        self.github_token = github_token
        self.repo_name = repo_name
        self.verbose = verbose
        self.run_discovery_timeout = run_discovery_timeout
        self.completion_timeout = completion_timeout
        self.poll_interval = poll_interval
        self.http_timeout, self.artifact_timeout = http_timeout, artifact_timeout
        self.workflow_registration_timeout = workflow_registration_timeout
        self._username: Optional[str] = None
        self._default_branch: Optional[str] = None
        self._prepared = False
        self._prepare_lock = threading.Lock()
        try:
            self._metadata = MetadataStore(metadata_path) if metadata_path is not None else None
        except (OSError, sqlite3.Error):
            self._metadata = None
            _log("Local history unavailable; continuing without persistence", verbose=verbose)

    @property
    def username(self) -> str:
        if self._username is None:
            self._username = _get_username(self.github_token, http_timeout=self.http_timeout)
        return self._username

    def prepare(self) -> None:
        if self._prepared:
            return
        with self._prepare_lock:
            if self._prepared:
                return
            _ensure_repo(self.github_token, self.username, self.repo_name, self.verbose,
                         private=self.private, http_timeout=self.http_timeout)
            _sync_files(self.github_token, self.username, self.repo_name, self.verbose,
                        http_timeout=self.http_timeout, registration_timeout=self.workflow_registration_timeout)
            self._default_branch = _get_default_branch(
                self.github_token, self.username, self.repo_name, http_timeout=self.http_timeout
            )
            self._prepared = True

    @property
    def default_branch(self) -> str:
        if self._default_branch is None:
            self._default_branch = _get_default_branch(
                self.github_token, self.username, self.repo_name, http_timeout=self.http_timeout)
        return self._default_branch

    def configure_provider(self, provider: ProviderConfig, api_key: str) -> None:
        """Encrypt and store a BYOK key as a GitHub Actions repository secret."""
        if not api_key:
            raise ValueError("api_key cannot be empty")
        self.prepare()
        _set_actions_secret(
            self.github_token,
            self.username,
            self.repo_name,
            provider.secret_name,
            api_key,
            http_timeout=self.http_timeout,
        )
        _log(f"Configured provider secret {provider.secret_name}", verbose=self.verbose)

    def submit(
        self,
        prompt: str,
        *,
        model: str = "tinyllama",
        system: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.7,
        cache: bool = True,
        n_ctx: Optional[int] = None,
        provider: Optional[ProviderConfig] = None,
        retry_policy: Optional[RetryPolicy] = None,
        agent_config: Optional[dict] = None,
        limits: Optional[Limits] = None,
        model_options: Optional[dict] = None,
        generation_options: Optional[dict] = None,
        _batch_id: Optional[str] = None,
        quantization: Optional[str] = None,
        filename: Optional[str] = None,
        revision: str = "main",
    ) -> AIJob:
        limits = limits if limits is not None else self.limits
        if not isinstance(limits, Limits):
            raise TypeError("limits must be a Limits instance")
        validate_options(model_options, generation_options)
        validate_text(prompt, system, cache, limits)
        if provider is not None and not isinstance(provider, ProviderConfig):
            raise TypeError("provider must be a ProviderConfig")
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a RetryPolicy")
        if agent_config is not None and not isinstance(agent_config, dict):
            raise TypeError("agent_config must be a dict")
        if provider is None:
            model = resolve_model(model, quantization, filename=filename, revision=revision)
        _validate(model if provider is None else None, max_tokens, temperature, n_ctx,
                  runner_ram_gb=self.runner_ram_gb, runner_disk_gb=self.runner_disk_gb, cache=cache, limits=limits)

        request = {
            "prompt": prompt,
            "system": system,
            "model": provider.model if provider else model,
            "local_model": model,
            "cache": cache,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_ctx": n_ctx,
            "provider": provider,
            "limits": limits.to_dict(),
            "model_options": dict(model_options or {}),
            "generation_options": dict(generation_options or {}),
        }
        if agent_config is not None:
            if provider is not None:
                raise ValueError("Agents use quantized models on GitHub workers, not hosted providers")
            request["agent"] = agent_config
        if provider is None and "files" in MODELS[model]:
            request["model_spec"] = MODELS[model]
        job = AIJob(
            self,
            uuid.uuid4().hex,
            request,
            retry_policy or RetryPolicy(),
            batch_id=_batch_id,
        )
        # Reject oversized inputs before provisioning a repository.
        self._workflow_inputs(job)
        self.prepare()
        try:
            self._dispatch(job)
        except requests.RequestException as error:
            # A timed-out POST may already have dispatched. Preserve identity for recovery.
            error.job = job
            self._record_job(job, JobStatus.FAILED, error=type(error).__name__)
            raise
        return job

    def submit_batch(self, requests_: list[InferenceRequest]) -> BatchJob:
        """Dispatch a collection of independent requests and return one batch handle."""
        if not requests_:
            raise ValueError("requests_ must contain at least one request")
        for request in requests_:
            if not isinstance(request, InferenceRequest):
                raise TypeError("each batch item must be an InferenceRequest")
            policy = request.limits if request.limits is not None else self.limits
            if not isinstance(policy, Limits):
                raise TypeError("limits must be a Limits instance")
            validate_options(request.model_options, request.generation_options)
            validate_text(request.prompt, request.system, request.cache, policy)
            if request.provider is not None and not isinstance(request.provider, ProviderConfig):
                raise TypeError("provider must be a ProviderConfig")
            if request.retry_policy is not None and not isinstance(request.retry_policy, RetryPolicy):
                raise TypeError("retry_policy must be a RetryPolicy")
            resolved = resolve_model(request.model) if request.provider is None else None
            _validate(
                resolved,
                request.max_tokens,
                request.temperature,
                request.n_ctx,
                runner_ram_gb=self.runner_ram_gb,
                runner_disk_gb=self.runner_disk_gb,
                cache=request.cache,
                limits=policy,
            )
        batch_id = uuid.uuid4().hex
        jobs = []
        for request in requests_:
            jobs.append(
                self.submit(
                    request.prompt,
                    model=request.model,
                    system=request.system,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    cache=request.cache,
                    n_ctx=request.n_ctx,
                    provider=request.provider,
                    retry_policy=request.retry_policy,
                    limits=request.limits,
                    model_options=request.model_options,
                    generation_options=request.generation_options,
                    _batch_id=batch_id,
                )
            )
        return BatchJob(jobs, batch_id=batch_id)

    def compare(self, prompt: str, models: list[str], **kwargs) -> ComparisonJob:
        """Dispatch the same prompt to distinct local models for side-by-side results."""
        if len(models) < 2:
            raise ValueError("compare requires at least two models")
        if len(set(models)) != len(models):
            raise ValueError("comparison models must be unique")
        if kwargs.get("provider") is not None:
            raise ValueError("compare currently supports local models; use submit_batch for providers")
        for model in models:
            resolve_model(model)
        requests_ = [InferenceRequest(prompt=prompt, model=model, **kwargs) for model in models]
        batch = self.submit_batch(requests_)
        return ComparisonJob(batch.jobs, models, batch_id=batch.batch_id)

    def list_jobs(self, limit: int = 100) -> list[JobRecord]:
        if self._metadata is None:
            return []
        return self._metadata.list(limit)

    def resume(self, job_id: str) -> AIJob:
        if self._metadata is None:
            raise RuntimeError("metadata persistence is disabled")
        record = self._metadata.get(job_id)
        if record is None:
            raise KeyError(f"Unknown job {job_id}")
        if record.owner != self.username or record.repo_name != self.repo_name:
            raise ValueError(
                f"Job {job_id} belongs to {record.owner}/{record.repo_name}, "
                f"not {self.username}/{self.repo_name}"
            )
        return AIJob(
            self,
            record.job_id,
            {},
            RetryPolicy(),
            attempt=record.attempt,
            created_at=datetime.fromisoformat(record.created_at),
            request_digest=record.request_sha256,
            prompt_digest=record.prompt_sha256,
            batch_id=record.batch_id,
        )

    def get_job(
        self,
        job_id: str,
        *,
        attempt: int = 1,
        created_at: Optional[datetime] = None,
    ) -> AIJob:
        """Reconstruct a job handle for status checks or cancellation."""
        return AIJob(
            self,
            job_id,
            {},
            RetryPolicy(),
            attempt=attempt,
            created_at=created_at or datetime.fromtimestamp(0, timezone.utc),
        )

    def _record_job(
        self,
        job: AIJob,
        status: JobStatus,
        *,
        result: Optional[InferenceResult] = None,
        error: Optional[str] = None,
    ) -> None:
        try:
            self._persist_job(job, status, result=result, error=error)
        except (OSError, sqlite3.Error):
            # A successful remote dispatch/result must survive a local disk failure.
            _log("Could not update local job history", verbose=self.verbose)

    def _persist_job(
        self,
        job: AIJob,
        status: JobStatus,
        *,
        result: Optional[InferenceResult] = None,
        error: Optional[str] = None,
    ) -> None:
        if self._metadata is None:
            return
        existing = self._metadata.get(job.job_id, job.attempt)
        request = job.request
        model = request.get("model") or (existing.model if existing else "unknown")
        provider_config = request.get("provider")
        provider = (
            provider_config.name
            if provider_config
            else (existing.provider if existing else "local")
        )
        fields = result_fields(result)
        record = JobRecord(
            correlation_id=job.correlation_id,
            job_id=job.job_id,
            attempt=job.attempt,
            batch_id=job.batch_id,
            owner=self.username,
            repo_name=self.repo_name,
            status=status.value,
            run_id=job.run_id,
            model=model,
            provider=provider,
            prompt_sha256=job.prompt_digest or (existing.prompt_sha256 if existing else ""),
            request_sha256=job.request_digest or (existing.request_sha256 if existing else ""),
            output_sha256=fields.get("output_sha256") or (existing.output_sha256 if existing else None),
            prompt_tokens=fields.get("prompt_tokens") if result else (existing.prompt_tokens if existing else None),
            completion_tokens=fields.get("completion_tokens") if result else (existing.completion_tokens if existing else None),
            total_tokens=fields.get("total_tokens") if result else (existing.total_tokens if existing else None),
            duration_seconds=fields.get("duration_seconds") if result else (existing.duration_seconds if existing else None),
            created_at=existing.created_at if existing else job.created_at.isoformat(),
            updated_at=now_iso(),
            error=error,
        )
        self._metadata.upsert(record)

    def _workflow_inputs(self, job: AIJob) -> dict:
        provider = job.request.get("provider")
        limits = Limits(**job.request["limits"]) if "limits" in job.request else self.limits
        config = {
            "model": job.request["model"],
            "max_tokens": job.request["max_tokens"],
            "temperature": job.request["temperature"],
            "n_ctx": job.request["n_ctx"],
            "provider_name": provider.name if provider else "local",
            "provider_url": provider.base_url if provider else "",
            "runner_label": self.runner_label,
            "model_sha256": MODELS[job.request["local_model"]]["sha256"] if provider is None else "",
            "cache_weights": (
                model_resources(job.request["local_model"], job.request["n_ctx"], job.request["cache"], limits=limits)["cache_weights"]
                if provider is None else False
            ),
        }
        for key in ("limits", "model_options", "generation_options", "model_spec"):
            if key in job.request:
                config[key] = job.request[key]
        config["job_timeout_minutes"] = limits.job_timeout_minutes
        config["artifact_retention_days"] = limits.artifact_retention_days
        inputs = {
            "job_id": job.job_id,
            "attempt": str(job.attempt),
            "prompt": job.request["prompt"],
            "system": job.request["system"],
            "backend": "provider" if provider else "local",
            "local_model": job.request["local_model"],
            "cache": "true" if job.request["cache"] else "false",
            "config": json.dumps(config, separators=(",", ":")),
        }
        if "agent" in job.request:
            config["agent"] = job.request["agent"]
            inputs["config"] = json.dumps(config, separators=(",", ":"))
        size = len(json.dumps(inputs, ensure_ascii=False, separators=(",", ":")))
        if limits.max_workflow_input_chars is not None and size > limits.max_workflow_input_chars:
            raise ValueError(f"Combined workflow inputs exceed configured {limits.max_workflow_input_chars:,} character limit")
        if size > 65_535:
            raise ValueError("Combined workflow inputs exceed GitHub's 65,535 character service limit")
        return inputs

    def _dispatch(self, job: AIJob) -> None:
        inputs = self._workflow_inputs(job)
        _log(f"Dispatching job {job.job_id} attempt {job.attempt}", verbose=self.verbose)
        response = requests.post(
            f"{API}/repos/{self.username}/{self.repo_name}/actions/workflows/inference.yml/dispatches",
            headers=_headers(self.github_token),
            json={"ref": self.default_branch, "inputs": inputs},
            timeout=self.http_timeout,
        )
        response.raise_for_status()
        self._record_job(job, JobStatus.DISPATCHED)

    def _find_run(self, job: AIJob, *, wait: bool, timeout: Optional[float] = None):
        if wait and timeout is not None and timeout <= 0:
            raise TimeoutError("Run discovery deadline has elapsed")
        run = _find_run(
            self.github_token,
            self.username,
            self.repo_name,
            job.correlation_id,
            self.verbose,
            branch=self.default_branch,
            created_at=job.created_at,
            timeout=(self.run_discovery_timeout if timeout is None else timeout) if wait else 0,
            http_timeout=self.http_timeout,
        )
        if wait and run is None:
            raise TimeoutError(f"Timed out waiting for workflow run {job.correlation_id} to appear.")
        return run

    def _wait_for_completion(self, job: AIJob, timeout: Optional[float] = None):
        return _wait_for_completion(
            self.github_token,
            self.username,
            self.repo_name,
            job.run_id,
            self.verbose,
            timeout=self.completion_timeout if timeout is None else timeout,
            poll=self.poll_interval,
            http_timeout=self.http_timeout,
        )

    def _download_result(self, job: AIJob, timeout: Optional[float] = None) -> InferenceResult:
        return _download_result(
            self.github_token,
            self.username,
            self.repo_name,
            job.run_id,
            job.job_id,
            job.correlation_id,
            self.verbose,
            expected_request_sha256=job.request_digest or None,
            timeout=self.artifact_timeout if timeout is None else timeout,
            http_timeout=self.http_timeout,
        )

    def _cancel_run(self, run_id: int) -> bool:
        response = requests.post(
            f"{API}/repos/{self.username}/{self.repo_name}/actions/runs/{run_id}/cancel",
            headers=_headers(self.github_token),
            timeout=self.http_timeout,
        )
        if response.status_code == 409:
            return False
        response.raise_for_status()
        return True
