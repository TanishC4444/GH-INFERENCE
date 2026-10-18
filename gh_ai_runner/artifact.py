import io
import json
import time
import zipfile

import requests

from .integrity import text_sha256
from .repo import API, _headers
from .types import (
    InferenceResult,
    IntegrityError,
    IntegrityInfo,
    ResourceUsage,
    TokenUsage,
)


def _download_result(
    token,
    username,
    repo_name,
    run_id,
    job_id,
    correlation_id,
    verbose,
    timeout=60,
    expected_request_sha256=None,
    http_timeout=30,
):
    deadline = time.monotonic() + timeout
    target = None
    artifact_name = f"ai-output-{correlation_id}"
    while time.monotonic() < deadline:
        response = requests.get(
            f"{API}/repos/{username}/{repo_name}/actions/runs/{run_id}/artifacts",
            headers=_headers(token),
            timeout=min(http_timeout, max(0.001, deadline - time.monotonic())),
        )
        response.raise_for_status()
        target = next(
            (item for item in response.json().get("artifacts", []) if item["name"] == artifact_name),
            None,
        )
        if target:
            break
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    if not target:
        raise TimeoutError(f"No {artifact_name} artifact found before the deadline")

    if time.monotonic() >= deadline:
        raise TimeoutError("Timed out before downloading the result artifact")
    response = requests.get(
        f"{API}/repos/{username}/{repo_name}/actions/artifacts/{target['id']}/zip",
        headers=_headers(token),
        allow_redirects=True,
        timeout=min(http_timeout, max(0.001, deadline - time.monotonic())),
    )
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = "result.json" if "result.json" in archive.namelist() else None
        if not name:
            raise RuntimeError(f"result.json not found in artifact. Contents: {archive.namelist()}")
        data = json.loads(archive.read(name).decode("utf-8"))

    if data.get("job_id") != job_id or data.get("correlation_id") != correlation_id:
        raise IntegrityError("Result identity does not match the requested job", run_id=run_id)
    integrity = data.get("integrity") or {}
    if integrity.get("algorithm") != "sha256":
        raise IntegrityError("Result uses an unsupported integrity algorithm", run_id=run_id)
    actual_output_sha256 = text_sha256(data.get("output", ""))
    if integrity.get("output_sha256") != actual_output_sha256:
        raise IntegrityError("Result output checksum verification failed", run_id=run_id)
    if expected_request_sha256 and integrity.get("request_sha256") != expected_request_sha256:
        raise IntegrityError("Result request checksum verification failed", run_id=run_id)
    if "agent" in data:
        actual_agent_sha256 = text_sha256(json.dumps(data["agent"], sort_keys=True, separators=(",", ":")))
        if integrity.get("agent_sha256") != actual_agent_sha256:
            raise IntegrityError("Agent result checksum verification failed", run_id=run_id)

    if data.get("error"):
        raise IntegrityError("Worker reported an unsuccessful inference", run_id=run_id)

    usage = data.get("usage") or {}
    resources = data.get("resources") or {}
    return InferenceResult(
        job_id=job_id,
        correlation_id=correlation_id,
        run_id=run_id,
        output=data.get("output", ""),
        model=data.get("model", ""),
        provider=data.get("provider", "local"),
        usage=TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        ),
        resources=ResourceUsage(**{
            field: resources.get(field, default)
            for field, default in {
                "cpu_count": 0,
                "memory_total_bytes": 0,
                "memory_available_bytes": 0,
                "disk_total_bytes": 0,
                "disk_free_bytes": 0,
                "peak_memory_bytes": 0,
                "duration_seconds": 0.0,
                "runner_os": "",
            }.items()
        }),
        integrity=IntegrityInfo(
            algorithm=integrity.get("algorithm", "sha256"),
            request_sha256=integrity.get("request_sha256", ""),
            output_sha256=integrity.get("output_sha256", ""),
        ),
        raw=data,
    )
