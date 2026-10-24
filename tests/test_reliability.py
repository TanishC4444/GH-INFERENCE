"""Regressions for the core dispatch, worker, and retrieval lifecycle."""
import io
import json
import sqlite3
import sys
import types
import zipfile
from unittest.mock import Mock, patch

import pytest
import requests

from gh_ai_runner import (
    GitHubAIRunner,
    IntegrityError,
    JobStatus,
    ProviderConfig,
    RetryPolicy,
)
from gh_ai_runner.artifact import _download_result
from gh_ai_runner.integrity import text_sha256
from gh_ai_runner.job import AIJob
from gh_ai_runner.metadata import MetadataStore
from gh_ai_runner.models import MODELS
from gh_ai_runner.polling import _find_run, _wait_for_completion
from gh_ai_runner.repo import (
    _ensure_repo,
    _get_remote_sha256,
    _repo_exists,
    _sync_files,
)
from gh_ai_runner.runner import INFERENCE_SCRIPT, WORKFLOW_YAML
from gh_ai_runner.validation import _validate


def response(payload=None, status=200, content=b""):
    result = Mock(status_code=status, content=content)
    result.json.return_value = payload or {}
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(response=result)
    return result


def runner():
    client = GitHubAIRunner("test", metadata_path=None, verbose=False)
    client._prepared = True
    client._username = "tester"
    client._default_branch = "main"
    return client


@pytest.mark.parametrize("model", [None, "tinyllama"])
@pytest.mark.parametrize("tokens,temp,context", [
    (0, .7, None), (8192, .7, None), (1.5, .7, None), (True, .7, None),
    (1, float("nan"), None), (1, float("inf"), None), (1, True, None),
    (1, .7, 0), (1, .7, -1), (1, .7, 1.5),
    (1, .7, True), (16, .7, 16),
])
def test_invalid_limits_rejected(model, tokens, temp, context):
    with pytest.raises(ValueError):
        _validate(model, tokens, temp, context)


def test_bad_requests_do_not_prepare_repository():
    client = runner()
    client.prepare = Mock()
    for kwargs in ({"prompt": " "}, {"prompt": "x", "n_ctx": 100000},
                   {"prompt": "x" * 60001}, {"prompt": "x", "system": None}):
        with pytest.raises((ValueError, TypeError)):
            client.submit(**kwargs)
    client.prepare.assert_not_called()


def test_system_message_is_unchanged():
    client = runner()
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)) as post:
        job = client.submit("hello", system="Reply only in JSON.")
    assert job.request["system"] == "Reply only in JSON."
    assert post.call_args.kwargs["json"]["inputs"]["system"] == "Reply only in JSON."


@pytest.mark.parametrize("private", [True, False])
def test_repository_visibility_is_explicit(private):
    with patch("gh_ai_runner.repo._repo_exists", return_value=False), patch(
        "gh_ai_runner.repo.requests.post", return_value=response(status=201)
    ) as post, patch("gh_ai_runner.repo._commit_file"), patch(
        "gh_ai_runner.repo._wait_for_workflow"
    ), patch("gh_ai_runner.repo.time.sleep"):
        _ensure_repo("token", "tester", "inference", False, private=private)
    assert post.call_args.kwargs["json"]["private"] is private
    assert GitHubAIRunner("token", metadata_path=None).private is True


@pytest.mark.parametrize("function,args", [
    (_repo_exists, ("t", "u", "r")),
    (_get_remote_sha256, ("t", "u", "r", "script.py")),
])
def test_permission_errors_do_not_trigger_provisioning(function, args):
    with patch("gh_ai_runner.repo.requests.get", return_value=response(status=403)) as get, pytest.raises(requests.HTTPError):
        function(*args)
    assert get.call_args.kwargs["timeout"] > 0


def test_unchanged_worker_files_do_not_commit():
    import hashlib
    with patch("gh_ai_runner.repo._get_remote_sha256", side_effect=[
        hashlib.sha256(INFERENCE_SCRIPT.encode()).hexdigest(),
        hashlib.sha256(WORKFLOW_YAML.encode()).hexdigest(),
    ]), patch("gh_ai_runner.repo._commit_file") as commit:
        _sync_files("t", "u", "r", False)
    commit.assert_not_called()


def test_run_discovery_searches_older_pages():
    pages = [response({"workflow_runs": [{"id": i, "display_title": "other"} for i in range(100)]}),
             response({"workflow_runs": [{"id": 999, "display_title": "AI inference wanted-1"}]})]
    with patch("gh_ai_runner.polling.requests.get", side_effect=pages) as get:
        assert _find_run("t", "u", "r", "wanted-1", False, timeout=0)["id"] == 999
    assert get.call_args.kwargs["params"]["page"] == 2


def test_completion_api_errors_are_reported():
    with patch("gh_ai_runner.polling.requests.get", return_value=response(status=403)), pytest.raises(requests.HTTPError):
        _wait_for_completion("t", "u", "r", 1, False)


def test_timeout_does_not_launch_duplicate_job():
    client = runner()
    client._find_run = Mock(return_value={"id": 1})
    client._wait_for_completion = Mock(side_effect=TimeoutError())
    client._dispatch = Mock()
    job = AIJob(client, "job", {"prompt": "test"}, RetryPolicy(max_retries=2))
    with pytest.raises(TimeoutError):
        job.wait()
    client._dispatch.assert_not_called()


def test_retry_does_not_dispatch_after_deadline():
    client = runner()
    client._find_run = Mock(return_value={"id": 1})
    client._wait_for_completion = Mock(return_value={"conclusion": "failure"})
    client._dispatch = Mock()
    job = AIJob(client, "job", {"prompt": "test"}, RetryPolicy(max_retries=2))
    with patch("gh_ai_runner.job.time.monotonic", side_effect=[0, 0, 0, 2, 2]), patch(
        "gh_ai_runner.job.time.sleep"
    ), pytest.raises(TimeoutError):
        job.wait(timeout=1)
    client._dispatch.assert_not_called()


def test_result_shares_deadline_with_download():
    client = runner()
    client._download_result = Mock(return_value=Mock(output="ok"))
    job = AIJob(client, "job", {}, RetryPolicy())
    job.wait = Mock()
    with patch("gh_ai_runner.job.time.monotonic", side_effect=[10, 12]):
        assert job.result(timeout=5).output == "ok"
    client._download_result.assert_called_once_with(job, timeout=3)


def test_metadata_failure_does_not_lose_dispatched_job():
    client = runner()
    client._metadata = Mock()
    client._metadata.get.side_effect = sqlite3.OperationalError("disk full")
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)):
        assert client.submit("hello").job_id


def test_metadata_does_not_chmod_existing_parent(tmp_path):
    tmp_path.chmod(0o755)
    MetadataStore(tmp_path / "jobs.sqlite3")
    assert tmp_path.stat().st_mode & 0o777 == 0o755


def test_metadata_does_not_persist_exception_contents(tmp_path):
    client = runner()
    client._metadata = MetadataStore(tmp_path / "jobs.sqlite3")
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)):
        job = client.submit("hello")
    job.wait = Mock()
    client._download_result = Mock(side_effect=ValueError("private-output-and-secret"))
    with pytest.raises(ValueError):
        job.result()
    assert client.list_jobs()[0].error == "ValueError"
    assert b"private-output-and-secret" not in (tmp_path / "jobs.sqlite3").read_bytes()


def test_artifact_without_integrity_is_rejected():
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("result.json", json.dumps({"job_id": "j", "correlation_id": "j-1", "output": "x"}))
    with patch("gh_ai_runner.artifact.requests.get", side_effect=[
        response({"artifacts": [{"id": 1, "name": "ai-output-j-1"}]}),
        response(content=zipped.getvalue()),
    ]), pytest.raises(IntegrityError):
        _download_result("t", "u", "r", 1, "j", "j-1", False)


@pytest.mark.parametrize("url", ["https://", "http://example.com/v1", "https://key@example.com/v1",
                                 "https://example.com/v1?key=secret", "https://exam\nple.com/v1"])
def test_provider_url_rejects_invalid_or_embedded_credentials(url):
    with pytest.raises(ValueError):
        ProviderConfig("test", "model", url)


def test_local_worker_round_trip(tmp_path, monkeypatch):
    """Run the actual embedded script, then parse its ZIP through the client."""
    client = runner()
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)) as post:
        job = client.submit("Explain a stack.", system="Be brief.", model="tinyllama")
    inputs = post.call_args.kwargs["json"]["inputs"]
    for key in ("job_id", "attempt", "prompt", "system", "backend", "local_model", "cache"):
        monkeypatch.setenv(key.upper(), inputs[key])
    monkeypatch.setenv("CONFIG_JSON", inputs["config"])
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "model_cache"
    cache.mkdir()
    (cache / MODELS["tinyllama"]["filename"]).write_bytes(b"fake-model")
    fake_llm = Mock()
    fake_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "A last-in, first-out collection."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }
    llama = Mock(return_value=fake_llm)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=llama))
    namespace = {}
    exec(compile(INFERENCE_SCRIPT, "run_inference.py", "exec"), namespace)  # noqa: S102
    assert set(namespace["MODEL_MAP"]) == set(MODELS)
    assert fake_llm.create_chat_completion.call_args.kwargs["messages"][0]["content"] == "Be brief."
    assert llama.call_args.kwargs["n_ctx"] == 2048
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.write(tmp_path / "result.json", "result.json")
    with patch("gh_ai_runner.artifact.requests.get", side_effect=[
        response({"artifacts": [{"id": 1, "name": f"ai-output-{job.correlation_id}"}]}),
        response(content=zipped.getvalue()),
    ]):
        result = _download_result("t", "u", "r", 1, job.job_id, job.correlation_id, False,
                                  expected_request_sha256=job.request_digest)
    assert result.output == "A last-in, first-out collection."
    assert result.usage.total_tokens == 18
    assert result.integrity.output_sha256 == text_sha256(result.output)


def test_status_terminal_property():
    assert JobStatus.COMPLETED.is_terminal
    assert JobStatus.CANCELLED.is_terminal
    assert JobStatus.FAILED.is_terminal
    assert not JobStatus.IN_PROGRESS.is_terminal


def test_expired_discovery_budget_does_not_make_network_request():
    client = runner()
    job = AIJob(client, "job", {}, RetryPolicy())
    with patch("gh_ai_runner.polling.requests.get") as get, pytest.raises(TimeoutError):
        client._find_run(job, wait=True, timeout=0)
    get.assert_not_called()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf")])
def test_nonfinite_wait_timeout_is_rejected(timeout):
    job = AIJob(runner(), "job", {}, RetryPolicy())
    with pytest.raises(ValueError):
        job.wait(timeout=timeout)
