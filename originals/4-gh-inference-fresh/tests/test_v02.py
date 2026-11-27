import io
import json
import threading
import urllib.request
import zipfile
from unittest.mock import Mock, patch

import pytest
from nacl import encoding, public

from gh_ai_runner import (
    GitHubAIRunner,
    JobCancelledError,
    JobStatus,
    ProviderConfig,
    RetryPolicy,
)
from gh_ai_runner.artifact import _download_result
from gh_ai_runner.core import ai_call
from gh_ai_runner.integrity import request_sha256, text_sha256
from gh_ai_runner.job import AIJob
from gh_ai_runner.polling import _find_run
from gh_ai_runner.repo import _set_actions_secret
from gh_ai_runner.runner import INFERENCE_SCRIPT, WORKFLOW_YAML


def response(json_data=None, status=200, content=b""):
    item = Mock(status_code=status, content=content)
    item.json.return_value = json_data or {}
    item.raise_for_status.side_effect = None if status < 400 else RuntimeError(status)
    return item


def prepared_runner(metadata_path=None):
    runner = GitHubAIRunner("token", verbose=False, metadata_path=metadata_path)
    runner._prepared = True
    runner._username = "octocat"
    runner._default_branch = "trunk"
    return runner


def test_submit_returns_without_polling_and_dispatches_exact_correlation():
    runner = prepared_runner()
    with patch("gh_ai_runner.client.uuid.uuid4", return_value=Mock(hex="abc123")), patch(
        "gh_ai_runner.client.requests.post", return_value=response(status=204)
    ) as post:
        job = runner.submit("hello", model="qwen")

    assert job.job_id == "abc123"
    assert job.correlation_id == "abc123-1"
    payload = post.call_args.kwargs["json"]
    assert payload["ref"] == "trunk"
    assert payload["inputs"]["job_id"] == "abc123"
    assert payload["inputs"]["attempt"] == "1"
    assert json.loads(payload["inputs"]["config"])["model"] == "qwen"


def test_find_run_matches_run_title_not_first_new_run():
    runs = {
        "workflow_runs": [
            {"id": 10, "display_title": "AI inference another-1"},
            {"id": 11, "display_title": "AI inference wanted-1"},
        ]
    }
    with patch("gh_ai_runner.polling.requests.get", return_value=response(runs)):
        found = _find_run("token", "octocat", "repo", "wanted-1", False, timeout=0)
    assert found["id"] == 11


def test_refresh_maps_github_waiting_state_to_queued():
    runner = prepared_runner()
    runner._find_run = Mock(return_value={"id": 22, "status": "waiting"})
    job = AIJob(runner, "job", {}, RetryPolicy())
    assert job.refresh() == JobStatus.QUEUED


def test_failed_job_retries_with_incremented_attempt():
    runner = prepared_runner()
    runner._find_run = Mock(side_effect=[{"id": 1}, {"id": 2}])
    runner._wait_for_completion = Mock(
        side_effect=[
            {"status": "completed", "conclusion": "failure"},
            {"status": "completed", "conclusion": "success"},
        ]
    )
    runner._dispatch = Mock()
    job = AIJob(
        runner,
        "retry-me",
        {"prompt": "x"},
        RetryPolicy(max_retries=1, delay_seconds=0),
    )
    job.wait()
    assert job.attempt == 2
    runner._dispatch.assert_called_once_with(job)


def test_result_does_not_hold_lock_while_waiting():
    runner = prepared_runner()
    entered = threading.Event()
    release = threading.Event()
    runner._find_run = Mock(return_value={"id": 1})

    def slow_completion(job, timeout=None):
        entered.set()
        release.wait(1)
        return {"status": "completed", "conclusion": "success"}

    runner._wait_for_completion = slow_completion
    runner._download_result = Mock(return_value=Mock(output="ok"))
    runner._cancel_run = Mock(return_value=True)
    job = AIJob(runner, "cancel-me", {}, RetryPolicy())
    def collect_cancelled_result():
        with pytest.raises(JobCancelledError):
            job.result()

    thread = threading.Thread(target=collect_cancelled_result)
    thread.start()
    assert entered.wait(1)
    assert job.cancel() is True
    release.set()
    thread.join(1)
    assert not thread.is_alive()


def test_cancel_completed_job_is_a_noop():
    runner = prepared_runner()
    runner._find_run = Mock(return_value={"id": 1, "status": "completed", "conclusion": "success"})
    runner._cancel_run = Mock()
    job = AIJob(runner, "done", {}, RetryPolicy())
    assert job.cancel() is False
    runner._cancel_run.assert_not_called()


def test_ai_call_keeps_string_return_compatibility():
    fake_job = Mock()
    fake_job.result.return_value = Mock(output="legacy string")
    fake_runner = Mock()
    fake_runner.submit.return_value = fake_job
    with patch("gh_ai_runner.core.GitHubAIRunner", return_value=fake_runner):
        assert ai_call("token", "hello", verbose=False) == "legacy string"


def test_structured_artifact_is_parsed():
    payload = {
        "job_id": "j", "correlation_id": "j-1", "model": "qwen",
        "provider": "local", "output": "answer",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        "resources": {"cpu_count": 4, "duration_seconds": 1.5},
        "integrity": {"algorithm": "sha256", "output_sha256": text_sha256("answer")},
    }
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("result.json", json.dumps(payload))
    calls = [
        response({"artifacts": [{"id": 9, "name": "ai-output-j-1"}]}),
        response(content=zipped.getvalue()),
    ]
    with patch("gh_ai_runner.artifact.requests.get", side_effect=calls):
        result = _download_result("t", "u", "r", 7, "j", "j-1", False)
    assert result.output == "answer"
    assert result.usage.total_tokens == 6
    assert result.resources.cpu_count == 4


def test_provider_requires_https_and_workflow_uses_only_secret_name():
    with pytest.raises(ValueError):
        ProviderConfig("bad", "model", "http://unsafe", "API_KEY")
    provider = ProviderConfig.groq("llama-3.3-70b-versatile")
    assert provider.secret_name == "GH_AI_PROVIDER_API_KEY"
    assert "secrets.GH_AI_PROVIDER_API_KEY" in WORKFLOW_YAML
    assert "provider_secret" not in WORKFLOW_YAML
    assert "inputs.api_key" not in WORKFLOW_YAML.lower()


def test_provider_key_is_sealed_before_upload():
    private_key = public.PrivateKey.generate()
    encoded_key = private_key.public_key.encode(encoder=encoding.Base64Encoder).decode()
    get_response = response({"key": encoded_key, "key_id": "kid"})
    put_response = response(status=201)
    with patch("gh_ai_runner.repo.requests.get", return_value=get_response), patch(
        "gh_ai_runner.repo.requests.put", return_value=put_response
    ) as put:
        _set_actions_secret("github-token", "octocat", "repo", "GH_AI_PROVIDER_API_KEY", "secret-value")

    uploaded = put.call_args.kwargs["json"]
    assert uploaded["key_id"] == "kid"
    assert uploaded["encrypted_value"] != "secret-value"
    decrypted = public.SealedBox(private_key).decrypt(
        uploaded["encrypted_value"].encode(), encoder=encoding.Base64Encoder
    )
    assert decrypted == b"secret-value"


def test_embedded_worker_compiles_and_emits_structured_result():
    compile(INFERENCE_SCRIPT, "run_inference.py", "exec")
    assert 'run-name: AI inference ${{ inputs.job_id }}-${{ inputs.attempt }}' in WORKFLOW_YAML
    assert "result.json" in INFERENCE_SCRIPT
    workflow_inputs = WORKFLOW_YAML.split("inputs:\n", 1)[1].split("\njobs:", 1)[0]
    assert sum(1 for line in workflow_inputs.splitlines() if line.startswith("      ")) <= 10


def test_embedded_provider_worker_reports_usage_and_resources(tmp_path, monkeypatch):
    provider_response = {
        "choices": [{"message": {"content": "provider answer"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    http_response = Mock()
    http_response.read.return_value = json.dumps(provider_response).encode()
    http_response.__enter__ = Mock(return_value=http_response)
    http_response.__exit__ = Mock(return_value=False)

    monkeypatch.chdir(tmp_path)
    environment = {
        "JOB_ID": "worker-job",
        "ATTEMPT": "1",
        "BACKEND": "provider",
        "PROMPT": "hello",
        "SYSTEM": "help",
        "LOCAL_MODEL": "tinyllama",
        "PROVIDER_API_KEY": "provider-secret",
        "CONFIG_JSON": json.dumps({
            "model": "remote-model", "max_tokens": 10, "temperature": 0.2,
            "provider_name": "test", "provider_url": "https://provider.example/v1",
        }),
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with patch.object(urllib.request, "urlopen", return_value=http_response) as urlopen:
        exec(compile(INFERENCE_SCRIPT, "run_inference.py", "exec"), {})  # noqa: S102

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["output"] == "provider answer"
    assert result["usage"]["total_tokens"] == 8
    assert result["resources"]["cpu_count"] >= 1
    expected_request = {
        "prompt": "hello", "system": "help", "model": "remote-model",
        "local_model": "tinyllama", "cache": True, "max_tokens": 10,
        "temperature": 0.2, "n_ctx": None,
        "provider": ProviderConfig("test", "remote-model", "https://provider.example/v1"),
    }
    assert result["integrity"]["request_sha256"] == request_sha256(expected_request)
    request = urlopen.call_args.args[0]
    assert request.headers["Authorization"] == "Bearer provider-secret"
