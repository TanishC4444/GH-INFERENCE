import io
import json
import zipfile
from unittest.mock import Mock, patch

import pytest

from gh_ai_runner import (
    GitHubAIRunner,
    InferenceRequest,
    InferenceResult,
    IntegrityError,
    IntegrityInfo,
    ProviderConfig,
    ResourceUsage,
    TokenUsage,
)
from gh_ai_runner.artifact import _download_result
from gh_ai_runner.batch import BatchJob
from gh_ai_runner.integrity import text_sha256


def response(json_data=None, status=200, content=b""):
    item = Mock(status_code=status, content=content)
    item.json.return_value = json_data or {}
    item.raise_for_status.side_effect = None if status < 400 else RuntimeError(status)
    return item


def prepared_runner(metadata_path=None):
    runner = GitHubAIRunner("github-token", verbose=False, metadata_path=metadata_path)
    runner._prepared = True
    runner._username = "octocat"
    runner._default_branch = "main"
    return runner


def zipped_result(payload):
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("result.json", json.dumps(payload))
    return zipped.getvalue()


def test_batch_dispatches_every_request_without_waiting():
    runner = prepared_runner()
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)) as post:
        batch = runner.submit_batch([
            InferenceRequest("first", model="tinyllama"),
            InferenceRequest("second", model="qwen"),
        ])
    assert len(batch.jobs) == 2
    assert batch.batch_id
    assert all(job.batch_id == batch.batch_id for job in batch.jobs)
    assert post.call_count == 2
    assert batch.jobs[0].job_id != batch.jobs[1].job_id


def test_compare_creates_model_keyed_job_group():
    runner = prepared_runner()
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)):
        comparison = runner.compare("same prompt", ["tinyllama", "qwen"])
    assert comparison.models == ["tinyllama", "qwen"]
    assert [job.request["model"] for job in comparison.jobs] == ["tinyllama", "qwen"]
    with pytest.raises(ValueError):
        runner.compare("same", ["qwen", "qwen"])
    with pytest.raises(ValueError):
        runner.compare("same", ["qwen", "not-a-model"])


def test_metadata_persists_hashes_without_sensitive_text(tmp_path):
    database = tmp_path / "jobs.sqlite3"
    runner = prepared_runner(database)
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)):
        job = runner.submit("private prompt text", model="qwen", system="private system")

    records = runner.list_jobs()
    assert len(records) == 1
    assert records[0].job_id == job.job_id
    assert records[0].prompt_sha256 == text_sha256("private prompt text")
    assert records[0].request_sha256 == job.request_digest
    contents = database.read_bytes()
    assert b"private prompt text" not in contents
    assert b"private system" not in contents
    assert b"github-token" not in contents

    resumed = runner.resume(job.job_id)
    assert resumed.correlation_id == job.correlation_id
    assert resumed.request_digest == job.request_digest


def test_metadata_schema_adds_batch_column_to_early_database(tmp_path):
    import sqlite3

    database = tmp_path / "old.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                correlation_id TEXT PRIMARY KEY, job_id TEXT, attempt INTEGER,
                owner TEXT, repo_name TEXT, status TEXT, run_id INTEGER, model TEXT,
                provider TEXT, prompt_sha256 TEXT, request_sha256 TEXT,
                output_sha256 TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,
                total_tokens INTEGER, duration_seconds REAL, created_at TEXT,
                updated_at TEXT, error TEXT
            )
            """
        )
    runner = prepared_runner(database)
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)):
        job = runner.submit_batch([InferenceRequest("migration")])
    assert runner.list_jobs()[0].batch_id == job.batch_id


def test_result_checksum_detects_modified_output():
    payload = {
        "job_id": "j", "correlation_id": "j-1", "model": "qwen",
        "provider": "local", "output": "changed",
        "usage": {}, "resources": {},
        "integrity": {
            "algorithm": "sha256", "request_sha256": "request",
            "output_sha256": text_sha256("original"),
        },
    }
    calls = [
        response({"artifacts": [{"id": 9, "name": "ai-output-j-1"}]}),
        response(content=zipped_result(payload)),
    ]
    with patch("gh_ai_runner.artifact.requests.get", side_effect=calls), pytest.raises(
        IntegrityError, match="output checksum"
    ):
        _download_result("t", "u", "r", 7, "j", "j-1", False)


def test_result_checksum_verifies_request_and_output():
    output = "verified"
    payload = {
        "job_id": "j", "correlation_id": "j-1", "model": "remote",
        "provider": "groq", "output": output,
        "usage": {"total_tokens": 4}, "resources": {},
        "integrity": {
            "algorithm": "sha256", "request_sha256": "request-hash",
            "output_sha256": text_sha256(output),
        },
    }
    calls = [
        response({"artifacts": [{"id": 9, "name": "ai-output-j-1"}]}),
        response(content=zipped_result(payload)),
    ]
    with patch("gh_ai_runner.artifact.requests.get", side_effect=calls):
        result = _download_result(
            "t", "u", "r", 7, "j", "j-1", False,
            expected_request_sha256="request-hash",
        )
    assert result.integrity.output_sha256 == text_sha256(output)


def test_batch_accepts_provider_requests():
    runner = prepared_runner()
    provider = ProviderConfig.groq("remote-model")
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)):
        batch = runner.submit_batch([InferenceRequest("remote", provider=provider)])
    assert batch.jobs[0].request["provider"] == provider


def test_batch_results_preserve_submission_order():
    first = Mock()
    first.result.return_value = "first-result"
    second = Mock()
    second.result.return_value = "second-result"
    batch = BatchJob([first, second])
    assert batch.results() == ["first-result", "second-result"]


def test_completed_result_updates_persistent_usage_metadata(tmp_path):
    runner = prepared_runner(tmp_path / "jobs.sqlite3")
    with patch("gh_ai_runner.client.requests.post", return_value=response(status=204)):
        job = runner.submit("track this", model="qwen")
    runner._find_run = Mock(return_value={"id": 33, "status": "completed"})
    runner._wait_for_completion = Mock(
        return_value={"status": "completed", "conclusion": "success"}
    )
    completed = InferenceResult(
        job_id=job.job_id,
        correlation_id=job.correlation_id,
        run_id=33,
        output="done",
        model="qwen",
        provider="local",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        resources=ResourceUsage(duration_seconds=3.5),
        integrity=IntegrityInfo(
            request_sha256=job.request_digest, output_sha256=text_sha256("done")
        ),
    )
    runner._download_result = Mock(return_value=completed)

    assert job.result().output == "done"
    record = runner.list_jobs()[0]
    assert record.status == "completed"
    assert record.run_id == 33
    assert record.total_tokens == 7
    assert record.duration_seconds == 3.5
    assert record.output_sha256 == text_sha256("done")
