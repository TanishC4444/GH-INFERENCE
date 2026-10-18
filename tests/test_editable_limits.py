"""Policy overrides and option transport; run on GitHub only."""
import json
from unittest.mock import Mock, patch

import pytest

from gh_ai_runner import Agent, GitHubAIRunner, InferenceRequest, Limits, model_resources
from gh_ai_runner.integrity import request_sha256
from gh_ai_runner.runner import WORKFLOW_YAML
from gh_ai_runner.validation import _validate, validate_text
from test_large_models import prepared_runner, worker_functions
from test_remote_agents import completion, execute_worker, remote_result


def test_experimental_factory_removes_project_policy_and_history_caps():
    agent = Agent.experimental(runner=Mock(), n_ctx=65536, max_tokens=None)
    assert not agent.trim_history
    assert agent.max_history_turns is None
    assert agent.limits == Limits.unrestricted()
    assert Agent.experimental(runner=Mock(), trim_history=True).trim_history


def test_model_and_output_limits_can_be_exceeded_without_faking_hardware():
    _validate("qwen0.5b", 999999, 5, 65536, limits=Limits.unrestricted())
    _validate("qwen0.5b", None, -1, 65536, limits=Limits.unrestricted())
    with pytest.raises(ValueError, match="supports n_ctx"):
        _validate("qwen0.5b", 1, .7, 65536)
    with pytest.raises(ValueError, match="RAM"):
        _validate("qwen0.5b", 1, .7, 65536,
                  limits=Limits(enforce_model_limits=False))


def test_custom_temperature_bounds_and_prompt_limit():
    _validate("qwen0.5b", 1, 4, None, limits=Limits(max_temperature=5))
    with pytest.raises(ValueError, match="temperature"):
        _validate("qwen0.5b", 1, 4, None)
    validate_text("x" * 60001, "", True, Limits.unrestricted())
    with pytest.raises(ValueError, match="10 characters"):
        validate_text("x" * 11, "", True, Limits(max_prompt_chars=10))


def test_cache_policy_and_estimates_are_editable():
    assert model_resources("deepseek-coder16b", limits=Limits.unrestricted())["cache_weights"]
    assert not model_resources("qwen0.5b", limits=Limits(max_cached_model_gb=0))["cache_weights"]
    result = model_resources("qwen0.5b", 65536, limits=Limits(
        enforce_model_limits=False, ram_per_1k_ctx_gb=0, runtime_overhead_gb=0,
        disk_reserve_gb=0))
    assert result["ram_gb"] < 1


def test_service_transport_limit_survives_unrestricted_mode():
    client = prepared_runner(limits=Limits.unrestricted())
    client.prepare = Mock()
    with pytest.raises(ValueError, match="GitHub's.*service limit"):
        client.submit("x" * 66000, model="qwen0.5b")
    client.prepare.assert_not_called()
    with patch("gh_ai_runner.client.requests.post") as post:
        client.submit("x" * 60001, model="qwen0.5b")
    assert post.called


def test_custom_policy_and_options_are_dispatched_and_integrity_bound():
    client = prepared_runner()
    limits = Limits.unrestricted(job_timeout_minutes=100, artifact_retention_days=7)
    with patch("gh_ai_runner.client.requests.post") as post:
        job = client.submit("hello", model="qwen0.5b", n_ctx=65536, max_tokens=None,
                            limits=limits, model_options={"n_threads": 1, "rope_freq_scale": .5},
                            generation_options={"top_k": 7, "repeat_penalty": 1.2})
    config = json.loads(post.call_args.kwargs["json"]["inputs"]["config"])
    assert config["limits"] == limits.to_dict()
    assert config["job_timeout_minutes"] == 100
    assert config["artifact_retention_days"] == 7
    assert config["model_options"]["n_threads"] == 1
    assert config["generation_options"]["top_k"] == 7
    before = request_sha256(job.request)
    job.request["generation_options"]["top_k"] = 8
    assert request_sha256(job.request) != before
    assert "timeout-minutes: ${{ fromJSON(inputs.config).job_timeout_minutes" in WORKFLOW_YAML
    assert "retention-days: ${{ fromJSON(inputs.config).artifact_retention_days" in WORKFLOW_YAML


def test_worker_bypasses_resource_estimates_and_context_ceiling(monkeypatch, tmp_path):
    namespace = worker_functions(monkeypatch, tmp_path)
    namespace["config"] = {"limits": Limits.unrestricted().to_dict(),
                           "model_options": {"n_threads": 1, "rope_freq_scale": .5},
                           "generation_options": {"top_k": 7}}
    namespace["memory_info"] = lambda: {"MemAvailable": 1}
    namespace["shutil"] = Mock()
    namespace["shutil"].disk_usage.return_value.free = 1
    cache = tmp_path / "model_cache"
    cache.mkdir()
    (cache / namespace["MODEL_MAP"]["qwen0.5b"]["filename"]).write_bytes(b"mock cache")
    with patch("llama_cpp.Llama") as constructor:
        namespace["local_inference"]([], None, 3, "qwen0.5b", 65536)
    assert constructor.call_args.kwargs["n_ctx"] == 65536
    assert constructor.call_args.kwargs["n_threads"] == 1
    assert constructor.return_value.create_chat_completion.call_args.kwargs["max_tokens"] is None
    assert constructor.return_value.create_chat_completion.call_args.kwargs["top_k"] == 7


def test_worker_passes_generation_options_and_none_output(tmp_path, monkeypatch):
    payload, _, llm, job = execute_worker(
        tmp_path, monkeypatch, ["hello"], [completion()], trim_history=False,
        inference_options={"max_tokens": None, "limits": Limits.unrestricted(),
                           "generation_options": {"top_k": 3, "seed": 42}})
    assert llm.create_chat_completion.call_args.kwargs["max_tokens"] is None
    assert llm.create_chat_completion.call_args.kwargs["seed"] == 42
    assert payload["integrity"]["request_sha256"] == job.request_digest


def test_batch_over_32_and_unlimited_turn_count_work(tmp_path, monkeypatch):
    payload, _, _, _ = execute_worker(
        tmp_path, monkeypatch, ["hello"] * 40, [completion()] * 40,
        trim_history=False, max_history_turns=None,
        inference_options={"limits": Limits.unrestricted()})
    assert len(payload["agent"]["results"]) == 40
    assert len(payload["agent"]["history"]) == 80
    runner = Mock()
    runner.submit.return_value.result.return_value = remote_result(["answer"] * 40)
    assert len(Agent.experimental(runner=runner).run_many(["hello"] * 40)) == 40


def test_batch_validates_overrides_before_dispatch():
    client = prepared_runner()
    with patch("gh_ai_runner.client.requests.post") as post, pytest.raises(ValueError):
        client.submit_batch([InferenceRequest("first"), InferenceRequest("bad", generation_options={"stream": True})])
    post.assert_not_called()


@pytest.mark.parametrize("options", [
    {"model_options": {"n_ctx": 100}}, {"model_options": {"use_mmap": True}},
    {"generation_options": {"max_tokens": 100}}, {"generation_options": {"stream": True}},
])
def test_core_execution_contract_not_overridden_through_extra_options(options):
    with pytest.raises(ValueError, match="cannot override"):
        Agent.experimental(runner=Mock(), **options)


@pytest.mark.parametrize("kwargs", [
    {"max_prompts": -1}, {"job_timeout_minutes": 0}, {"check_resources": "false"},
    {"refiner_context_fraction": float("inf")}, {"max_temperature": float("nan")},
    {"min_temperature": 3, "max_temperature": 2},
])
def test_malformed_policy_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Limits(**kwargs)
