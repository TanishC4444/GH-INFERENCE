"""Client and worker integration tests run by GitHub Actions."""
import io
import json
import sys
import types
import zipfile
from unittest.mock import Mock, patch

import pytest

from gh_ai_runner import Agent, GitHubAIRunner, InferenceResult, ai_call
from gh_ai_runner.artifact import _download_result
from gh_ai_runner.models import MODELS
from gh_ai_runner.runner import INFERENCE_SCRIPT


def response(data=None, content=b""):
    item = Mock(content=content)
    item.json.return_value = data or {}
    return item


def completion(text="answer", finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}


def remote_result(outputs, timed=False):
    return InferenceResult("j", "j-1", 1, outputs[-1], "qwen0.5b", "local", raw={
        "agent": {"results": [
            {"output": text, "usage": {"total_tokens": 7},
             "timings": {"total_seconds": 1, "generation_seconds": .5} if timed else None,
             "model_reused": index > 0}
            for index, text in enumerate(outputs)
        ], "history": [{"role": "user", "content": "previous"},
                         {"role": "assistant", "content": outputs[-1]}]},
    })


def test_agent_returns_text_and_many_prompts_use_one_dispatch():
    runner = Mock()
    runner.submit.return_value.result.return_value = remote_result(["one", "two"], timed=True)
    agent = Agent(runner=runner, track_time=True)
    assert agent.run_many(["first", "second"]) == ["one", "two"]
    assert agent.last_result.model_reused
    assert agent.last_result.timings.remote_seconds > 0
    runner.submit.assert_called_once()
    assert runner.submit.call_args.kwargs["agent_config"]["tasks"] == ["second"]


def test_single_prompt_and_callable_are_plain_strings():
    runner = Mock()
    runner.submit.return_value.result.return_value = remote_result(["answer"])
    agent = Agent(runner=runner)
    assert agent.run("question") == "answer"
    assert agent("another") == "answer"
    assert agent.last_result.output == "answer"


def test_token_can_be_set_once_for_both_clients(monkeypatch):
    monkeypatch.setenv("GH_AI_TOKEN", "test-token")
    assert GitHubAIRunner(metadata_path=None).github_token == "test-token"
    assert Agent().runner.github_token == "test-token"
    monkeypatch.delenv("GH_AI_TOKEN")
    with pytest.raises(ValueError, match="GH_AI_TOKEN"):
        Agent()


def test_connect_is_optional_and_chainable():
    runner = Mock()
    agent = Agent(runner=runner)
    assert agent.connect() is agent
    runner.prepare.assert_called_once()


def test_conversation_reset_and_disabled_history():
    runner = Mock()
    runner.submit.return_value.result.return_value = remote_result(["answer"])
    agent = Agent(runner=runner)
    agent.run("first")
    agent.run("next")
    assert runner.submit.call_args.kwargs["agent_config"]["history"]
    agent.reset()
    agent.run("fresh")
    assert not runner.submit.call_args.kwargs["agent_config"]["history"]
    Agent(runner=runner, remember=False).run("independent")
    assert not runner.submit.call_args.kwargs["agent_config"]["remember"]


@pytest.mark.parametrize("prompts", [[], "not a list", [" "], [None]])
def test_invalid_prompts_are_rejected_before_dispatch(prompts):
    runner = Mock()
    with pytest.raises(ValueError):
        Agent(runner=runner).run_many(prompts)
    runner.submit.assert_not_called()


def execute_worker(tmp_path, monkeypatch, tasks, answers, inference_options=None, **options):
    client = GitHubAIRunner("token", verbose=False, metadata_path=None)
    client._prepared, client._username, client._default_branch = True, "tester", "main"
    settings = {"model": "qwen0.5b", "max_tokens": 64, **(inference_options or {})}
    with patch("gh_ai_runner.client.requests.post", return_value=response()) as post:
        job = client.submit(tasks[0], agent_config={"tasks": tasks[1:], "track_time": True, **options},
                            **settings)
    inputs = post.call_args.kwargs["json"]["inputs"]
    for key in ("job_id", "attempt", "prompt", "system", "backend", "local_model", "cache"):
        monkeypatch.setenv(key.upper(), inputs[key])
    monkeypatch.setenv("CONFIG_JSON", inputs["config"])
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "model_cache"
    cache.mkdir(exist_ok=True)
    (cache / MODELS["qwen0.5b"]["filename"]).write_bytes(b"simulated weights")
    llm = Mock()
    llm.tokenize.side_effect = lambda text, **kwargs: list(range(len(text.split())))
    llm.create_chat_completion.side_effect = answers
    constructor = Mock(return_value=llm)
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=constructor))
    exec(compile(INFERENCE_SCRIPT, "run_inference.py", "exec"), {})  # noqa: S102
    return json.loads((tmp_path / "result.json").read_text()), constructor, llm, job


def test_worker_keeps_one_model_in_ram_for_all_prompts(tmp_path, monkeypatch):
    payload, constructor, llm, _ = execute_worker(
        tmp_path, monkeypatch, ["first", "second"], [completion("one"), completion("two")])
    assert constructor.call_count == 1
    assert constructor.call_args.kwargs["use_mmap"] is False
    assert constructor.call_args.kwargs["use_mlock"] is True
    llm.reset.assert_not_called()
    first, second = payload["agent"]["results"]
    assert not first["model_reused"] and second["model_reused"]
    assert second["timings"]["load_seconds"] == 0
    assert payload["usage"]["total_tokens"] == 14


def test_refinement_uses_same_model_and_counts_both_calls(tmp_path, monkeypatch):
    payload, constructor, _, _ = execute_worker(
        tmp_path, monkeypatch, ["please " * 140 + "return exactly 42"],
        [completion("return exactly 42"), completion("42")], refine_prompt=True)
    result = payload["agent"]["results"][0]
    assert constructor.call_count == 1
    assert result["refinement"]["applied"]
    assert result["usage"]["total_tokens"] == 14


def test_refinement_falls_back_when_a_literal_is_lost(tmp_path, monkeypatch):
    payload, _, llm, _ = execute_worker(
        tmp_path, monkeypatch, ["please " * 140 + "return exactly 42"],
        [completion("return 43"), completion("42")], refine_prompt=True)
    assert not payload["agent"]["results"][0]["refinement"]["applied"]
    assert "42" in llm.create_chat_completion.call_args.kwargs["messages"][-1]["content"]


def test_agent_artifact_round_trip_verifies_request_and_agent_body(tmp_path, monkeypatch):
    payload, _, _, job = execute_worker(tmp_path, monkeypatch, ["task"], [completion("answer")])
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("result.json", json.dumps(payload))
    with patch("gh_ai_runner.artifact.requests.get", side_effect=[
        response({"artifacts": [{"id": 1, "name": f"ai-output-{job.correlation_id}"}]}),
        response(content=zipped.getvalue()),
    ]):
        result = _download_result("t", "u", "r", 1, job.job_id, job.correlation_id,
                                  False, expected_request_sha256=job.request_digest)
    assert result.raw["agent"]["results"][0]["output"] == "answer"
    assert result.resources.cpu_seconds is not None
    assert result.resources.cpu_seconds >= 0


def test_old_api_returns_text_unless_timing_is_explicit():
    runner = Mock()
    runner.submit.return_value.result.return_value = InferenceResult("j", "j-1", 1, "answer", "qwen", "local")
    with patch("gh_ai_runner.core.GitHubAIRunner", return_value=runner):
        assert ai_call("token", "task") == "answer"
        assert ai_call("token", "task", track_time=True).timings.total_seconds > 0


def test_dispatch_error_preserves_job_handle_without_reposting():
    import requests
    runner = GitHubAIRunner('test-token', metadata_path=None, verbose=False)
    runner.prepare = Mock()
    runner._dispatch = Mock(side_effect=requests.ReadTimeout('uncertain POST'))
    agent = Agent(runner=runner)
    with pytest.raises(requests.ReadTimeout) as raised:
        agent.run('task')
    assert agent.last_job is raised.value.job
    assert agent.last_job.job_id
    runner._dispatch.assert_called_once()


@pytest.mark.parametrize('topology', ['all', 'ring'])
def test_discussion_runs_synchronized_rounds_on_one_loaded_model(tmp_path, monkeypatch, topology):
    replies = [completion(f'reply-{i}') for i in range(6)]
    payload, constructor, llm, _ = execute_worker(
        tmp_path, monkeypatch, ['design a library'], replies,
        discussion=dict(agents=3, rounds=2, peer_chars=80, topology=topology))
    assert constructor.call_count == 1
    results = payload['agent']['results']
    assert len(results) == 6
    assert payload['usage']['completion_tokens'] == 12
    assert not results[0]['model_reused']
    assert all(item['model_reused'] for item in results[1:])
    assert all(item['timings']['load_seconds'] == 0 for item in results[1:])
    prompts = [call.kwargs['messages'][-1]['content'] for call in llm.create_chat_completion.call_args_list]
    assert all('reply-' not in prompt for prompt in prompts[:3])
    assert 'Agent 2: reply-2' in prompts[3]
    if topology == 'all':
        assert all(f'Agent {i}: reply-{i}' in prompts[3] for i in range(3))
    else:
        assert 'reply-0' not in prompts[3] and 'reply-1' not in prompts[3]
    assert 'reply-3' not in prompts[4]  # never observe current-round answers
    llm.reset.assert_not_called()


def test_discussion_client_dispatches_once_and_preserves_conversation():
    runner = Mock()
    runner.submit.return_value.result.return_value = remote_result(['a', 'b', 'c', 'd'])
    agent = Agent(runner=runner, max_tokens=None)
    agent._history = [{'role': 'user', 'content': 'private history'}]
    assert agent.discuss('task', agents=2, rounds=2) == [['a', 'b'], ['c', 'd']]
    runner.submit.assert_called_once()
    request = runner.submit.call_args.kwargs
    assert request['max_tokens'] is None
    assert not request['agent_config']['history']
    assert agent._history[0]['content'] == 'private history'


@pytest.mark.parametrize('options', [dict(agents=True), dict(rounds=0), dict(topology='bad'),
                                     dict(agents=20, rounds=3)])
def test_discussion_rejects_bad_options_before_dispatch(options):
    runner = Mock()
    with pytest.raises(ValueError):
        Agent(runner=runner).discuss('task', **options)
    runner.submit.assert_not_called()


def test_template_options_are_bound_to_formatter_not_model_constructor(tmp_path, monkeypatch):
    from functools import partial
    payload, constructor, llm, _ = execute_worker(
        tmp_path, monkeypatch, ['write final text'], [completion('final text')],
        inference_options={'model_options': {'chat_template_options': {'enable_thinking': False}}})
    assert 'chat_template_options' not in constructor.call_args.kwargs
    assert isinstance(llm.chat_handler, partial)
    assert llm.chat_handler.keywords == {'enable_thinking': False}
    assert 'runtime_version' in payload['resources']


@pytest.mark.parametrize('value', ['bad', {'enable_thinking': 'no'}, {'messages': []}])
def test_invalid_template_options_fail_before_dispatch(value):
    runner = Mock()
    with pytest.raises(ValueError, match='chat_template_options'):
        Agent(runner=runner, model_options={'chat_template_options': value})
    runner.submit.assert_not_called()


def test_runtime_version_is_pinned_and_separates_cached_environments():
    from gh_ai_runner.runner import RUNTIME_VERSION, WORKFLOW_YAML
    assert f'llama-cpp-python=={RUNTIME_VERSION}' in WORKFLOW_YAML
    assert f'python311-llama-cpp-{RUNTIME_VERSION}-v5' in WORKFLOW_YAML
    assert '__RUNTIME_VERSION__' not in WORKFLOW_YAML
