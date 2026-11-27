from copy import deepcopy
from unittest.mock import Mock

import pytest

from gh_ai_runner import Agent
from gh_ai_runner.pipeline import normalize_pipeline
from test_remote_agents import completion, execute_worker, remote_result


PLAN = [
    {'agent': 'writer', 'instruction': 'Write the draft.', 'inputs': []},
    {'agent': 'reviewer', 'instruction': 'Check the draft.', 'inputs': [0]},
    {'agent': 'writer', 'instruction': 'Fix the draft.', 'inputs': [0, 1]},
]


def test_review_loop_reuses_model_and_routes_immediate_replies(tmp_path, monkeypatch):
    payload, constructor, llm, _ = execute_worker(
        tmp_path, monkeypatch, ['shared task'],
        [completion('DRAFT_A'), completion('FEEDBACK_A'), completion('FIXED_A')], pipeline=PLAN)
    assert constructor.call_count == 1
    llm.reset.assert_not_called()
    rows = payload['agent']['results']
    assert [row['model_reused'] for row in rows] == [False, True, True]
    assert [row['agent_id'] for row in rows] == ['writer', 'reviewer', 'writer']
    assert [row['input_steps'] for row in rows] == [[], [0], [0, 1]]
    assert all(row['timings']['load_seconds'] == 0 for row in rows[1:])
    assert all(row['timings']['routing_seconds'] >= 0 for row in rows)
    prompts = [call.kwargs['messages'] for call in llm.create_chat_completion.call_args_list]
    assert 'DRAFT_A' not in prompts[0][-1]['content']
    assert 'DRAFT_A' in prompts[1][-1]['content']
    assert 'FEEDBACK_A' not in prompts[1][-1]['content']
    assert prompts[2][-1]['content'].count('DRAFT_A') == 1
    assert prompts[2][-1]['content'].count('FEEDBACK_A') == 1
    assert all('shared task' in prompt[0]['content'] for prompt in prompts)
    assert payload['usage']['completion_tokens'] == 6
    assert rows[0]['handoff_bytes'] == 0


def test_broadcast_and_gather_send_each_selected_output_once(tmp_path, monkeypatch):
    steps = [
        PLAN[0],
        {'agent': 'critic-a', 'instruction': 'Review draft.', 'inputs': [0]},
        {'agent': 'critic-b', 'instruction': 'Independently review.', 'inputs': [0]},
        {'agent': 'writer', 'instruction': 'Combine feedback.', 'inputs': [0, 1, 0, 2, 1]},
    ]
    payload, constructor, llm, _ = execute_worker(
        tmp_path, monkeypatch, ['task'],
        [completion(text) for text in ['DRAFT_UNIQUE', 'CRITIC_A_PRIVATE', 'CRITIC_B_PRIVATE', 'FIXED']],
        pipeline=steps)
    prompts = [call.kwargs['messages'][-1]['content'] for call in llm.create_chat_completion.call_args_list]
    assert 'CRITIC_A_PRIVATE' not in prompts[2]
    for text in ['DRAFT_UNIQUE', 'CRITIC_A_PRIVATE', 'CRITIC_B_PRIVATE']:
        assert prompts[3].count(text) == 1
    assert payload['agent']['results'][3]['input_steps'] == [0, 1, 2]
    # Shared draft appears before varying critic identities/instructions, allowing
    # the backend to reuse that complete prefix across broadcast recipients.
    common = prompts[1].split('Your role:', 1)[0]
    assert prompts[2].startswith(common)
    assert 'DRAFT_UNIQUE' in common
    assert constructor.call_count == 1


def test_pipeline_client_dispatches_once_and_keeps_metadata_and_conversation():
    runner = Mock()
    remote = remote_result(['draft', 'review', 'fixed'], timed=True)
    for index, item in enumerate(remote.raw['agent']['results']):
        item.update(agent_id=PLAN[index]['agent'], step_index=index,
                    input_steps=PLAN[index]['inputs'], handoff_chars=10, handoff_bytes=12)
    runner.submit.return_value.result.return_value = remote
    agent = Agent(runner=runner, track_time=True, max_tokens=None)
    agent._history = [{'role': 'user', 'content': 'private conversation'}]
    assert agent.review('task') == 'fixed'
    runner.submit.assert_called_once()
    request = runner.submit.call_args.kwargs
    assert request['max_tokens'] is None
    assert not request['agent_config']['history']
    assert [step['inputs'] for step in request['agent_config']['pipeline']] == [[], [0], [0, 1]]
    assert agent._history[0]['content'] == 'private conversation'
    assert agent.last_result.agent_id == 'writer'
    assert agent.last_result.input_steps == [0, 1]
    assert agent.last_result.handoff_bytes == 12
    assert agent.last_result.resources is agent.last_results[0].resources


def test_repeated_revision_only_routes_latest_draft_and_feedback():
    runner = Mock()
    runner.submit.return_value.result.return_value = remote_result(['d0', 'r0', 'd1', 'r1', 'd2'])
    agent = Agent(runner=runner)
    assert agent.review('task', revisions=2) == 'd2'
    plan = runner.submit.call_args.kwargs['agent_config']['pipeline']
    assert plan[-2]['inputs'] == [2]
    assert plan[-1]['inputs'] == [2, 3]
    runner.submit.assert_called_once()


@pytest.mark.parametrize('bad', [None, [], ['step'], [{}],
    [{'agent': 'a', 'instruction': 'x', 'inputs': [0]}],
    [{'agent': 'a', 'instruction': 'x', 'inputs': [-1]}],
    [PLAN[0], {'agent': 'b', 'instruction': 'x', 'inputs': [True]}],
    [{'agent': '', 'instruction': 'x'}],
    [{'agent': 'a', 'instruction': None}],
    [{'agent': 'a', 'instruction': 'x', 'typo': True}],
    [{'agent': 'a', 'instruction': 'x', 'inputs': '0'}]])
def test_bad_pipeline_fails_before_dispatch(bad):
    runner = Mock()
    with pytest.raises(ValueError):
        Agent(runner=runner).pipeline('task', bad)
    runner.submit.assert_not_called()


def test_pipeline_normalization_preserves_user_plan_and_reference_order():
    steps = deepcopy(PLAN)
    steps[-1]['inputs'] = [1, 0, 1]
    original = deepcopy(steps)
    assert normalize_pipeline(steps)[-1]['inputs'] == [1, 0]
    assert steps == original
    with pytest.raises(ValueError, match='max_prompts'):
        normalize_pipeline(PLAN, max_prompts=2)


def test_worker_also_rejects_forward_references(tmp_path, monkeypatch):
    bad = [{'agent': 'a', 'instruction': 'x', 'inputs': [1]}]
    with pytest.raises(ValueError, match='earlier'):
        execute_worker(tmp_path, monkeypatch, ['task'], [], pipeline=bad)


@pytest.mark.parametrize('revisions', [0, -1, True, 1.5, 100])
def test_bad_review_limits_do_not_dispatch(revisions):
    runner = Mock()
    with pytest.raises(ValueError):
        Agent(runner=runner).review('task', revisions=revisions)
    runner.submit.assert_not_called()


def test_unicode_handoff_metrics_and_dynamic_generation(tmp_path, monkeypatch):
    payload, constructor, llm, _ = execute_worker(
        tmp_path, monkeypatch, ['task'], [completion('雪🙂'), completion('checked')],
        inference_options={'max_tokens': None}, pipeline=PLAN[:2])
    row = payload['agent']['results'][1]
    assert row['handoff_bytes'] > row['handoff_chars']
    prompt = llm.create_chat_completion.call_args_list[-1].kwargs['messages'][-1]['content']
    handoff = prompt.split('Provided outputs:\n', 1)[1].split('\n\nYour role:', 1)[0]
    assert row['handoff_bytes'] == len(handoff.encode('utf-8'))
    assert row['handoff_chars'] == len(handoff)
    assert constructor.call_count == 1


def test_context_overflow_does_not_silently_truncate_selected_outputs(tmp_path, monkeypatch):
    text = 'unique-draft ' * 1800
    with pytest.raises(ValueError, match='exceed context'):
        execute_worker(tmp_path, monkeypatch, ['task'], [completion(text), completion('unused')],
                       inference_options={'n_ctx': 1024}, pipeline=PLAN[:2])
