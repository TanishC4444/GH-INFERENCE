import json
import threading
from types import SimpleNamespace

import pytest

from gh_ai_runner import AgentResult, AgentTimings, ResourceUsage, TokenUsage, stress_test


class FakeAgent:
    barrier = None
    prompts = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.last_job = SimpleNamespace(actions_url='https://example.test/run')

    def connect(self):
        return self

    def run(self, prompt, **kwargs):
        self.barrier.wait(timeout=10)
        self.prompts.append(prompt)
        self.last_result = AgentResult(
            output='peer proposal', model=self.model,
            usage=TokenUsage(4, 8, 12),
            timings=AgentTimings(generation_seconds=2),
            resources=ResourceUsage(duration_seconds=10, cpu_seconds=4, peak_memory_bytes=100))


def test_twenty_concurrent_agents_share_previous_round_and_save(tmp_path):
    FakeAgent.barrier = threading.Barrier(20)
    FakeAgent.prompts = []
    path = tmp_path / 'report.json'
    report = stress_test(['a', 'b'], rounds=2, estimated_worker_watts=36,
                         agent_factory=FakeAgent, output=path)
    assert len(report['records']) == 40
    assert all('Agent 19: peer proposal' in prompt for prompt in FakeAgent.prompts[20:])
    assert all('Agent 0:' not in prompt for prompt in FakeAgent.prompts[:20])
    assert report['summary']['a']['completion_tokens_per_generation_second'] == 4
    assert report['summary']['a']['cpu_seconds'] == 80
    assert report['summary']['a']['estimated_energy_wh'] == pytest.approx(2)
    assert json.loads(path.read_text()) == report


def test_failure_is_retained_and_energy_unknown():
    class Failing(FakeAgent):
        def run(self, *args, **kwargs):
            raise RuntimeError('worker failed')
    report = stress_test(['a'], agents=2, rounds=2, agent_factory=Failing, output=None)
    assert report['summary']['a']['failed'] == 4
    assert report['summary']['a']['estimated_energy_wh'] is None


@pytest.mark.parametrize('options', [{'agents': 0}, {'rounds': True},
    {'estimated_worker_watts': float('nan')}, {'models': []}, {'timeout': -1}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        stress_test(agent_factory=FakeAgent, output=None, **options)


def test_resources_include_cpu_and_default_to_unknown_for_old_workers():
    assert ResourceUsage().cpu_seconds is None
    assert AgentResult('answer', 'model').resources.cpu_seconds is None


def test_ring_shares_only_previous_neighbor_and_accepts_dynamic_output():
    FakeAgent.barrier = threading.Barrier(3)
    FakeAgent.prompts = []
    report = stress_test(['a'], agents=3, rounds=2, topology='ring', max_tokens=None,
                         agent_factory=FakeAgent, output=None)
    assert len(report['records']) == 6
    assert report['max_tokens'] is None
    for prompt in FakeAgent.prompts[3:]:
        assert prompt.count('Agent ') == 1
