"""Check standalone orchestration with simulated inference; no model downloads."""
import json
import runpy
from pathlib import Path

import pytest


def module():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "showcase.py"))


def test_workflow_exposes_parallel_tests_and_dependency_graph():
    yaml = pytest.importorskip("yaml")
    tour = module()
    workflow = yaml.load(tour["workflow_text"](), Loader=yaml.BaseLoader)
    assert workflow["on"]["workflow_dispatch"]["inputs"]["parallel"]["default"] == "6"
    assert len(workflow["jobs"]["test"]["strategy"]["matrix"]["case"]) == 6
    assert "needs" not in workflow["jobs"]["draft"]
    for name, parent in [("critique", "draft"), ("revise", "critique"), ("loop", "revise")]:
        job = workflow["jobs"][name]
        assert job["needs"] == [parent]
        download = next(s for s in job["steps"] if s.get("uses") == "actions/download-artifact@v4")
        assert download["with"] == {"name": "showcase-" + parent, "path": "incoming"}
        execute = next(s for s in job["steps"] if s.get("name") == "Run test and write readable summary")
        assert execute["env"]["CASE"] == name
        assert "inputs.context" in execute["env"]["SHOWCASE_CONTEXT"]
    assert tour["agent_cases"]()["critique"]["model"] != tour["agent_cases"]()["draft"]["model"]


def test_feedback_loop_reuses_namespace_and_passes_failed_answer_forward(monkeypatch, tmp_path):
    import gh_ai_runner.runner
    fake = '''
import json
calls = []
def run_agent(agent, messages, config):
    calls.append(messages[-1]['content'])
    profit = 42 if len(calls) == 1 else 30
    if len(calls) == 2:
        assert '"profit": 42' in calls[-1]
    item = {'output': json.dumps({'revenue':126,'cost':96,'profit':profit}),
            'timings':{'generation_seconds':.1,'load_seconds':0},
            'usage':{'total_tokens':10}, 'dropped_history_turns':0}
    return {'results':[item]}
started = time.monotonic()
'''
    monkeypatch.setattr(gh_ai_runner.runner, "INFERENCE_SCRIPT", fake)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    (tmp_path / "incoming").mkdir()
    (tmp_path / "incoming/report.json").write_text(json.dumps({
        "passed": True, "case": "revise", "result": {"output": "Prior actual answer"},
        "handoff": [{"agent": "draft", "answer": "Original draft"}]}))
    assert module()["worker"]("loop") == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["result"]["iterations"] == 2
    assert report["result"]["validator_passed"]
    assert "Prior actual answer" in report["prompts"][0]
    assert len(report["handoff"]) == 2


def test_failed_predecessor_cannot_be_used_as_agent_evidence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "incoming").mkdir()
    (tmp_path / "incoming/report.json").write_text('{"passed":false}')
    with pytest.raises(ValueError, match="Upstream agent failed"):
        module()["worker"]("critique")
