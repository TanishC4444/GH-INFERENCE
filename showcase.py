"""Share this ONE file. Python 3.9+.

SETUP: python -m pip install gh-inference==0.5.0
RUN:   python showcase.py
       python showcase.py --parallel 3 --details

Enter a GitHub token with repo + workflow scopes when prompted (hidden).
Creates/reuses a private gh-inference-showcase repository. One workflow contains
six independent matrix jobs plus a draft → critique → revise → feedback chain.
All independent jobs are eligible to run together (account limits still apply).
Each job loads Llama 3.1 8B Q4 into RAM on GitHub, never on your computer.
This uses GitHub Actions minutes/storage. Allow several minutes on CPU.
Prompts/results remain in that repository and its three-day Actions artifacts.
Only this file and its managed showcase.yml workflow are uploaded.
--details prints full prompts/responses; normal output is intentionally short.
"""
import argparse
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile
import re
from types import SimpleNamespace

MODEL = "llama8b"
CONTEXT = int(os.environ.get("SHOWCASE_CONTEXT", "4096"))
OUTPUT = int(os.environ.get("SHOWCASE_OUTPUT", "512"))
SYSTEM = "Answer concisely. State assumptions, give a brief solution summary, and verify the final answer with a calculation or counterexample."
REWRITE_TASK = ("Please be clear and concise. " * 35 +
                "A shop buys 12 items at $8 each and sells 9 at $14 each. "
                "Calculate revenue minus the cost of all 12 items. Include the number 30.")


def cases():
    history = []
    for i in range(max(12, CONTEXT // 45)):
        history += [{"role": "user", "content": f"Old note {i}: " + "archived detail " * 45},
                    {"role": "assistant", "content": "Noted."}]
    history += [{"role": "user", "content": "The current project code is JADE42."},
                {"role": "assistant", "content": "The project code is JADE42."}]
    return {
        "reasoning": {"title": "Reasoning / constraints", "prompts": [
            "Three boxes hold 2, 3 and 5 balls. Move exactly one ball so two boxes have equal counts. Give the resulting counts and identify the move.",
            "A train goes 60 km at 30 km/h and returns 60 km at 60 km/h. What is its average speed for the full trip? Show the short calculation."],
            "checks": [None, "40"]},
        "analysis": {"title": "Analysis / missing evidence", "prompts": [
            "Experiment A has 18 successes out of 20; B has 80 out of 100. Compare observed rates, recommend what to test next, and explain why this alone does not prove A is better.",
            "A report says sales doubled after a redesign. List two alternative explanations and one measurement that would help separate correlation from causation."],
            "checks": [None, None]},
        "context": {"title": "Context / trimming / updated facts", "history": history, "prompts": [
            "What is the current project code? Reply with the code only.",
            "Update: the project code is now CEDAR9. Reply with the new code only.",
            "What is the current project code? Reply with the code only."],
            "checks": ["JADE42", "CEDAR9", "CEDAR9"]},
        "overflow": {"title": "Context overflow / trimming OFF", "history": history,
                     "trim_history": False, "expected_error": True,
                     "prompts": ["What is the project code?"], "checks": [None]},
        "original": {"title": "Original prompt / baseline", "prompts": [REWRITE_TASK], "checks": ["30"]},
        "refined": {"title": "Same prompt / refinement ON", "refine_prompt": True,
                    "prompts": [REWRITE_TASK], "checks": ["30"]},
    }


def agent_cases():
    return {
        "draft": {"title": "Agent 1 / draft solution", "model": "llama8b",
                  "prompts": ["A shop buys 12 items at $8 each and sells 9 at $14 each. Draft a concise financial analysis: revenue, cost of all stock, cash profit, and unsold stock. State assumptions and verify the arithmetic."], "checks": [None]},
        "critique": {"title": "Agent 2 / independent critique", "model": "qwen3b", "parent": "draft",
                     "prompts": ["Review the draft below. Check the arithmetic, inventory accounting and distinction between cash profit and profit on sold goods. Identify errors or ambiguities, then give concrete corrections. Do not assume the draft is correct."], "checks": [None]},
        "revise": {"title": "Agent 1 / revise from critique", "model": "llama8b", "parent": "critique",
                   "prompts": ["Use the prior draft and critique below to produce a corrected concise answer. Clearly separate cash profit, profit on sold units and remaining inventory. Verify each number."], "checks": [None]},
        "loop": {"title": "Agent loop / answer → validator → retry", "model": "llama8b", "parent": "revise",
                 "prompts": ["Use the analysis below, but independently verify it. For 12 items bought at $8 and 9 sold at $14, return ONLY JSON with numeric keys revenue, cost, profit. Profit means revenue minus cost of ALL 12 items."], "checks": [None]},
    }


def all_cases():
    return {**cases(), **agent_cases()}


def stamp(message):
    print(f"[{datetime.now().astimezone():%H:%M:%S}] {message}", flush=True)


def compact(text, limit=210):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def worker(name):
    """Invoked by the matrix workflow, exclusively on a GitHub worker."""
    from gh_ai_runner.runner import INFERENCE_SCRIPT
    from gh_inference import MODELS, Limits
    spec = dict(all_cases()[name])
    spec["prompts"] = list(spec["prompts"])
    model = spec.get("model", MODEL)
    inherited = {}
    if spec.get("parent"):
        inherited = json.loads(Path("incoming/report.json").read_text())
        if not inherited.get("passed"):
            raise ValueError("Upstream agent failed; refusing to build on an incomplete result")
        evidence = inherited.get("handoff", []) + [{"agent": inherited["case"], "answer": inherited["result"]["output"]}]
        spec["prompts"][0] += "\n\nPRIOR AGENT OUTPUTS (evidence to verify):\n" + json.dumps(evidence)
    worker_started = datetime.now().astimezone().isoformat(timespec="seconds")
    cfg = {"model": model, "n_ctx": CONTEXT, "max_tokens": OUTPUT,
           "temperature": 0, "cache_weights": True,
           # Llama 3.1 8B GQA: 32 layers × 8 KV heads × 128 dimensions × K/V × 2 bytes.
           "limits": Limits(ram_per_1k_ctx_gb=.134217728 if model == "llama8b" else None).to_dict(),
           "generation_options": {"seed": 42},
           "agent": {"tasks": spec["prompts"][1:], "history": spec.get("history", []),
                     "remember": True, "max_history_turns": None,
                     "trim_history": spec.get("trim_history", True),
                     "refine_prompt": spec.get("refine_prompt", False), "track_time": True}}
    os.environ.update({"JOB_ID": name, "ATTEMPT": "1", "BACKEND": "local",
                       "LOCAL_MODEL": model, "CACHE": "true", "SYSTEM": SYSTEM,
                       "PROMPT": spec["prompts"][0], "CONFIG_JSON": json.dumps(cfg)})
    Path("worker.py").write_text(INFERENCE_SCRIPT, encoding="utf-8")
    stamp(f"{spec['title']} | {MODELS[model]['name']} | context={CONTEXT}")
    # Separate process makes an overflowing test release its model RAM reliably.
    if name == "loop":
        # Load once; feed actual answers and deterministic feedback into up to 3 calls.
        namespace = {}
        exec(compile(INFERENCE_SCRIPT.split("\nstarted = time.monotonic()", 1)[0], "resident_worker", "exec"), namespace)
        namespace["config"] = cfg
        items, prompts, accepted = [], [], False
        prompt = spec["prompts"][0]
        try:
            for iteration in range(3):
                stamp(f"Feedback loop: iteration {iteration + 1}/3")
                prompts.append(prompt)
                response = namespace["run_agent"]({**cfg["agent"], "tasks": [], "history": []},
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], cfg)
                item = response["results"][0]
                items.append(item)
                try:
                    candidate = json.loads(re.search(r"\{[^{}]*\}", item["output"])[0])
                    accepted = all(type(candidate.get(k)) in (int, float) and candidate[k] == v
                                   for k, v in {"revenue": 126, "cost": 96, "profit": 30}.items())
                except (ValueError, TypeError):
                    accepted = False
                if accepted:
                    break
                prompt = ("Your previous response failed a deterministic check. Return only a JSON object: "
                          "revenue = 9*14; cost = 12*8; profit = revenue-cost.\nPrevious response:\n" + item["output"])
            result = {"agent": {"results": items}, "output": items[-1]["output"],
                      "validator_passed": accepted, "iterations": len(items)}
            if not accepted:
                result["error"] = {"message": "Validator failed after 3 iterations"}
            Path("result.json").write_text(json.dumps(result))
            spec["prompts"], spec["checks"] = prompts, [None] * len(prompts)
            execution = SimpleNamespace(returncode=0 if accepted else 1, stderr="")
        except Exception as error:
            Path("result.json").write_text(json.dumps({"error": {"message": str(error)}}))
            execution = SimpleNamespace(returncode=1, stderr="")
    else:
        execution = subprocess.run([sys.executable, "worker.py"], capture_output=True, text=True)
    result = json.loads(Path("result.json").read_text()) if Path("result.json").exists() else {
        "error": {"message": "Worker exited without result.json"}}
    error = result.get("error")
    expected = bool(spec.get("expected_error"))
    context_error = bool(error and any(word in str(error).lower() for word in
                                     ["context", "n_ctx", "token count", "exceed"]))
    okay = (bool(execution.returncode) and context_error) if expected else not execution.returncode and not error
    checks = []
    if not expected and len(result.get("agent", {}).get("results", [])) != len(spec["prompts"]):
        okay = False
    for item, needle in zip(result.get("agent", {}).get("results", []), spec["checks"]):
        checks.append(None if needle is None else needle.lower() in item["output"].lower())
    if any(value is False for value in checks):
        okay = False
    report = {"case": name, "title": spec["title"], "passed": bool(okay),
              "model": model, "context": CONTEXT, "max_output": OUTPUT,
              "handoff": inherited.get("handoff", []) + ([{"agent": inherited["case"], "answer": inherited["result"]["output"]}] if inherited else []),
              "started_at": worker_started,
              "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "expected_overflow": expected, "prompts": spec["prompts"],
              "checks": checks, "result": result}
    Path("report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [f"## {spec['title']}", f"**{'PASS' if okay else 'FAIL'}** · {model} · {CONTEXT} context tokens", ""]
    if error:
        rows.append("Expected overflow rejection." if okay else compact(error))
    for i, item in enumerate(result.get("agent", {}).get("results", [])):
        t = item["timings"]
        rows += [f"### Prompt {i + 1}", spec["prompts"][i], "", "### Response", item["output"], "",
                 f"Tokens: {item['usage']['total_tokens']} · generation {t['generation_seconds']:.2f}s · "
                 f"load {t['load_seconds']:.2f}s · trimmed {item['dropped_history_turns']} turns", ""]
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text("\n".join(rows), encoding="utf-8")
    stamp(f"{'PASS' if okay else 'FAIL'} — full prompt/response comparison is in this job's summary and artifact.")
    if not okay and execution.stderr:
        print(execution.stderr[-1800:], flush=True)
    return 0 if okay else 1


WORKFLOW = '''name: 8B capability showcase
run-name: 8B showcase ${{ inputs.ticket }}
on:
  workflow_dispatch:
    inputs:
      ticket: {required: true, description: Unique run ID}
      parallel: {required: true, default: '6', description: Simultaneous independent test workers}
      context: {required: true, default: '4096', description: Context tokens per model}
      output: {required: true, default: '512', description: Maximum response tokens}
permissions:
  contents: read
jobs:
  test:
    name: ${{ matrix.case }} / Llama 8B
    runs-on: ubuntu-latest
    timeout-minutes: 40
    strategy:
      fail-fast: false
      max-parallel: ${{ fromJSON(inputs.parallel) }}
      matrix:
        case: [reasoning, analysis, context, overflow, original, refined]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Restore quantized weights
        uses: actions/cache@v4
        with:
          path: model_cache
          key: showcase-llama8b-__SHA__
      - name: Install CPU runtime on GitHub
        run: |
          python -m pip install gh-inference==0.5.0 -q
          python -m pip install --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu 'llama-cpp-python>=0.3.16' -q
      - name: Run test and write readable summary
        env:
          CASE: ${{ matrix.case }}
          SHOWCASE_CONTEXT: ${{ inputs.context }}
          SHOWCASE_OUTPUT: ${{ inputs.output }}
        run: python showcase.py --worker "$CASE"
      - name: Save full results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: showcase-${{ matrix.case }}
          path: |
            report.json
            result.json
          retention-days: 3
'''


def workflow_text():
    from gh_inference import MODELS
    workflow = WORKFLOW.replace("__SHA__", MODELS[MODEL]["sha256"])
    steps = WORKFLOW.split("    steps:\n", 1)[1]
    for name, spec in agent_cases().items():
        job_steps = steps.replace("${{ matrix.case }}", name)
        job_steps = job_steps.replace("showcase-llama8b-__SHA__",
                                      "showcase-" + spec["model"] + "-" + MODELS[spec["model"]]["sha256"])
        if spec.get("parent"):
            download = ("      - name: Read preceding agent output\n"
                        "        uses: actions/download-artifact@v4\n"
                        "        with:\n"
                        f"          name: showcase-{spec['parent']}\n"
                        "          path: incoming\n")
            job_steps = job_steps.replace("      - name: Run test and write readable summary\n",
                                          download + "      - name: Run test and write readable summary\n")
        workflow += f"  {name}:\n    name: '{spec['title']}'\n"
        if spec.get("parent"):
            workflow += f"    needs: [{spec['parent']}]\n"
        workflow += "    runs-on: ubuntu-latest\n    timeout-minutes: 40\n    steps:\n" + job_steps
    return workflow


def client(args):
    import getpass
    import requests
    from gh_inference import MODELS
    token = os.environ.get("GH_AI_TOKEN") or getpass.getpass("GitHub token (hidden): ").strip()
    if not token:
        raise ValueError("A GitHub token is required")
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"})

    def api(method, path, body=None, allow_missing=False):
        response = session.request(method, "https://api.github.com" + path, json=body, timeout=45)
        if allow_missing and response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json() if response.content else None

    owner = api("GET", "/user")["login"]
    root = f"/repos/{owner}/{args.repo}"
    repo = api("GET", root, allow_missing=True)
    if repo is None:
        repo = api("POST", "/user/repos", {"name": args.repo, "private": True, "auto_init": True})
    if not repo["private"]:
        raise ValueError("Use a private dedicated repo; this showcase uploads prompts and responses")
    branch = repo["default_branch"]
    head = api("GET", root + "/git/ref/heads/" + branch)["object"]["sha"]
    script = Path(__file__).read_text(encoding="utf-8")
    workflow = workflow_text()
    tree = api("POST", root + "/git/trees", {"base_tree": head, "tree": [
        {"path": path, "mode": "100644", "type": "blob", "content": content}
        for path, content in [("showcase.py", script), (".github/workflows/showcase.yml", workflow)]]})
    commit = api("POST", root + "/git/commits", {"message": "Install standalone 8B capability showcase",
                                                "tree": tree["sha"], "parents": [head]})
    api("PATCH", root + "/git/refs/heads/" + branch, {"sha": commit["sha"], "force": False})
    ticket = uuid.uuid4().hex[:12]
    workflow_path = root + "/actions/workflows/showcase.yml"
    deadline = time.monotonic() + 180
    while api("GET", workflow_path, allow_missing=True) is None:
        if time.monotonic() > deadline:
            raise TimeoutError("Workflow registration timed out; check Actions are enabled")
        time.sleep(4)
    api("POST", workflow_path + "/dispatches", {"ref": branch, "inputs": {
        "ticket": ticket, "parallel": str(args.parallel),
        "context": str(args.context), "output": str(args.max_tokens)}})
    stamp(f"8B showcase · six independent tests (parallel={args.parallel}) + four linked agent jobs")
    stamp(f"Context {args.context:,} · output up to {args.max_tokens:,} · graph: draft → critique → revise → loop")
    stamp(f"Workflow: https://github.com/{owner}/{args.repo}/actions/workflows/showcase.yml")
    run = None
    deadline = time.monotonic() + 10800
    seen = {}
    while time.monotonic() < deadline:
        if run is None:
            runs = api("GET", workflow_path + "/runs?event=workflow_dispatch&per_page=30")["workflow_runs"]
            run = next((r for r in runs if ticket in r.get("display_title", "")), None)
            if run:
                stamp("LIVE RUN: " + run["html_url"])
        if run:
            jobs = api("GET", root + f"/actions/runs/{run['id']}/jobs?per_page=100")["jobs"]
            for job in jobs:
                state = job.get("conclusion") or job["status"]
                if seen.get(job["id"]) != state:
                    stamp(f"{job['name']}: {state.replace('_', ' ')}")
                    seen[job["id"]] = state
            run = api("GET", root + f"/actions/runs/{run['id']}")
            if run["status"] == "completed":
                break
        time.sleep(12)
    else:
        raise TimeoutError("Stopped waiting after three hours; remote jobs are not cancelled")
    artifacts = api("GET", root + f"/actions/runs/{run['id']}/artifacts?per_page=100")["artifacts"]
    reports = {}
    for artifact in artifacts:
        if not artifact["name"].startswith("showcase-"):
            continue
        response = session.get(artifact["archive_download_url"], timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if "report.json" not in archive.namelist():
                continue
            if archive.getinfo("report.json").file_size > 10_000_000:
                raise ValueError("Unexpectedly large report")
            report = json.loads(archive.read("report.json"))
            reports[report["case"]] = report
    for case_index, (name, spec) in enumerate(all_cases().items()):
        report = reports.get(name)
        stamp(f"\n{'PASS' if report and report['passed'] else 'FAIL'} | {spec['title']}")
        if not report:
            print("  No report. Inspect this job's setup/runtime logs.")
            continue
        result = report["result"]
        print(f"  Worker: {report['started_at']} → {report['finished_at']}")
        if report["expected_overflow"]:
            print("  Expected: reject input beyond allocated context without removing history.")
            print("  Actual:  " + compact(result.get("error", "No rejection")))
        for i, item in enumerate(result.get("agent", {}).get("results", [])):
            prompt = report["prompts"][i]
            if name in ("original", "refined") and not args.details:
                prompt = "[35 filler repetitions] " + REWRITE_TASK.split("Please be clear and concise. " * 35)[-1]
            display = str if args.details else compact
            print(f"  P{i + 1} → {display(prompt)}\n  R{i + 1} ← {display(item['output'])}")
            t, u = item["timings"], item["usage"]
            print(f"       {u['prompt_tokens']} in / {u['completion_tokens']} out · "
                  f"generate {t['generation_seconds']:.1f}s · load {t['load_seconds']:.1f}s · "
                  f"reuse {'yes' if item['model_reused'] else 'no'} · "
                  f"cache {'hit' if item['weights_cached'] else 'miss'} · "
                  f"trimmed {item['dropped_history_turns']} turns")
            refinement = item["refinement"]
            if refinement["attempted"]:
                print(f"       Refiner: {refinement['original_tokens']} → {refinement['refined_tokens']} prompt tokens; "
                      f"{t['refinement_seconds']:.1f}s extra; {refinement['reason']}")
        if "iterations" in result:
            print(f"  Feedback loop: {result['iterations']}/3 iterations · validator "
                  f"{'accepted' if result['validator_passed'] else 'rejected'}")
        if sys.stdin.isatty() and not args.no_pause and case_index < len(all_cases()) - 1:
            input("  Enter → next result ")
    print("\nChecks are targeted answer markers and expected errors, not a reasoning-quality benchmark.")
    print("Compare original/refined TOTAL tokens and timing in their artifacts; a shorter prompt can cost more overall.")
    stamp(f"Overall workflow: {run['conclusion']} | full results: {run['html_url']}")
    return 0 if len(reports) == len(all_cases()) and all(r["passed"] for r in reports.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="gh-inference-showcase")
    parser.add_argument("--parallel", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--context", type=int, default=4096, help="Allocated context tokens; larger windows need more RAM")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum output tokens per answer")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--no-pause", action="store_true", help="Print all result cards without waiting for Enter")
    parser.add_argument("--worker", choices=list(all_cases()), help=argparse.SUPPRESS)
    options = parser.parse_args()
    try:
        if options.context < 256 or not 1 <= options.max_tokens < options.context:
            raise ValueError("Use context >= 256 and 1 <= max-tokens < context")
        sys.exit(worker(options.worker) if options.worker else client(options))
    except Exception as error:
        stamp(f"Stopped: {type(error).__name__}: {error}")
        sys.exit(1)
