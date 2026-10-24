"""Run yourself: python examples/console_tour.py [--full] [--cancel].

Client orchestration runs on your computer; every model runs on GitHub Actions.
Default: three remote jobs. --full adds four jobs; --cancel adds one more.
"""
import argparse
from dataclasses import asdict
import getpass
import json
import os
from pathlib import Path
import time
import threading

from gh_inference import (
    Agent, GitHubAIRunner, InferenceRequest, Limits, MODELS, RetryPolicy,
    list_quantizations, model_resources, resolve_model, search_models,
)


RAW = False
OUTPUT_LOCK = threading.RLock()


def heading(text):
    print(f"\n{'─' * 78}\n{text}\n{'─' * 78}", flush=True)


def label(value):
    return str(value).replace("_", " ").capitalize()


def readable(value, key=""):
    if value is None:
        return "Not set"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if key.endswith("_bytes"):
            return f"{value / 1e9:.2f} GB"
        if key.endswith("_seconds"):
            return f"{value:.2f}s"
        if key.endswith("_gb"):
            return f"{value:.2f} GB"
        return f"{value:,}"
    return str(value)


def show(value, indent=0):
    """Readable summaries by default; --raw preserves the complete JSON view."""
    if RAW and indent == 0:
        print(json.dumps(value, indent=2, default=str), flush=True)
        return
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                print(f"{prefix}{label(key)}:", flush=True)
                show(item, indent + 2)
            else:
                print(f"{prefix}{label(key):28} {readable(item, str(key))}", flush=True)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                show(item, indent + 2)
                print()
            else:
                print(f"{prefix}• {item}", flush=True)
    else:
        print(prefix + str(value), flush=True)


def watch_job(job):
    """Observe only; monitoring never launches or cancels inference."""
    previous = None
    linked_run = None
    warned = False
    started = time.monotonic()
    while True:
        try:
            state = job.refresh()
            with OUTPUT_LOCK:
                if state.value != previous:
                    print(f"\n  [Job {job.job_id[:8]} · {time.monotonic() - started:.0f}s] "
                          f"{label(state.value)}", flush=True)
                    previous = state.value
                if job.actions_url and job.actions_url != linked_run:
                    print(f"  LIVE RUN → {job.actions_url}\n", flush=True)
                    linked_run = job.actions_url
            if state.is_terminal:
                return
        except Exception:
            if not warned:
                with OUTPUT_LOCK:
                    print("  Live status temporarily unavailable; the main call is still waiting.", flush=True)
                warned = True
        time.sleep(8)


class VisibleRunner(GitHubAIRunner):
    def submit(self, *args, **kwargs):
        job = super().submit(*args, **kwargs)
        with OUTPUT_LOCK:
            print(f"\n  Submitted: {job.job_id[:8]}", flush=True)
            print(f"  ALL RUNS → https://github.com/{self.username}/{self.repo_name}/actions", flush=True)
        threading.Thread(target=watch_job, args=(job,), daemon=True).start()
        return job


def diagnostics(agent):
    for index, result in enumerate(agent.last_results, 1):
        print(f"\n  ANSWER {index}\n", flush=True)
        print("\n".join("  " + line for line in result.output.strip().splitlines()))
        if RAW:
            show(asdict(result))
            continue
        print(f"\n  Tokens: {result.usage.prompt_tokens:,} input + "
              f"{result.usage.completion_tokens:,} output = {result.usage.total_tokens:,} total")
        print(f"  Same loaded model reused: {readable(result.model_reused)}  |  "
              f"Weights restored from cache: {readable(result.weights_cached)}")
        if result.timings:
            t = result.timings
            print(f"  Download {t.download_seconds:.2f}s  ·  Load {t.load_seconds:.2f}s  ·  "
                  f"Generate {t.generation_seconds:.2f}s  ·  Refine {t.refinement_seconds:.2f}s")
        print(f"  Finished: {result.finish_reason}  |  Old turns removed: {result.dropped_history_turns}")
        r = result.refinement
        if r.attempted or r.reason != "disabled":
            print(f"  Refiner: {r.reason}")
            print(f"  Main prompt: {r.original_tokens} → {r.refined_tokens} tokens; "
                  f"saved {r.tokens_saved}. Refinement itself used {r.usage.total_tokens} tokens.")
    print("Exact GitHub run:", agent.last_job.actions_url, flush=True)
    # Already retrieved by Agent; this does not start another inference.
    remote = agent.last_job.result()
    print("\n  GITHUB WORKER", flush=True)
    if RAW:
        show({"resources": asdict(remote.resources), "integrity": asdict(remote.integrity)})
    else:
        r = remote.resources
        print(f"  {r.runner_os} · {r.cpu_count} CPUs · {r.memory_total_bytes / 1e9:.2f} GB total RAM")
        print(f"  Peak process memory: {r.peak_memory_bytes / 1e9:.2f} GB · "
              f"Free disk at measurement: {r.disk_free_bytes / 1e9:.2f} GB")
        print("  Request/output integrity: verified by the client")
        if agent.last_result.timings:
            print(f"  Whole client operation: {agent.last_result.timings.remote_seconds:.1f}s "
                  "(includes setup, queue, execution and retrieval)")


def main():
    global RAW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also test independent parallel jobs and model comparison")
    parser.add_argument("--cancel", action="store_true", help="Dispatch an extra job and request cancellation")
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--repo", default="gh-inference-playground")
    parser.add_argument("--raw", action="store_true", help="Show complete JSON diagnostics")
    args = parser.parse_args()
    RAW = args.raw
    if not os.environ.get("GH_AI_TOKEN"):
        os.environ["GH_AI_TOKEN"] = getpass.getpass("GitHub token (hidden; repo + workflow scopes): ").strip()
        if not os.environ["GH_AI_TOKEN"]:
            raise SystemExit("A GitHub token is required.")

    heading("MODEL DISCOVERY — metadata only")
    print("Built-in aliases:", ", ".join(MODELS))
    print("Public Qwen GGUF repositories:")
    show(search_models("Qwen", limit=5))
    print("Published variants of your selection:")
    variants = list_quantizations(args.model)
    if RAW:
        show(variants)
    else:
        for name, variant in variants.items():
            print(f"  {variant['size_bytes'] / 1e9:5.2f} GB  {name}  ({len(variant['files'])} file(s))")
    selected = resolve_model(args.model, args.quantization)
    print("Resolved selection:", selected)
    show(model_resources(selected, n_ctx=2048))

    heading("CONNECT — private runner repository")
    runner = VisibleRunner(
        repo_name=args.repo, private=True, verbose=RAW,
        metadata_path=Path.home() / ".gh-inference-tour" / "jobs.sqlite3",
        completion_timeout=1800,
    )
    runner.prepare()
    print(f"Runner repository: https://github.com/{runner.username}/{runner.repo_name}")
    print(f"ALL RUNS → https://github.com/{runner.username}/{runner.repo_name}/actions")
    print(f"WORKFLOW → https://github.com/{runner.username}/{runner.repo_name}/actions/workflows/inference.yml")
    print("Open a LIVE RUN link below to watch setup, cache restoration, inference and result upload.")
    agent = Agent(runner=runner, model=selected, n_ctx=2048, max_tokens=96,
                  temperature=0, track_time=True,
                  instructions="You are a helpful assistant. Answer briefly.",
                  generation_options={"seed": 42, "top_p": .9})

    heading("JOB 1 — one model load, conversation, timing")
    prompts = ["Remember: my project is called Cedar. Acknowledge briefly.",
               "What is my project called?", "What is 2 plus 2?"]
    for index, prompt in enumerate(prompts, 1):
        print(f"Input {index}: {prompt}")
    agent.run_many(prompts, timeout=1800)
    diagnostics(agent)
    print("Reuse checks:", [result.model_reused for result in agent.last_results])
    print("Expected: False, True, True. Later prompts should have load_seconds=0.")

    heading("JOB 2 — new worker, cached weights, retained conversation")
    print("Input: What project name did I ask you to remember?")
    agent("What project name did I ask you to remember?", timeout=1800)
    diagnostics(agent)
    print("weights_cached can be True; model_reused is False because this is a new worker.")
    print("Cache restoration is best effort. Compare download_seconds and load_seconds.")
    agent.reset()
    print("Conversation reset. No new job was started by reset().")

    heading("JOB 3 — optional prompt refiner and sampling options")
    refiner = Agent(runner=runner, model=selected, n_ctx=2048, max_tokens=96,
                    refine_prompt=True, track_time=True, remember=False,
                    instructions="Answer briefly.")
    prompt = ("Please be clear and concise. " * 40
              + "Explain a Python list in one sentence. Include the number 42.")
    print("Input: 40 repetitions of 'Please be clear and concise.' followed by:")
    print("       Explain a Python list in one sentence. Include the number 42.")
    if RAW:
        print("Full input:", prompt)
    refiner(prompt, timeout=1800)
    diagnostics(refiner)
    print("Refinement can fall back to the original. Check attempted/applied/reason and usage.")

    heading("LIMITS — show configuration without launching jobs")
    if RAW:
        show({"standard": asdict(Limits()), "experimental": asdict(Limits.unrestricted())})
    else:
        print("This tour: 2,048 total context tokens · up to 96 output tokens · CPU model in RAM.")
        print("Standard policy checks model ceilings/resources and trims old history to fit.")
        print("Experimental policy removes those project checks; --raw shows every setting.")
    print("Agent.experimental(n_ctx=32768, max_tokens=None) permits larger-context attempts.")
    print("Actual model support, RAM, backend and GitHub transport limits still apply.")
    print("trim_history=False retains history; it does not enlarge n_ctx.")

    if args.full:
        heading("JOBS 4–5 — async batch, separate GitHub workers")
        batch = runner.submit_batch([
            InferenceRequest("Give one useful Python tip.", model=selected, n_ctx=2048, max_tokens=64),
            InferenceRequest("Give one useful Git tip.", model=selected, n_ctx=2048, max_tokens=64,
                             retry_policy=RetryPolicy(max_retries=1)),
        ])
        show({key: value.value for key, value in batch.refresh().items()})
        for job, result in zip(batch.jobs, batch.results(timeout=1800)):
            print("GitHub run:", job.actions_url)
            show(result.to_dict())

        heading("JOBS 6–7 — compare two models")
        comparison = runner.compare("What is 2 plus 2? Answer briefly.",
                                    ["qwen0.5b", "tinyllama"], n_ctx=2048, max_tokens=48)
        for model, result in comparison.results_by_model(timeout=1800).items():
            print(model, "=>", result.output)
            show({"usage": asdict(result.usage), "resources": asdict(result.resources)})

    if args.cancel:
        heading("OPTIONAL CANCELLATION")
        job = runner.submit("Explain sorting algorithms.", model=selected, n_ctx=2048, max_tokens=256)
        print("Cancellation accepted:", job.cancel())
        print("GitHub run:", job.actions_url)
        print("A job that already finished cannot be cancelled.")

    heading("JOB HISTORY + RESUME")
    for record in runner.list_jobs(limit=12):
        if RAW:
            show(asdict(record))
        else:
            print(f"  {record.job_id[:8]}  {record.status:12}  "
                  f"{record.total_tokens if record.total_tokens is not None else '—'} tokens  {record.model}")
            if record.run_id:
                print(f"    https://github.com/{record.owner}/{record.repo_name}/actions/runs/{record.run_id}")
    resumed = runner.resume(agent.last_job.job_id)
    print("Restored job:", resumed.job_id, "status:", resumed.refresh().value)
    print("Resume restores the job handle, not the agent's conversation.")
    print("Done. Every model download/load/generation above happened on GitHub.")


if __name__ == "__main__":
    started = time.monotonic()
    try:
        main()
    except Exception as error:
        print(f"\nTour stopped: {type(error).__name__}: {error}", flush=True)
        print("Inspect the last printed Actions URL. A client timeout does not cancel the worker.")
        raise SystemExit(1)
    finally:
        print(f"\nTotal client wall time: {time.monotonic() - started:.1f}s", flush=True)
