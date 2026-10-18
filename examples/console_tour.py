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

from gh_inference import (
    Agent, GitHubAIRunner, InferenceRequest, Limits, MODELS, RetryPolicy,
    list_quantizations, model_resources, resolve_model, search_models,
)


def heading(text):
    print(f"\n{'=' * 16} {text} {'=' * 16}", flush=True)


def show(value):
    print(json.dumps(value, indent=2, default=str), flush=True)


class VisibleRunner(GitHubAIRunner):
    def submit(self, *args, **kwargs):
        job = super().submit(*args, **kwargs)
        print(f"Dispatched job: {job.job_id}", flush=True)
        print(f"Watch GitHub: https://github.com/{self.username}/{self.repo_name}/actions", flush=True)
        return job


def diagnostics(agent):
    for index, result in enumerate(agent.last_results, 1):
        print(f"\nAnswer {index}: {result.output}", flush=True)
        show(asdict(result))
    print("Exact GitHub run:", agent.last_job.actions_url, flush=True)
    # Already retrieved by Agent; this does not start another inference.
    remote = agent.last_job.result()
    print("Worker resources and integrity checks:", flush=True)
    show({"resources": asdict(remote.resources), "integrity": asdict(remote.integrity)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also test independent parallel jobs and model comparison")
    parser.add_argument("--cancel", action="store_true", help="Dispatch an extra job and request cancellation")
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--repo", default="gh-inference-playground")
    args = parser.parse_args()
    if not os.environ.get("GH_AI_TOKEN"):
        os.environ["GH_AI_TOKEN"] = getpass.getpass("GitHub token (hidden; repo + workflow scopes): ").strip()
        if not os.environ["GH_AI_TOKEN"]:
            raise SystemExit("A GitHub token is required.")

    heading("MODEL DISCOVERY — metadata only")
    print("Built-in aliases:", ", ".join(MODELS))
    print("Public Qwen GGUF repositories:")
    show(search_models("Qwen", limit=5))
    print("Published variants of your selection:")
    show(list_quantizations(args.model))
    selected = resolve_model(args.model, args.quantization)
    print("Resolved selection:", selected)
    show(model_resources(selected, n_ctx=2048))

    heading("CONNECT — private runner repository")
    runner = VisibleRunner(
        repo_name=args.repo, private=True, verbose=True,
        metadata_path=Path.home() / ".gh-inference-tour" / "jobs.sqlite3",
        completion_timeout=1800,
    )
    runner.prepare()
    print(f"Runner repository: https://github.com/{runner.username}/{runner.repo_name}")
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
    print("Input:", prompt)
    refiner(prompt, timeout=1800)
    diagnostics(refiner)
    print("Refinement can fall back to the original. Check attempted/applied/reason and usage.")

    heading("LIMITS — show configuration without launching jobs")
    show({"standard": asdict(Limits()), "experimental": asdict(Limits.unrestricted())})
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
        show(asdict(record))
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
