"""Text in, text out. All model work happens on GitHub Actions."""
from __future__ import annotations

import math
import os
import threading
import time

from .agent_types import AgentResult, RefinementInfo
from .client import GitHubAIRunner
from .models import MODELS
from .hub import resolve_model
from .limits import Limits, validate_options
from .types import AgentTimings, TokenUsage
from .pipeline import normalize_pipeline


class Agent:
    """A reusable GitHub connection with optional conversation and timing.

    Set GH_AI_TOKEN once, then use Agent().run("your prompt"). run_many() keeps
    the model loaded for all prompts within one GitHub job.
    """

    def __init__(self, github_token=None, *, model="qwen0.5b",
                 instructions="You are a helpful assistant.", track_time=False,
                 refine_prompt=False, remember=True, max_history_turns=6, trim_history=True,
                 n_ctx=None, max_tokens=512, temperature=.7,
                 limits=None, cache=True, model_options=None, generation_options=None,
                 repo_name="ai-inference-runner", runner=None,
                 quantization=None, filename=None, revision="main"):
        model = resolve_model(model, quantization, filename=filename, revision=revision)
        if not isinstance(instructions, str):
            raise TypeError("instructions must be a string")
        for name, value in (("track_time", track_time), ("refine_prompt", refine_prompt),
                            ("remember", remember), ("trim_history", trim_history), ("cache", cache)):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool")
        if max_history_turns is not None and (type(max_history_turns) is not int or max_history_turns < 1):
            raise ValueError("max_history_turns must be a positive integer or None")
        if limits is not None and not isinstance(limits, Limits):
            raise TypeError("limits must be a Limits instance")
        self.limits = limits if limits is not None else (runner.limits if isinstance(runner, GitHubAIRunner) else Limits())
        validate_options(model_options, generation_options)
        self.model_options, self.generation_options = dict(model_options or {}), dict(generation_options or {})
        self.cache = cache
        self.model, self.instructions = model, instructions
        self.track_time, self.refine_prompt, self.remember = track_time, refine_prompt, remember
        self.max_history_turns = max_history_turns
        self.trim_history = trim_history
        self.n_ctx, self.max_tokens, self.temperature = n_ctx, max_tokens, temperature
        self.last_result = None
        self.last_results = []
        self.last_job = None
        self._history = []
        self._lock = threading.RLock()
        if runner is None:
            token = github_token or os.environ.get("GH_AI_TOKEN")
            if not token:
                raise ValueError("Set GH_AI_TOKEN once, or pass Agent(github_token=...). Models run only on GitHub.")
            runner = GitHubAIRunner(token, repo_name=repo_name, metadata_path=None, limits=self.limits)
        self.runner = runner

    @classmethod
    def experimental(cls, **kwargs):
        """Remove project ceilings and history trimming; external limits still apply."""
        options = {"limits": Limits.unrestricted(), "trim_history": False,
                   "max_history_turns": None}
        options.update(kwargs)
        return cls(**options)

    def connect(self):
        """Optional: prepare the repository now. run() also does this automatically."""
        self.runner.prepare()
        return self

    def reset(self):
        """Forget this conversation without changing the runner repository."""
        with self._lock:
            self._history = []

    def __call__(self, prompt, **kwargs):
        return self.run(prompt, **kwargs)

    def run(self, prompt, *, timeout=None, track_time=None):
        """Return plain text. Inspect last_result for optional diagnostics."""
        return self.run_many([prompt], timeout=timeout, track_time=track_time)[0]

    def run_many(self, prompts, *, timeout=None, track_time=None):
        """Return text answers from one GitHub job; load the model only once."""
        return self._run_many(prompts, timeout=timeout, track_time=track_time)

    def discuss(self, prompt, *, agents=4, rounds=3, peer_chars=240,
                topology="all", timeout=None, track_time=None):
        """Return rounds of replies from one resident model on one GitHub worker.

        Logical agents take turns sequentially. Each sees only the previous round,
        not this Agent's conversation. Set Limits.max_prompts >= agents * rounds.
        Ring topology sends only the previous neighbor's reply to each agent.
        """
        for name, value in (("agents", agents), ("rounds", rounds), ("peer_chars", peer_chars)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if topology not in ("all", "ring"):
            raise ValueError("topology must be all or ring")
        if self.limits.max_prompts is not None and agents * rounds > self.limits.max_prompts:
            raise ValueError("Discussion exceeds Limits.max_prompts; raise it to agents * rounds")
        interaction = dict(agents=agents, rounds=rounds, peer_chars=peer_chars, topology=topology)
        outputs = self._run_many([prompt], timeout=timeout, track_time=track_time,
                                 discussion=interaction)
        return [outputs[i:i + agents] for i in range(0, len(outputs), agents)]

    def pipeline(self, prompt, steps, *, timeout=None, track_time=None):
        """Run routed steps in one job, returning outputs in step order.

        Steps name an agent, an instruction, and earlier output indices in inputs.
        Omitted inputs default to the preceding step. Only explicitly selected
        outputs are provided; no implicit per-agent conversation is accumulated.
        """
        plan = normalize_pipeline(steps, self.limits.max_prompts)
        return self._run_many([prompt], timeout=timeout, track_time=track_time, pipeline=plan)

    def review(self, prompt, *, revisions=1, timeout=None, track_time=None):
        """Draft -> review -> revise without restarting the worker or model."""
        if type(revisions) is not int or revisions < 1:
            raise ValueError("revisions must be a positive integer")
        if self.limits.max_prompts is not None and 1 + 2 * revisions > self.limits.max_prompts:
            raise ValueError("Review exceeds configured max_prompts")
        steps = [{"agent": "writer", "instruction": "Complete the shared task.", "inputs": []}]
        for _ in range(revisions):
            draft = len(steps) - 1
            steps.append({"agent": "reviewer", "instruction":
                          "Check the draft against the shared task. Give specific, actionable corrections.",
                          "inputs": [draft]})
            steps.append({"agent": "writer", "instruction":
                          "Revise the draft using the review. Return the complete corrected answer.",
                          "inputs": [draft, draft + 1]})
        return self.pipeline(prompt, steps, timeout=timeout, track_time=track_time)[-1]

    def _run_many(self, prompts, *, timeout=None, track_time=None, discussion=None, pipeline=None):
        if not isinstance(prompts, (list, tuple)) or not prompts:
            raise ValueError("prompts must be a nonempty list or tuple")
        if self.limits.max_prompts is not None and len(prompts) > self.limits.max_prompts:
            raise ValueError(f"prompts exceeds configured max_prompts={self.limits.max_prompts}")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("Every prompt must be a nonempty string")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")
        if track_time is not None and type(track_time) is not bool:
            raise TypeError("track_time must be a bool")
        enabled = self.track_time if track_time is None else track_time
        with self._lock:
            started = time.monotonic()
            config = {
                "tasks": list(prompts[1:]), "refine_prompt": self.refine_prompt,
                "track_time": enabled, "remember": self.remember,
                "max_history_turns": self.max_history_turns,
                "trim_history": self.trim_history,
                "history": self._history if self.remember else [],
            }
            if discussion is not None:
                config.update(discussion=discussion, remember=False, history=[])
            if pipeline is not None:
                config.update(pipeline=pipeline, remember=False, history=[])
            transport_dropped = 0
            while True:
                try:
                    job = self.runner.submit(
                        prompts[0], model=self.model, system=self.instructions,
                        n_ctx=self.n_ctx, max_tokens=self.max_tokens, temperature=self.temperature,
                        cache=self.cache, limits=self.limits,
                        model_options=self.model_options, generation_options=self.generation_options,
                        agent_config=config,
                    )
                    break
                except ValueError as error:
                    if (not self.trim_history or "Combined workflow inputs exceed" not in str(error)
                            or not config["history"]):
                        raise
                    config["history"] = config["history"][2:]
                    transport_dropped += 1
                except Exception as error:
                    if getattr(error, "job", None) is not None:
                        self.last_job = error.job
                    raise
            self.last_job = job
            remote = job.result(timeout=timeout)
            elapsed = time.monotonic() - started
            payload = remote.raw.get("agent")
            expected = (len(pipeline) if pipeline is not None else
                        discussion["agents"] * discussion["rounds"] if discussion else len(prompts))
            if not isinstance(payload, dict) or len(payload.get("results", [])) != expected:
                raise RuntimeError("GitHub worker returned an incomplete agent result")
            results = []
            for item in payload["results"]:
                refinement = dict(item.get("refinement", {}))
                refinement["usage"] = TokenUsage(**refinement.get("usage", {}))
                timings = AgentTimings(**item["timings"]) if enabled else None
                if timings is not None:
                    timings.remote_seconds = elapsed
                results.append(AgentResult(
                    output=item["output"], model=self.model, resources=remote.resources,
                    usage=TokenUsage(**item.get("usage", {})), timings=timings,
                    refinement=RefinementInfo(**refinement), model_reused=item.get("model_reused", False),
                    weights_cached=item.get("weights_cached", False),
                    dropped_history_turns=item.get("dropped_history_turns", 0) + transport_dropped,
                    finish_reason=item.get("finish_reason", "stop"),
                    agent_id=item.get("agent_id", ""), step_index=item.get("step_index"),
                    input_steps=item.get("input_steps", []),
                    handoff_chars=item.get("handoff_chars", 0), handoff_bytes=item.get("handoff_bytes", 0),
                ))
            if self.remember and discussion is None and pipeline is None:
                self._history = payload.get("history", [])
            self.last_results, self.last_result = results, results[-1]
            return [result.output for result in results]
