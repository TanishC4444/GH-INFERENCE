# GH-INFERENCE

**Text in, text out. Quantized models run on GitHub servers.**

Your Python client only connects, submits prompts, and retrieves answers. Model
downloads, model loading, inference, and optional prompt refinement all happen
inside GitHub Actions. No model runtime or GPU is required on your machine.

## Start

Install the client in your environment (Python 3.9+):

```bash
pip install gh-inference
export GH_AI_TOKEN="your-github-token"
```

Use a classic GitHub token with `repo` and `workflow` scopes. Do not commit it.

```python
from gh_inference import Agent

agent = Agent()
print(agent.run("Explain recursion briefly."))
```

That is the whole flow. The first call automatically creates or reuses the
`ai-inference-runner` repository, installs the workflow, starts a GitHub worker,
and returns its answer as a plain string. `agent("prompt")` is equivalent.
You can also pass `Agent(github_token=...)` instead of setting the environment.

`agent.connect()` optionally prepares the repository ahead of the first prompt;
it is never required. Connecting does not start or reserve a model worker.

## Built-in writing models

Four tested Q4_K_M writing models are included as ready-to-use aliases:

| Alias | Model | Weight download |
| --- | --- | ---: |
| `glitter-12b` | Gemma 3 Glitter 12B | 7.30 GB |
| `gemma-4-12b` | Gemma 4 12B Instruct | 7.12 GB |
| `gemma-4-styletune-12b` | Gemma 4 12B StyleTune | 7.95 GB |
| `rocinante-x-12b` | Rocinante-X 12B v1 | 7.48 GB |

Downloads are pinned to verified publisher commits and SHA-256 checksums.
These are weights only; runtime and context need additional RAM. Start with a
16 GB runner and a 4,096-token context, as used in our scene comparisons:

```python
from gh_inference import Agent, GitHubAIRunner

writer = Agent(model="glitter-12b", n_ctx=4096,
               runner=GitHubAIRunner(runner_ram_gb=16))
print(writer.run("Write a tense scene between two estranged crewmates."))
```

For either Gemma 4 alias, set
`model_options={"chat_template_options": {"enable_thinking": False}}`
to request direct writing without a thinking phase. These models are useful
candidates, not a guaranteed ranking of literary quality.

## Models and every published GGUF variant

The 16 pinned aliases below remain convenient defaults. Model support is open:
use any public, ungated Hugging Face **GGUF** repository supported by the installed
llama.cpp runtime. Search discovers Qwen, Llama, Gemma, Phi, Mistral, DeepSeek,
SmolLM and other families, with no project parameter-count ceiling. A model being
discoverable does not mean it fits your runner or its architecture is supported.

```python
from gh_inference import Agent, search_models, list_quantizations

print(search_models("Qwen3"))
print(list_quantizations("Qwen/Qwen2.5-0.5B-Instruct-GGUF"))
agent = Agent(model="Qwen/Qwen2.5-0.5B-Instruct-GGUF", quantization="Q8_0")
print(agent("Explain quantization in one sentence."))
```

`quantization` defaults to `Q4_K_M` for repository selections. All published GGUF
labels can be selected, including Q2/Q3/Q4/Q5/Q6/Q8, K-quants, IQ, TQ and floating
point variants when provided and supported by your backend. We do not manufacture
missing variants or convert weights. AWQ/GPTQ/Safetensors require other backends
and are not accepted by this GGUF CPU runner. Model licenses still apply.

For ambiguous repositories use `Agent(model="owner/repo", filename="exact.gguf")`.
You can also paste a Hugging Face repository URL or an exact GGUF `blob`/`resolve`
URL directly into `Agent(model=...)`. File links select that file (and its complete
shard group); repository links default to Q4_K_M. Links are resolved to a pinned
commit and verified checksums, just like repository names.

```python
agent = Agent(model="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF")
```

For your own public single-file GGUF download on another host, register its link
with the publisher's actual SHA-256 and byte size:

```python
from gh_inference import Agent, add_model

model = add_model(
    "https://your-host.example/writer.gguf",
    sha256=publisher_sha256,
    size_bytes=publisher_size_bytes,
    max_n_ctx=8192,  # optional advertised context ceiling
)
agent = Agent(model=model)
```

Registration lasts for the Python process; each submitted job carries the model
metadata to its worker. Weights download only on GitHub and are verified before
loading. Custom hosts must allow unauthenticated HTTPS downloads. The model must
fit the runner and use a GGUF architecture supported by the installed llama.cpp
runtime; a link does not add a new inference backend. Resource checks still apply.

Split GGUF files are grouped, all shards are checked and downloaded, and the first
shard is loaded. Incomplete shard groups are excluded. `list_quantizations` lists
exact filenames and aggregate bytes; projector files are excluded (text only).

Selections resolve metadata once per client process and pin a Hub commit plus
each file's SHA-256 and size. Workers verify downloads before loading. Subsequent
jobs reuse the Actions weights cache when eligible; `run_many()` keeps one model
resident in RAM throughout that job. An Actions job ending releases its RAM.
Use `revision="commit-or-tag"` for reproducibility; `clear_model_cache()` refreshes
cached branch metadata. Hub metadata calls download no model weights locally.

Context defaults to about 8K (or a known smaller ceiling). Existing aliases retain
their documented limits; arbitrary repositories use Hub GGUF `context_length`
when supplied. If absent, the advertised ceiling is unknown, not invented.
RAM estimates for arbitrary models are approximate; architecture and KV cache
shape affect actual usage. `Limits` and `Agent.experimental()` below remain
available, but cannot create RAM, extend GitHub quotas, or add backend support.

```bash
gh-inference models Qwen3
gh-inference variants Qwen/Qwen2.5-0.5B-Instruct-GGUF
gh-inference run "What is recursion?" --model qwen0.5b --track-time
```

For source development, `pip install -e '.[dev]'` installs this checkout. The
legacy `gh_ai_runner` import remains supported; new code can use `gh_inference`.

## Publishing

The `publish.yml` workflow builds and tests the wheel before publishing to PyPI
using GitHub OIDC, with no stored PyPI token. Configure the PyPI trusted publisher
for owner `TanishC4444`, repository `GH-INFERENCE`, workflow `publish.yml`, and
environment `pypi`, then run it with `publish=true` or publish a `v*` release.
You can also publish by updating `.github/release.json` on `main` to the package
version; the workflow checks that they match before building and uploading.
Each version is immutable on PyPI; bump `gh_ai_runner.__version__` for a release.

New repositories are **private by default**. Existing visibility is never changed.
Actions must be enabled. Private Actions usage consumes included minutes and can
incur charges. Use a dedicated repository: the package manages and can overwrite
`run_inference.py` and `.github/workflows/inference.yml` there.

## Experiments: remove project ceilings

```python
from gh_ai_runner import Agent

agent = Agent.experimental(
    model="qwen0.5b",
    n_ctx=65536,          # Deliberately above this model's advertised 32K.
    max_tokens=None,     # Ask the runtime to use remaining context for the answer.
    track_time=True,
)
print(agent.run("Explain context windows."))
print(agent.last_result.usage)
```

`Agent.experimental()` uses `Limits.unrestricted()`, `trim_history=False`, and
`max_history_turns=None`. It removes our advertised model/output checks, estimated
RAM/disk rejection, temperature bounds, prompt-character cap, batch-count cap,
formatting reserves, cache-size threshold, and project workflow-input budget.
Default context remains 8K (2K for TinyLlama) unless you set `n_ctx`; default answer
length remains 512 unless you change `max_tokens`. All computation stays on GitHub.

This permits an attempt; it does not guarantee that the backend can execute it.
A bigger `n_ctx` allocates a bigger window. Sending more messages alone does not.
Going beyond a model's trained/advertised length may require positional scaling
and can produce poor answers. Disabling memory checks cannot create memory.

### Edit individual rules

```python
from gh_ai_runner import Agent, Limits

limits = Limits.unrestricted(
    check_resources=True,
    max_prompts=100,
    max_cached_model_gb=6,
    job_timeout_minutes=90,
)
agent = Agent.experimental(limits=limits, n_ctx=16384, max_tokens=2048)
```

`Limits()` preserves standard checks. `Limits.unrestricted(**overrides)` disables
project ceilings and applies your overrides. Both accept every field below. Pass
the configuration to `Agent`, `GitHubAIRunner`, `ai_call`, or an individual
`InferenceRequest`/`submit()`; request settings override runner defaults.
Configurations are immutable: make a new one or use `dataclasses.replace()`.

| `Limits` field | Standard default | Meaning / override |
| --- | --- | --- |
| `enforce_model_limits` | `True` | Advertised context and recorded output ceilings; `False` permits attempts beyond them. |
| `enforce_output_budget` | `True` | Reject an explicit answer limit at least as large as `n_ctx` and reserve the answer during trimming; `False` delegates answer-space handling to the runtime. |
| `check_resources` | `True` | Estimated RAM/disk checks on client and worker; `False` lets allocation/downloads proceed to real limits. |
| `max_prompt_chars` | `60000` | Current prompt plus instructions; `None` removes the project check. |
| `max_workflow_input_chars` | `65000` | Serialized workflow-input budget; `None` removes this budget. GitHub's service limit remains. |
| `max_prompts` | `32` | Prompts per `run_many()`; `None` removes the count cap. |
| `min_temperature`, `max_temperature` | `0.0`, `2.0` | Either bound can be `None`; values must still be finite. |
| `context_reserve_tokens` | `128` | Conservative extra allowance; can be `0`. |
| `message_overhead_tokens` | `32` | Formatting allowance per message; can be `0`. Actual formatting still uses tokens. |
| `max_cached_model_gb` | `6.0` | Cache files below this decimal-GB threshold; `None` permits all sizes, `0` skips caching. |
| `runtime_overhead_gb` | `1.5` | Estimated RAM above weights/context. |
| `disk_reserve_gb` | `2.0` | Additional estimated free disk. |
| `ram_per_1k_ctx_gb` | `None` | Use catalog's estimate; a number overrides it. Changing estimates does not change real allocation. |
| `job_timeout_minutes` | `30` | Remote timeout; unrestricted preset uses `360`. Hosted GitHub jobs have a six-hour service maximum. |
| `artifact_retention_days` | `3` | Requested result retention, subject to repository policy. |
| `download_timeout_seconds` | `30` | Model-download socket timeout. |
| `provider_timeout_seconds` | `300` | Legacy hosted-provider request timeout. |
| `refiner_min_tokens` | `128` | Minimum prompt size considered for refinement. |
| `refiner_max_tokens` | `512` | Rewrite ceiling, also bounded by original length and available context. |
| `refiner_context_fraction` | `0.25` | Context fraction used to plan the rewrite allowance. |

The unrestricted preset sets the three enforcement flags to `False`; optional
prompt, workflow-budget, batch, temperature, and cache-size limits to `None`;
formatting reserves to `0`; and remote timeout to `360`. Other defaults remain.
The experimental agent factory additionally disables trimming and turn-count caps.
Explicit arguments to the factory override those choices.

### Model loading and sampling

```python
agent = Agent.experimental(
    n_ctx=65536,
    max_tokens=4096,
    model_options={
        "n_threads": 2,
        "n_batch": 256,
        "rope_scaling_type": "yarn",
        "rope_freq_scale": 0.5,
        "yarn_orig_ctx": 32768,
        "yarn_ext_factor": 1.0,
    },
    generation_options={"top_k": 40, "top_p": 0.9, "repeat_penalty": 1.1, "seed": 42},
)
```

`model_options` forwards JSON-compatible keywords to
[`llama_cpp.Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/),
including CPU threads, batch sizes, KV settings, and positional scaling.
`rope_scaling_type` accepts a backend integer enum or a name such as `"yarn"` or
`"linear"`. These are experimental settings, not verified 64K quality claims.
Increasing context does not retrain the weights. Qwen 7B/14B still apply their
published factor-4 YaRN recipe above 32K unless you override it. Beyond advertised
lengths, choose and evaluate scaling yourself. Other entries keep GGUF metadata.

`generation_options` forwards sampling keywords to `create_chat_completion`,
including `top_k`, `top_p`, `min_p`, penalties, `seed`, and `stop`. It applies to
main answers; refinement uses separate deterministic sampling. Unknown keywords
are rejected by the installed backend. Use named `n_ctx`, `max_tokens`, and
`temperature` arguments for those values. Extra-option dictionaries cannot replace
the managed model path, CPU/RAM placement, messages, or nonstreaming text contract.

`max_tokens=None` requests remaining context. Standard model-limit enforcement
still applies a recorded publisher output ceiling; unrestricted mode does not.
The runtime can shorten an explicit answer allowance to remaining space and stop
early at EOS. Input exceeding allocated `n_ctx` is rejected by llama-cpp-python.
Custom policies/options are sent to the worker and covered by request checksums.

### Connection and waiting

```python
from gh_ai_runner import Agent, GitHubAIRunner, Limits

runner = GitHubAIRunner(
    limits=Limits.unrestricted(job_timeout_minutes=90),
    completion_timeout=5500,
    run_discovery_timeout=300,
    poll_interval=10,
    http_timeout=60,
    artifact_timeout=120,
    workflow_registration_timeout=120,
)
agent = Agent.experimental(runner=runner, limits=runner.limits)
```

Runner defaults, all in seconds: `completion_timeout=1800`,
`run_discovery_timeout=180`, `poll_interval=8`, `http_timeout=30`,
`artifact_timeout=60`, `workflow_registration_timeout=60`. Values must be finite
and positive; there is no project upper bound. HTTP timeout applies per API
request. Discovery/completion/artifact deadlines apply to their own phases.
`agent.run(prompt, timeout=...)` bounds waiting/result retrieval after dispatch,
not repository preparation or model execution. A client timeout does not cancel
the remote job. Increase client waiting alongside longer remote job timeouts.

### External limits remain

- [GitHub workflow inputs](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onworkflow_dispatchinputs)
  have a 65,535-character service limit including history, configuration, escaping,
  and JSON overhead. Characters are not tokens. This version has no alternate
  large-prompt transport, even when model context is 128K or larger.
- [GitHub Actions limits](https://docs.github.com/en/actions/reference/limits)
  include the six-hour hosted-job maximum, account minutes, concurrency, storage,
  and API limits. A larger configured timeout cannot extend the service maximum.
- Real RAM/disk, CPU time, backend numeric ranges, supported architectures and
  scaling, and model quality remain constraints. Allocation, disk-full, timeout,
  and worker-termination failures are possible when resource checks are disabled.

The curated 16B catalog is not a mathematical parameter-count cap; larger entries
need verified GGUF download metadata and appropriate workers. Valid input types,
checksums, and job identity checks remain required in both presets.

## Choose a model or role

```python
agent = Agent(model="qwen", instructions="You are a concise coding assistant.")
answer = agent.run("Write a function that merges overlapping intervals.")
```

The default `qwen0.5b` is a roughly 491 MB Q4_K_M starter model. Larger models can
provide better answers but take more download time, RAM, and CPU time.

| Model keys | Parameter sizes |
| --- | --- |
| `qwen0.5b`, `tinyllama`, `llama`, `qwen` | 0.5B, 1.1B, 1B, 1.5B |
| `gemma2`, `qwen3b`, `phi3`, `deepseek` | 2B, 3B, 3.8B, 1.5B |
| `qwen7b`, `qwen-coder7b`, `llama8b`, `gemma9b` | 7B, 7B, 8B, 9B |
| `mistral12b`, `qwen14b`, `qwen-coder14b`, `deepseek-coder16b` | 12B, 14B, 14B, 16B |

All catalog entries are prequantized GGUF files. Their sizes and SHA-256 hashes
are recorded in `models.py`; new downloads must match both before loading.

## Reuse one loaded model

```python
answers = agent.run_many([
    "Explain a Python list.",
    "Compare it with a tuple.",
    "Give a small example using both.",
])
for answer in answers:
    print(answer)
```

By default, up to 32 prompts run sequentially in **one GitHub job**. Change
`Limits.max_prompts` to raise/remove that cap. The full quantized model
loads into CPU RAM once and remains loaded for all those prompts and refinement
steps. Memory mapping and GPU offloading are disabled; memory locking is requested,
subject to the worker OS's permissions.

Separate `run()` calls create separate workers. Hosted GitHub workers cannot keep
RAM between jobs. Group prompts with `run_many()` to avoid repeated model loading.
GitHub caches downloaded weights and the Python environment across jobs, although
caches can be evicted. Model cache keys include the expected weight checksum.
Setup prefers official prebuilt CPU packages so a new worker normally avoids
compiling the inference runtime; source compilation is a fallback.

By default, files at least 6 GB skip weight caching because uploading their archive can need
another model-sized copy on the worker's limited disk. They still load once per
`run_many()` job, but a new job downloads them again.
Set `max_cached_model_gb=None` to permit all sizes, or `cache=False` on `Agent`
to skip weight caching. Unrestricted mode permits all sizes; cache archives can
still fail if the worker's disk or GitHub cache storage is too small.

## Timing

```python
agent = Agent(track_time=True)
print(agent.run("What is a B-tree?"))
print(agent.last_result.timings)
```

The return value stays plain text. Optional details are in `agent.last_result`
and `agent.last_results`: token usage, download/load/generation timing, weight-cache
hits, and within-job model reuse. `agent.last_job` exposes the Actions URL and
low-level status/cancellation handle.

`total_seconds` measures worker time for that prompt, including shared setup on
the first prompt. `remote_seconds` measures the full client operation, including
repository preparation, queuing, waiting, and result retrieval. For `run_many()`,
the whole-job remote duration appears on every result; do not sum it.
Override tracking per call with `agent.run(prompt, track_time=True)`.

## Controlled writing comparison

```bash
python examples/writing_benchmark.py --output writing-results.json
```

With `GH_AI_TOKEN` set, this compares Magnum v4 12B, Gemma 4 12B, Gemma 3 12B,
Gemma 3 Glitter 12B, and Qwen3.5 9B on strict dialogue, rhetoric, and JSON-scene
prompts. Five independent jobs reuse one loaded model each for three independent
samples. The benchmark explicitly configures a 16 GB runner budget, 4K context,
768 output tokens, four CPU threads, fixed sampling settings, and no automatic
repair or grammar constraints. It requires a repository whose workers really
have 16 GB; the budget itself does not upgrade hardware.

The report preserves prompts, pinned GGUF revisions/checksums, full raw samples,
rule violations, worker resources/runtime version, and Actions URLs. Mechanical
constraint scores do not measure literary quality, and one sample per case is
exploratory evidence. A loading/runtime failure is not scored as bad writing.
Large model downloads and runtime preparation can dominate cold job latency.

Worker environments are pinned to llama-cpp-python 0.3.36 with a versioned cache
key. New architectures require a compatible runtime, not just a GGUF filename.
`model_options={"chat_template_options": {"enable_thinking": False}}` binds that
option to the embedded default template, allowing supported reasoning models to
produce final artifacts without requesting reasoning. It does not strip or
rewrite output; unsupported templates fail explicitly. Currently only the boolean
`enable_thinking` template option is accepted.

`examples/story_turn_contract.json` proposes the next input/output boundary:
application-owned identities/routing, visibility-filtered context, generated content,
strict validation, and a committed event. It is a design proposal, not an
implemented persistent-worker service.

## Efficient draft, review, and revision

```python
from gh_inference import Agent

agent = Agent(model="qwen0.5b", max_tokens=None, track_time=True)
answer = agent.review("Write a scene between a cautious pilot and an impatient scientist.",
                      revisions=2)
```

This executes writer → reviewer → writer → reviewer → writer in **one job with
one model load**. Reviews receive the latest draft. Revisions receive that draft
and its review; obsolete drafts and reviews are not automatically accumulated.
The method returns the final answer. `last_results` contains every step.

For directed handoffs, multiple reviewers, or sharing selected outputs, use a plan:

```python
outputs = agent.pipeline("Write a tense two-character scene.", [
    {"agent": "writer", "instruction": "Draft the scene.", "inputs": []},
    {"agent": "dialogue-critic", "instruction": "Check distinct character voices.", "inputs": [0]},
    {"agent": "continuity-critic", "instruction": "Check facts and continuity.", "inputs": [0]},
    {"agent": "writer", "instruction": "Revise using both reviews.", "inputs": [0, 1, 2]},
])
```

Step indices start at zero. An omitted `inputs` selects the preceding step; `[]`
selects none. References must name earlier steps. Repeated indices are forwarded
only once, in the supplied order. A reviewer only receives selected source outputs,
not other reviewers' replies. To broadcast a draft, let several steps reference it;
to gather feedback, select those review indices on the receiving step.

The client sends the task and plan once. Intermediate outputs remain in worker
memory; source blocks are formatted once and reused for recipients. Shared source
text is placed before varying role/instruction text to enable prefix reuse between
recipients of the same draft. No workflow
launch, artifact download, or client/GitHub round trip occurs between steps.
Each model call must still process its supplied context; references do not eliminate
that computation. Turns execute sequentially, with the same loaded model and
backend prefix reuse. These methods do not create separate per-agent KV caches or
persistent multi-runner actors. Step names identify roles, not separate models.

Inspect `last_results[i].agent_id`, `.step_index`, `.input_steps`, `.handoff_chars`,
and `.handoff_bytes` for routing diagnostics, plus `.usage.prompt_tokens` for actual
whole-input token usage. Handoff size measures the selected in-memory payload,
including source labels; it is **not network traffic**. With timing enabled,
`.timings.routing_seconds` measures preparation of each step's handoff.
Whole-job resources are shared across results; count them once. Existing Agent
conversation history is neither sent to a plan nor replaced by it. No automatic
summarization is performed, and the shared task plus selected inputs must fit the
configured prompt/context limits. `Limits.max_prompts` bounds the total step count.

## Faster agent discussions on one worker

```bash
gh-inference discuss "Improve this inference library" --agents 20 --rounds 3 --topology ring
```

This runs 60 logical turns in **one GitHub job**, with one model kept in RAM.
Messages pass directly between rounds inside the worker. It avoids 59 workflow
launches and their setup, polling, and result downloads. Agents take turns on
one shared model; this mode does not provide 20 simultaneous inference workers.
Whether it beats independent workers in elapsed time depends on prompt/output
length and available parallel hardware. It is intended to reduce orchestration
and repeated computation for small, closely interacting agents.

```python
from gh_inference import Agent, Limits

agent = Agent(model="qwen0.5b", max_tokens=None, track_time=True,
              limits=Limits(max_prompts=60))
rounds = agent.discuss("Improve this inference library", agents=20, rounds=3,
                       topology="ring")
print(rounds[-1])  # Last round's 20 replies, in agent order.
```

`topology="all"` (the Python default) shares the previous round's replies with
every agent. `topology="ring"` shares only the previous neighbor's reply, reducing
peer-message entries per later round from N² to N. This changes the discussion's
information flow; it is an explicit option. Peer replies default to 240 characters.
Each agent has an identity, and no agent sees partially completed current rounds.
The existing conversation on the Agent is neither read nor replaced by a discussion.
`last_results` holds every turn in round order. Whole-job resources are shared
across all results and must be counted only once.

Repeated generation now lets llama.cpp reuse matching prompt prefixes rather than
forcing a reset before each reply. The backend compares exact tokens and discards
nonmatching suffixes; this is computation reuse, not implicit conversation memory.
See the [backend's prefix matching implementation](https://github.com/abetlen/llama-cpp-python/blob/v0.3.16/llama_cpp/llama.py#L828).
The worker also caches repeated token-count calculations within a turn. Shared
model download/load time is attributed only to the first turn.

Both `discuss` and `stress` accept `--max-tokens auto` for dynamic output length,
subject to remaining context, model limits and job timeout. Long responses can
increase latency; dynamic length is not itself a speed optimization.

## Twenty-agent interaction stress test

```bash
gh-inference stress --agents 20 --rounds 3 --output stress-results.json
```

Use `--topology ring` to reduce peer traffic while retaining independent parallel
workers; the default remains all-to-all. Or use `stress_test()` from Python. One command compares `qwen0.5b`, `llama`,
`qwen`, and `tinyllama`: five agents per model, 20 concurrent client calls per
round, three synchronized rounds, 60 remote jobs total. Each round broadcasts
all successful agents' previous replies to all agents, capped at 240 characters
per reply. Agents critique peers and propose improvements. Conversation is
bounded to the previous round; increase `rounds` for longer interaction.
GitHub account concurrency and available capacity determine actual simultaneous
workers. These jobs consume your Actions allowance.

The JSON report saves answers, errors, Actions links, settings, and per-model
summaries after each round. It separates generation throughput (completion tokens
per second, including prompt processing) from client latency (including queue and
setup), process CPU seconds, and process peak memory. CPU seconds include all
inference process threads, model loading, and refinement; they exclude workflow
installation/cache steps. `Agent.last_result.resources` exposes these worker
counters. Resources describe the entire job: for `run_many()`, all results share
them, so count them once. Old workers without CPU counters report `None`.

Hosted worker energy is unavailable here. `--estimated-worker-watts 40` optionally
calculates assumed constant watts × inference-worker wall seconds / 3600, labeled
**estimated Wh**. It excludes provisioning, dependency installation, cache transfer,
and data-center overhead; it is neither measured power nor total job energy.
No default wattage is invented.

Use `--models qwen0.5b llama` to choose candidates. Compare both failure rates and
throughput; the benchmark does not score answer quality or establish a universal
best model. It resets model RAM every job, so later rounds can reuse downloaded
weights but do not measure resident-model serving speed. Use `run_many()` when
measuring repeated inference within one loaded model.

## Optional prompt refinement

```python
agent = Agent(refine_prompt=True, track_time=True)
print(agent.run("Your detailed task and constraints ..."))
print(agent.last_result.refinement.tokens_saved)
```

Refinement runs on the **same loaded model on GitHub**, not a separate service or
second model. Short prompts are skipped. A rewrite is accepted only when it uses
fewer main-model tokens, finishes normally, and preserves protected numbers, URLs,
double-quoted literals, and code. Otherwise the original prompt is retained.

It can reduce the main generation's input processing, but adds its own tokens
and time. Total usage includes that work; `last_result.refinement.usage` reports
it separately. Shorter input does not shrink model weights or automatically reduce
the allocated context cache. Literal checks do not guarantee unchanged meaning.

## Conversation and limits

Conversation is remembered on the agent object and passed to subsequent GitHub
jobs. `run_many()` manages it within the same worker. No prompt/output history is
automatically written to disk by `Agent`. Use `remember=False` for independent
prompts and `agent.reset()` to forget previous turns.

Older conversation turns are dropped as needed to leave room for the new prompt
and answer. Current prompt text is never silently truncated. The main options are:

| Option | Default | Limit |
| --- | --- | --- |
| `max_tokens` | 512 | Positive integer or `None`; standard policy checks output/context ceilings |
| `n_ctx` | 8,192; TinyLlama: 2,048 | Positive integer; advertised ceiling and resource checks are editable policies |
| `temperature` | 0.7 | Finite number; standard bounds are 0 to 2 |
| `max_history_turns` | 6 | Any positive integer, or `None` for no exchange-count cap |
| `trim_history` | `True` | Set `False` to retain all exchanges and pass overflow to the model runtime |

Context includes instructions, history, formatting, and the answer. Worker
tokenization supplies actual usage; context planning includes a conservative
formatting allowance. The model can stop before reaching `max_tokens`.

### Context ceilings

The old blanket 8,192-context and 4,096-output caps are removed. Standard starting windows are
8,192 total tokens, except TinyLlama, whose supported maximum is only 2,048.
These advertised ceilings are enforced by standard policy and bypassable with
`enforce_model_limits=False`. They describe support, not a promise that a worker has enough
RAM or CPU time to use them. The catalog includes each limit's publisher source
in `MODELS[key]["context_source"]`.

| Model keys | Maximum total context | Separate published output ceiling |
| --- | ---: | ---: |
| `tinyllama` | 2,048 | — |
| `gemma2`, `gemma9b` | 8,192 | — |
| `qwen0.5b`, `qwen`, `qwen3b` | 32,768 | 8,192 |
| `qwen7b`, `qwen14b` | 131,072 | 8,192 |
| `qwen-coder7b`, `qwen-coder14b` | 131,072 | — |
| `llama`, `llama8b`, `phi3`, `deepseek` | 131,072 | — |
| `mistral12b`, `deepseek-coder16b` | 131,072 (128K) | — |
| `glitter-12b`, `rocinante-x-12b` | 131,072 | — |
| `gemma-4-12b`, `gemma-4-styletune-12b` | 262,144 | — |

`—` means no separate output ceiling is recorded; the available total context
still bounds generation. It does not mean unlimited or tested long-form output.
Sources include the [Qwen 0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[Qwen 7B long-context recipe](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts),
[Gemma 2 model card](https://ai.google.dev/gemma/docs/core/model_card_2), and the
individual publisher links in the catalog.

```python
agent = Agent(model="qwen0.5b", n_ctx=32768, max_tokens=8192)
print(agent.run("Explain the design in detail."))
```

The answer allowance shares the window with instructions, history, and your prompt.
Increasing context increases RAM usage even if the current prompt is short. The
worker automatically enables Qwen's factor-4 YaRN settings above 32,768 tokens for
the 7B/14B Qwen and Qwen Coder entries. Other models retain the scaling encoded in
their GGUF files. Long-context quality and peak RAM are not verified at every ceiling.

### When a conversation grows past its window

A 32K model can participate in a conversation totaling 64K tokens across many
calls. It can only receive one window's worth on any given call. This agent keeps
the instructions and current prompt, reserves the requested answer allowance,
and removes the oldest complete user/assistant pairs until the request fits.
It does not automatically summarize or retrieve the removed messages. Details
that appeared only in those messages are no longer available to the model.

The default history cap is also six previous exchanges, even when context has
space. Increase `max_history_turns` or set it to `None` to retain more exchanges; the token
budget and workflow transport can still trim them earlier. `n_ctx=32768` must be
set explicitly to use a 32K window instead of the 8K default.

```python
agent = Agent(model="qwen0.5b", n_ctx=32768, max_history_turns=32)
print(agent.run("Continue our discussion."))
print(agent.last_result.dropped_history_turns)  # history removed to fit context/transport
```

With the default trimming policy, one oversized current prompt is rejected, rather than cut silently. A 64K-token
prompt cannot fit an allocated 32K window. Standard policy rejects allocations
above the advertised maximum; experimental mode allows those attempts to reach
the runtime.

### Experiment without automatic history removal

```python
agent = Agent(
    model="qwen0.5b",
    n_ctx=8192,             # Try 16384 or 32768 in a separate experiment.
    trim_history=False,
    track_time=True,
)
print(agent.run("Remember this fact: my project is called Cedar."))
print(agent.run("What is my project called?"))
print(agent.last_result.usage)
```

`trim_history=False` disables all automatic history removal: the exchange-count
cap, token-budget trimming on GitHub, and client-side transport trimming.
`max_history_turns` is ignored in this mode. Completed exchanges stay on the agent
until you call `reset()` or discard the object; `remember=False` still disables
conversation memory entirely. This does not restore messages removed earlier.

The full retained history is passed to llama-cpp-python without our pre-generation
fit check. It does **not** enlarge `n_ctx`, bypass model ceilings or RAM checks, or
promise that the runtime accepts overflow. The runtime may reject an oversized
input or shorten generation to the remaining context. Use `Agent.experimental()`
or custom `Limits` to bypass the separate advertised-ceiling/resource policies.
Check the Actions job/logs
via `agent.last_job.actions_url` if a call fails. A failed call does not replace
the client's previously retained conversation or append its failed prompt.

GitHub's workflow-input transport limit still applies. When it is exceeded, the
call fails instead of dropping history, before starting a new worker. Setting a
32K context and sending 64K of messages therefore does not create 64K memory.
For controlled comparisons, start fresh agents with the same prompts, disable
refinement (the default), and vary `n_ctx` while keeping answer length fixed.

Standard policy limits prompt plus instructions to 60,000 characters, and the complete
workflow-input payload must fit GitHub's limit. Oversized jobs are rejected before
provisioning. These are **separate transport limits**, not token limits: a model
can allow 128K context while this version cannot transmit a full 128K-token prompt
or history through workflow inputs. With `trim_history=True`, history may be trimmed to satisfy transport
before it reaches the model. Workflow timeout and client completion wait default
to 30 minutes; both are editable as described above.
A client timeout does not automatically cancel a job; use `agent.last_job` when
you need to inspect or cancel it.

Tools, file access, Python callbacks, and shell execution are deliberately out of
scope. This version is text in, text out.

## Larger workers

The default client budget is 8 GB RAM and 14 GB workspace disk. GitHub lists 8 GB
RAM for standard private Linux workers and 16 GB for public ones. See
[GitHub runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

```python
from gh_ai_runner import Agent, GitHubAIRunner

runner = GitHubAIRunner(
    runner_label="your-linux-32gb-runner",
    runner_ram_gb=32,
    runner_disk_gb=40,
)
agent = Agent(model="deepseek-coder16b", runner=runner)
print(agent.run("Explain this algorithm."))
```

The runner label must already be available to your repository; changing budgets
does not purchase hardware. To deliberately create a public standard runner,
configure a new `repo_name`, `private=False`, and `runner_ram_gb=16`. Existing
visibility is never changed automatically.

RAM/disk estimates are checked before dispatch and available resources are checked
again on the worker. Large CPU models can be slow even when they fit.

Quantization shrinks weight storage, not the parameter count. Idealized 16B weights
take 32 GB at 16 bits or 8 GB at 4 bits; actual Q4_K_M mixes precisions and includes
scales/metadata. The catalog's 16B file is 10.36 GB. RAM also needs context/KV cache
and computation buffers. See [llama.cpp quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).

## Existing APIs

`ai_call(token, prompt, ...)` remains a blocking text-returning helper. Explicit
`track_time=True` returns an `InferenceResult` with `.output` and `.timings`.
`GitHubAIRunner()` also reads `GH_AI_TOKEN` automatically and retains submission,
batch/comparison, polling, cancellation, retry, resume, and hosted-provider APIs.

`submit_batch()` launches independent workflows; `Agent.run_many()` reuses one
worker/model. Only failed workflow conclusions are automatically retried when
requested by `RetryPolicy`, avoiding duplicate work after uncertain dispatches.

## Privacy and verification

Prompts, history, and answers are sent to GitHub. Results are Actions artifacts,
retained for three days. Public repository permissions and Actions logs matter.
Checksums verify job identity, requests, output, and structured agent results; they
are not signatures protecting against a compromised worker/repository.

Test/build CI runs on GitHub. The manual **GitHub model smoke test** workflow also
downloads a real small model and checks that two prompts reuse it in RAM. Client
orchestration tests use simulated API/model responses. No model runtime is installed
by this package on the client.

MIT license. See [LICENSE](LICENSE).

### Resident local sessions (0.6.1)

`ResidentSession` runs on the host where it is called and requires a separately installed `llama-cpp-python`. It retains one model's weights while resetting token context and sending only the current system and user messages on every call. It does not dispatch remote work or share weights between GitHub jobs. Calls are serialized; use separate sessions/processes for independent parallelism. Input token counts plus requested output and a conservative template reserve are checked before generation, with no silent trimming. Optional `response_schema` constrains JSON decoding; application validation is still required.

```python
from gh_inference import ResidentSession
with ResidentSession("qwen3b", n_ctx=8192) as session:
    text, diagnostics = session.complete("Review this scoped evidence...", system="Be precise.", max_tokens=800)
    other, diagnostics = session.complete("A separate task...", max_tokens=800)
```

Local worker defaults now use all available CPU cores. GPU offload is opt-in through `model_options={"n_gpu_layers": -1}` and requires a compatible runtime and GPU host; this does not provision a GPU on standard Actions runners.
