## 0.6.1

- Add stateless ResidentSession with loaded-weight reuse, token-budget checks and optional constrained JSON output.
- Use all available CPU cores by default; allow explicit GPU offload on compatible hosts.
- Preserve independent prompts and reject context overflow without truncation.

# Changelog

## 0.6.0

- Add built-in, commit-pinned Q4_K_M writing models: Glitter 12B, Gemma 4 12B,
  Gemma 4 StyleTune 12B, and Rocinante-X 12B.
- Accept Hugging Face repository and GGUF file links; add `add_model` for public
  custom HTTPS GGUF downloads with worker-side checksum and size verification.
- Add resident multi-agent discussion, routed pipelines and draft/review/revise
  loops with selective handoffs and reusable prompt prefixes.
- Expand compute diagnostics with CPU time, peak process memory, routing time,
  runtime version and handoff sizes; add stress and writing benchmarks.
- Support dynamic output budgets and native thinking-template controls; pin the
  GitHub inference runtime to llama-cpp-python 0.3.36.

## 0.5.0

- Publish as `gh-inference` with the `gh_inference` import and `gh-inference` CLI.
- Discover public Hugging Face GGUF repositories across model families, without
  a fixed catalog or quantization whitelist.
- Select exact variants and revisions; verify commit-pinned downloads by size
  and SHA-256, including complete split GGUF groups.
- Preserve the legacy import, resident model reuse, caching, timing and editable limits.
- Add GitHub-only verification, wheel checks and PyPI trusted publishing.

## 0.4.0

- Add a GitHub-only `Agent` facade: `run()` returns text and `run_many()` reuses one loaded model.
- Keep full quantized models resident in worker RAM across tasks, refinement, and tool steps in one job.
- Add optional prompt refinement using the same loaded model, with tokenizer-based savings and fallback checks.
- Add optional timing, prompt refinement, and bounded conversation history; defer tools and file access.
- Add a small Qwen 0.5B starter model; retain all existing model keys and APIs.
- Verify structured agent results as well as the main output.
- Add manual GitHub model smoke testing; no client-side inference backend or runtime dependency is shipped.

## 0.3.2

- Expand the Q4_K_M catalog to 15 models through DeepSeek Coder V2 Lite 16B.
- Add configurable Linux runner labels and RAM/disk budgets with worker preflight checks.
- Verify downloaded weight sizes and hashes; avoid weight-cache archives for large models.
- Document quantization, context memory, and the difference between model storage and RAM.

- Default newly created runner repositories to private; expose explicit public opt-in.
- Bound GitHub HTTP requests, surface API errors, and paginate workflow discovery.
- Validate finite numeric settings and context limits before provisioning.
- Preserve caller system messages and require output checksums on every result.
- Share timeout budgets with artifact retrieval and avoid retries after the deadline.
- Keep metadata errors sanitized and preserve existing directory permissions.
- Generate the worker model catalog from the client catalog and download weights atomically.
- Use one package version source and simplify documentation around the core inference flow.
- Remove stale committed build outputs and add Python 3.9/3.11/3.13 test/build CI.
- Keep optional local history failures from interrupting inference.

## 0.3.0

- Add concurrent batch submission and collection with `InferenceRequest` and `BatchJob`.
- Add side-by-side local model comparisons with `ComparisonJob`.
- Add canonical request and output SHA-256 verification.
- Add SQLite job and batch metadata, history listing, and cross-process job resume.
- Exclude prompts, outputs, GitHub tokens, and provider keys from persistent metadata.

## 0.2.0

- Add asynchronous `AIJob` handles with status, waiting, cancellation, and retries.
- Add exact workflow correlation and job-specific structured artifacts.
- Add token and runner resource telemetry.
- Add encrypted BYOK setup for OpenAI-compatible HTTPS providers.
- Preserve the blocking `ai_call()` API.
