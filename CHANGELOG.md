# Changelog

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
