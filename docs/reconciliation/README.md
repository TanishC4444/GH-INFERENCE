# Inference repository reconciliation

GH-INFERENCE is the maintained source for `gh-inference`. The older
`github-ai` repository retains the history of the `gh-ai-runner` package.
Both Python import names remain supported.

## Combined implementation

The reconciliation starts from GH-INFERENCE commit `224cae6` and brings in the
uncommitted experimental vision implementation from the desktop `github-ai`
checkout. It retains the newer `ResidentSession` API, current release metadata,
all available CPU cores by default, and the worker's configurable GPU option.
The library version remains 0.6.2 until a separate release is prepared; these
source changes are recorded under Unreleased in the changelog.

The image path includes matched, commit-pinned GGUF weights and projector
registration, checksum verification, an image-aware worker, a smoke-test example,
and offline regression coverage. Unsafe projector paths fail before metadata
requests, and missing projector checksum metadata produces a useful error.
The tests cover the image request followed by a text-only conversation, model
reuse, corrupt downloads, transport integrity, and both public import names.

Vision remains experimental. Offline tests use mocked models and downloads;
they do not establish compatibility with a real publisher's model/projector or
the accuracy of an image description. Image URLs are sent to GitHub and fetched
by the worker. Use only images suitable for that environment. The example in
`examples/vision_smoke.py` runs a real test when given compatible files and an
accessible image URL.

## Preserved experiments

The [desktop evidence](desktop-evidence/) contains the eight original local
JSON/text result files, preserved byte for byte. They are historical experiment
records, not a fresh benchmark of this reconciled version. Reports include their
original caveats and recovered or failed outcomes; energy figures are estimates.
The related writing guidance is retained under `skills/distinct-writing/`.

The older `gh-inference-source` and `gh-inference-fresh` snapshots contained the
same source files and older verification reports that had already been removed
from the maintained repository. They remain in the separate full local backup
rather than being restored as current release evidence.

## Validation

Run the offline suite and build from a clean environment:

```sh
python -m pip install -e '.[dev]' build twine PyYAML
python -m pytest -q
python -m build
python -m twine check dist/*
```

The existing CI matrix checks Python 3.9, 3.11, and 3.13. Keep model-runtime
installation on the execution host; installing the client does not require
`llama-cpp-python`. No PyPI release is implied by reconciling this source tree.

## Retiring old copies

Keep one active clone of GH-INFERENCE. The desktop `github-ai`, old
`gh-ai-runner:`, and the two older inference snapshots can be retired after the
reconciled branch and complete filesystem backup are verified. Keep the backup
outside any folder being removed. Local experimental output is not evidence
that a similarly named GitHub worker repository is unused.

Retain `ai-inference-runner`: it is the client's default worker repository.
Other worker/verification repositories require a caller and recent-activity
check before archiving. Do not delete the historical `github-ai` repository
while old package consumers or documentation may still depend on it.
