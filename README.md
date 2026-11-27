> **Archived duplicate.** Continue in [TanishC4444/GH-INFERENCE](https://github.com/TanishC4444/GH-INFERENCE). This repository's source branches and release tags are preserved there under `history/github-ai/`. The old copy remains available read-only.

# github-ai

Historical source for the `gh-ai-runner` package. Current development and fixes
are maintained in **[GH-INFERENCE](https://github.com/TanishC4444/GH-INFERENCE)**,
published as `gh-inference`.

## Use the maintained package

```sh
pip install --upgrade gh-inference
```

```python
from gh_inference import Agent

# Set GH_AI_TOKEN in your environment first.
print(Agent()("Explain recursion in one sentence."))
```

Follow the [quick start](https://github.com/TanishC4444/GH-INFERENCE#quick-start)
for token setup, runtime behavior, costs, and usage limits. The legacy
`gh_ai_runner` import remains available in the maintained package.

[Historical documentation](docs/legacy-guide.md) ·
[Original local versions and histories](https://github.com/TanishC4444/GH-INFERENCE/tree/preservation/inference-originals-2026-11-26/originals) ·
[License](LICENSE)

This repository remains available for provenance and existing references.
