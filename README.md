# gh-inference

Run quantized language models on GitHub Actions from Python. Your computer sends
prompts and receives text; model weights and inference stay on the GitHub worker.

## Quick start

Requires Python 3.9+ and a GitHub account with Actions enabled.

```sh
pip install --upgrade gh-inference
```

Create a classic GitHub token with `repo` and `workflow` scopes, then set it in
your environment. Keep it out of source control.

```sh
export GH_AI_TOKEN="your-token"       # macOS / Linux
```

On PowerShell, use `$env:GH_AI_TOKEN="your-token"`.

```python
from gh_inference import Agent

agent = Agent()
print(agent("Explain recursion in one sentence."))
```

The first call creates or reuses a private `ai-inference-runner` repository,
installs its worker, and waits for an answer. Startup includes runner scheduling,
dependency installation, and model download; expect a cold start.

## Common tasks

```python
# Keep a conversation.
print(agent("Give me a short example."))
agent.reset()

# Answer several prompts with one model load.
answers = agent.run_many(["What is a list?", "What is a tuple?"])

# Choose a model and limit the answer.
small = Agent(model="qwen0.5b", max_tokens=128)
print(small("Name three sorting algorithms."))
```

Or use the terminal: `gh-inference run "Explain recursion briefly."`

## Before you run

- Prompts and results go to GitHub. Private Actions usage can consume minutes or incur charges.
- Use a dedicated worker repository: the client manages its inference script and workflow.
- Models must fit the runner's RAM and use a supported GGUF architecture.
- Image input is experimental. Local `ResidentSession` is optional and needs its own model runtime.

## Next steps

[Models and larger workers](https://github.com/TanishC4444/GH-INFERENCE/blob/main/docs/advanced.md#models-and-every-published-gguf-variant) ·
[Advanced guide](https://github.com/TanishC4444/GH-INFERENCE/blob/main/docs/advanced.md) ·
[Troubleshooting](https://github.com/TanishC4444/GH-INFERENCE/blob/main/docs/troubleshooting.md) ·
[Verified package results](https://github.com/TanishC4444/GH-INFERENCE/blob/main/verification-results/pypi-0.6.3.json)

The legacy `gh_ai_runner` import remains supported.
[MIT license](https://github.com/TanishC4444/GH-INFERENCE/blob/main/LICENSE).
