# Troubleshooting

Start with `Agent()` and the default small model before changing model or worker
settings. The client requires network access to GitHub; model discovery also
requests public Hugging Face metadata.

| Symptom | What to check |
|---|---|
| `Set GH_AI_TOKEN` | Set the environment variable in the same shell or process that runs Python. Restart an already running notebook kernel if needed. |
| GitHub authentication or permission error | Confirm the token is valid and has classic `repo` and `workflow` scopes. Account or organization policies can further restrict Actions and repository creation. |
| The first call takes a while | The worker must be scheduled, install dependencies, and download model weights. The caller waits for completion; no model server remains reserved between jobs. |
| Workflow is missing or disabled | Open the dedicated worker repository's Actions tab and check that Actions is enabled. `agent.connect()` prepares the files but does not start a model job. |
| RAM or disk budget error | Try the default model and a smaller context first. A larger model needs a runner with enough actual memory and disk; increasing the client's declared budget does not upgrade the hardware. |
| Too many GGUF matches | Use `list_quantizations("owner/repository")` and pass an exact `filename` to `Agent`. Discovery does not guarantee architecture compatibility with the pinned runtime. |
| An image request is rejected | Use a model registered with `add_vision_model` and exactly one HTTPS image URL. Images cannot be combined with prompt refinement. This API remains experimental. |

## See the worker run

```python
from gh_inference import Agent

agent = Agent(track_time=True)
try:
    print(agent("Explain a Python list briefly."))
finally:
    if agent.last_job is not None:
        print(agent.last_job.actions_url)
```

The Actions page shows the specific step that failed. When reporting an issue,
include the package version, model selection, error message, and a run URL only
when it is appropriate to share. Keep tokens and private prompts out of reports.

## Update the client

```sh
python -m pip install --upgrade gh-inference
python -c "import gh_inference; print(gh_inference.__version__)"
```

Use the same Python environment for installation and execution. See the
[advanced guide](advanced.md) for context policy, retry behavior, caching,
resource estimates, and local resident sessions.
