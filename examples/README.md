# Console tour

Run from the repository folder in VSCode's terminal:

```bash
python -m pip install --upgrade gh-inference
python examples/console_tour.py
```

The token prompt hides your input. The normal tour creates three GitHub jobs;
your computer runs the client, and GitHub runs every model.

| Command option | What it adds |
| --- | --- |
| `--full` | Four jobs for parallel requests and two-model comparison |
| `--cancel` | One additional job and a cancellation request |
| `--raw` | Complete JSON diagnostics instead of concise summaries |
| `--quantization Q8_0` | Select a published Q8 variant |
| `--repo my-test-runner` | Use a dedicated runner repository of your choice |

## Read the output

- **ANSWER** contains the model's text, with its line breaks preserved.
- **Tokens** separates input, output and total usage, including refinement work.
- **Download / Load / Generate / Refine** are worker timings.
- **Same loaded model reused** means a preceding prompt in this job already loaded it.
- **Weights restored from cache** means the downloaded files were present; a new worker still loads RAM.
- **Whole client operation** also includes setup, queuing and result retrieval.
- **LIVE RUN** links directly to the individual GitHub run while it is executing.

## Watch GitHub

For the default playground:

- [All inference runs](https://github.com/TanishC4444/gh-inference-playground/actions)
- [Inference workflow](https://github.com/TanishC4444/gh-inference-playground/actions/workflows/inference.yml)

These links require access to the private playground repository. The console
prints equivalent links for your account and selected repository.

Open a run and select its job. Expand the setup, cache, **Run inference on GitHub**
and **Upload structured result** steps to inspect logs. Completed runs have an
`ai-output-…` artifact containing `result.json` with the answers and diagnostics.

Queueing is normal. A completed job has historical logs rather than live progress.
The watcher only observes status; it does not start additional inference jobs.
